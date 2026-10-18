"""Regeneration loop: rubric judge -> regenerate-with-the-failure-stated.

This is the production feedback path. It exists because of a measurement, not a
preference: on 426 consensus-rejected sentences with MATCHED prompts, regenerating
with the failure stated beat judge-authored correction in EVERY defect stratum,
including the near-miss pool built to favour correction (11.6% vs 19.6%,
McNemar p=2.2e-4; confirmed by a second judge, then by the rubric judge against
two more). Correction is anchored to the rejected sentence, and the rejection was
usually about the content frame, so a minimal edit preserves the defect. We do not
correct. We regenerate and state what was wrong.

Policies that come from the measurements
----------------------------------------
1. ONLY structural/linguistic failures trigger regeneration. A `LEVEL` or
   `REGISTER` failure is valid Japanese that sits above (or beside) the target --
   regenerating it burns a call to re-roll vocabulary, when the right response is
   to re-label or accept. Filtering LEVEL/REGISTER out of reason_codes took arm A
   from 7.7% to 3.5% on the same sentences.

2. The failure is stated in full: which dimensions failed, the reason codes, and
   the judge's own words. Retry-with-context is the one mechanism we have measured
   to work twice (duplicates 43% -> 0%; the A/B win).

3. The original is immutable. This writes a NEW candidate with provenance
   (parent sentence, attempt, the reasons it was regenerated); nothing is
   overwritten. Corrections or re-evaluations create audit records.

4. Bounded. A sentence that needs more than MAX_ATTEMPTS is a discard, not a
   rescue -- generating a fresh sample is cheaper than iterating on a bad one.
"""
import os, re, json, time, collections, argparse, urllib.request, urllib.error
import rubric_judge as RJ
import guardrails as G

GEN = r"H:\jlpt\gen"
RUNS = r"H:\jlpt\runs"

# --- word pool + ledger ----------------------------------------------------
# Without these the loop collapses to templates: measured on 128 regenerations,
# only 86 distinct sentences came back (33% duplicates of another child) because a
# bare "write a different sentence" prompt lets the model re-use its favourite
# frame. generate2.py avoids this with pool routing plus a rotating ledger, so the
# regeneration path uses the same machinery instead of inventing a weaker one.
_LEX = _POOLS = _LED = _TARGETS = None


def _init_pools():
    global _LEX, _POOLS, _LED, _TARGETS
    if _POOLS is None:
        _LEX = json.load(open(os.path.join(GEN, "data", "lexicon.json"), encoding="utf-8"))
        _POOLS = G.build_pools(_LEX)
        _LED = G.Ledger()        # the constructor loads RUNS/ledger.json; no load() method
        _TARGETS = {t["grammar_point_id"]: t for t in
                    json.load(open(os.path.join(GEN, "data", "targets.json"),
                                   encoding="utf-8"))["targets"]}
    return _POOLS, _LED, _TARGETS


def draw_words(gp, n=4):
    """Required vocabulary for this point, least-used first."""
    pools, led, tg = _init_pools()
    t = tg.get(gp)
    if not t:
        return []
    need = G.pos_need(t["pattern"], t.get("meaning_en", ""), gp)
    pool = G.filter_pool(gp, pools.get(need) or pools["noun"])
    return led.draw(gp, pool, min(n, len(pool))) if pool else []


# which failures are worth regenerating?
REGEN_ON = ("BROKEN", "INCOHERENT", "WRONGPAT", "NUANCE")
# these are valid Japanese needing a re-label or a policy decision, not a re-roll
NO_REGEN = ("LEVEL", "REGISTER")

MODEL = "openai/gpt-6-luna"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 2

# Content-addressed prompt identity.
#
# A hand-maintained version string is a CONVENTION; a hash is EVIDENCE. A developer
# (me) can forget to bump PROMPT_VERSION, and the code then treats two different
# prompts as equivalent -- silently, which is the failure mode this whole mechanism
# exists to prevent. PROMPT_VERSION is kept as a human-readable label, but the
# identity is derived from the prompt source itself, so any edit invalidates it
# whether or not anyone remembered to bump anything.
PROMPT_VERSION = "regen-v1"

REGEN_SYS = ("あなたは日本語教材の例文を作る専門家です。JLPT N5 の短い例文だけを書きます。")

# the sampling temperature is drawn per call from this range; it is a configuration
# fact for identity purposes, not a per-call identity, so the RANGE is hashed
TEMP_RANGE = (0.7, 1.0)
MAX_TOKENS = 600


def prompt_sha256():
    """sha256 of the full REQUEST-BUILDING logic, truncated.

    Covers regen_prompt, build_user and the system message. Hashing only the template
    was not enough: build_user() constructs the actual user message from the targets and
    the required words, so a change there alters the request while regen_prompt's text
    stays identical. Hashing the sources means any edit -- wording, ordering,
    instruction count, even a comment inside the template -- changes the identity.
    """
    import hashlib
    import inspect
    import guardrails as G
    src = "\n".join([
        inspect.getsource(regen_prompt),
        "\n--BUILD_USER--\n",
        inspect.getsource(G.build_user),
        "\n--SYS--\n",
        REGEN_SYS,
    ])
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


# Bump when the SHAPE of the identity payload changes, so an old hash is never
# mistaken for a new one under a different field set. A bump INVALIDATES all prior
# config keys by definition: rows carrying an older schema are treated as `different`,
# never as matching, which is the migration policy -- there is no attempt to upgrade
# an old key to a new schema, because the fields that changed are exactly the ones the
# old key did not cover.
HASH_SCHEMA = 2


def data_sha256():
    """Identity of the curated inputs that shape generation.

    targets.json drives the pattern, frame and check_regex for every point; lexicon.json
    and the class files drive which words a point's pool contains. A pool or target edit
    changes what the model is asked for, so it must invalidate generation identity even
    though no prompt text changed. The transitive-pool change is a worked example: it
    altered 77 points' pools with zero edits to any prompt.
    """
    import hashlib
    h = hashlib.sha256()
    for name in ("targets.json", "lexicon.json", "noun_classes.json",
                 "noun_class_overrides.json"):
        p = os.path.join(GEN, "data", name)
        h.update(name.encode())
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def config_key(model=None, temperature=None):
    """CONFIGURATION identity of the generator: model, code, prompts, parameters, data.

    This answers "which generator made this?" It deliberately does NOT include the
    per-request inputs (the grammar target, the drawn words), so drawing different words
    does not invalidate the configuration -- that is what request_identity() is for.

    Deliberately does NOT include the judge: a judge change must not force regeneration,
    while a generation change must never reuse old output.

    Establishes configuration identity, NOT output reproducibility: identical
    configuration does not make a stochastic model deterministic.
    """
    import hashlib
    import json as _json
    payload = {
        "schema": HASH_SCHEMA,
        "model": model or MODEL,
        "prompt_sha256": prompt_sha256(),
        "data_sha256": data_sha256(),
        "prompt_version_label": PROMPT_VERSION,
        "temperature_range": list(TEMP_RANGE),
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": False},   # output-affecting API setting
        "response_format": "json",        # what we ask for and parse
    }
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def request_identity(item, required, model=None, temperature=None):
    """REQUEST identity: a fingerprint of the actual serialised request.

    Distinct from config_key on purpose. Three identities, three questions:

      config_key       which GENERATOR made this?      (model, code, prompts, data)
      request_identity what exact INPUT was sent?      (target, words, params)
      the output row   what came back?                 (the text, recorded as-is)

    Two runs can share a config_key and differ here, because the word draw differs.
    Keeping them separate means one changed draw does not invalidate the whole
    configuration, while a changed draw is still individually auditable.
    """
    import hashlib
    import json as _json
    body = {
        "model": model or MODEL,
        "system": REGEN_SYS,
        "user": regen_prompt(item, {"reason_codes": [], "reason": ""}, required),
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": False},
    }
    return hashlib.sha256(
        _json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def judge_config_key(judge_model):
    """Identity of the JUDGING configuration, separate from generation."""
    import hashlib
    import json as _json
    return hashlib.sha256(_json.dumps(
        {"judge_model": judge_model, "rubric": "rubric_judge-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]


def regen_prompt(item, verdict, required=()):
    """State exactly what was wrong, and ask for a fresh sentence -- not a fix."""
    # .get with defaults: request_identity() rebuilds this prompt from the same inputs
    # without a judge verdict, and a prompt builder crashing on a missing optional key
    # is a fragility that would surface as a KeyError mid-run.
    bad = ", ".join(verdict.get("reason_codes") or [])
    dims = ", ".join(verdict.get("failed_dimensions") or [])
    words = "、".join(v["text"] for v in required) if required else ""
    return (
        "次の日本語の例文は、JLPT N5 の教材としては不合格でした。"
        "同じ文法を使って、別の新しい例文を1つ書いてください。"
        "前の例文を直すのではなく、まったく別の文を作ること。\n\n"
        f"対象文法: {item.get('pattern','')}（{item.get('meaning_en','')}）\n"
        + (f"構造条件: {item['frame']}\n" if item.get("frame") else "")
        + (f"必ず使う語: {words}\n" if words else "")
        + f"\n前の例文: {item['japanese']}\n"
        + (f"問題点: {bad}  （観点: {dims}）\n" if (bad or dims) else "")
        + f"校閲者の指摘: {verdict.get('reason','')[:200]}\n\n"
        "守ること:\n"
        "- 1文だけ。20文字以内。です・ます体。\n"
        "- 対象文法を文の中心として実際に使うこと。\n"
        "- 前の例文と同じ語順・同じ語をなぞらないこと。\n"
        + ("- 「必ず使う語」をすべて文の中に入れること。\n" if words else "")
        + "- N5のやさしい語彙と漢字だけを使うこと。\n\n"
        '出力はJSONのみ: {"sentences":["例文"]}'
    )


def gen_one(item, verdict, key, required=(), temperature=0.8, model=None):
    prompt = regen_prompt(item, verdict, required)
    # `model` comes from --model. It used to be hardcoded to the module constant,
    # which silently ignored the flag: only --judge-model was honoured, so any
    # generator comparison run through this script compared luna against luna.
    # Reported by an external code review, verified here.
    body = {"model": model or MODEL, "temperature": temperature,
            "reasoning": {"enabled": False}, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": REGEN_SYS},
                         {"role": "user", "content": prompt}]}
    rq = urllib.request.Request(OR_URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=200))
    txt = r["choices"][0]["message"].get("content") or ""
    i, j = txt.find("{"), txt.rfind("}")
    out = []
    if i >= 0:
        try:
            out = [s for s in json.loads(txt[i:j + 1]).get("sentences", []) if isinstance(s, str)]
        except Exception:
            out = re.findall(r'"([^"]{2,60})"', txt[i:j + 1])
    return [s.strip() for s in out if s.strip()], r.get("usage", {})


def classify_provenance(rows, cfg, schema=None):
    """Sort existing output rows against the current generation config.

    Returns (matching, missing, invalid, incompatible, different) where each of the
    first three is a set of (parent, grammar_point_id) keys and `different` is a count.

    Extracted so each case can be tested directly rather than only through a paid run.
    Five cases, because they all force regeneration but mean different things:

      missing      -- nothing recorded; provenance is unverifiable
      invalid      -- a config field that cannot be compared at all; a defect
      incompatible -- a well-formed hash under a DIFFERENT hash schema; a stale key
      different    -- a well-formed current-schema config that is simply not this one
      matching     -- identical config, safe to reuse

    `incompatible` exists because a hash being well-formed hexadecimal says nothing
    about whether it was computed under the same payload shape. Without the schema
    check, an old key would be silently compared against a new one and could even
    collide by accident.
    """
    if schema is None:
        schema = HASH_SCHEMA

    def _cfg_ok(c):
        return (isinstance(c, str) and len(c) == 16
                and all(ch in "0123456789abcdef" for ch in c))

    matching, missing, invalid, incompatible, different = set(), set(), set(), set(), 0
    for d in rows:
        k = (d.get("parent"), d.get("grammar_point_id"))
        c = d.get("config")
        if c is None:
            missing.add(k)
        elif not _cfg_ok(c):
            invalid.add(k)
        elif d.get("config_schema") != schema:
            # well-formed, but not comparable: it was computed under another payload
            incompatible.add(k)
        else:
            matching.add(k + (c,))
            if c != cfg:
                different += 1
    return matching, missing, invalid, incompatible, different


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="sentences jsonl")
    ap.add_argument("--out", required=True, help="regenerated candidates jsonl")
    ap.add_argument("--verdicts", default=None, help="reuse existing verdicts")
    ap.add_argument("--model", default=RJ.DEFAULT_MODEL)
    ap.add_argument("--judge-model", default=RJ.DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key", default=None)
    ap.add_argument("--trust-legacy", action="store_true",
                    help="accept existing rows that record no config, instead of "
                         "regenerating them (unknown provenance is otherwise not a match)")
    a = ap.parse_args()

    key = RJ.load_key(a.key)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"input {len(rows)} sentences", flush=True)

    # --- judge ---------------------------------------------------------
    if a.verdicts and os.path.exists(a.verdicts):
        vd = {(r["grammar_point_id"], r["japanese"]): r
              for r in (json.loads(l) for l in open(a.verdicts, encoding="utf-8") if l.strip())}
        print(f"reusing {len(vd)} verdicts from {a.verdicts}", flush=True)
    else:
        vd, t0, cost, nod = {}, time.time(), 0.0, 0
        for k in range(0, len(rows), a.batch):
            b = rows[k:k + a.batch]
            try:
                res, usage, _ = RJ.judge_batch(b, model=a.judge_model)
                cost += float(usage.get("cost", 0) or 0)
                for src, v in zip(b, res):
                    if v is None: nod += 1; continue
                    vd[(src["grammar_point_id"], src["japanese"])] = v
            except urllib.error.HTTPError as e:
                print(f"  judge HTTP {e.code} {e.read().decode()[:140]}", flush=True)
            except Exception as e:
                print(f"  judge {type(e).__name__} {str(e)[:100]}", flush=True)
        print(f"judged {len(vd)} in {time.time()-t0:.0f}s  ${cost:.5f}  no_verdict {nod}", flush=True)

    # --- decide what to regenerate -------------------------------------
    todo, skipped = [], {"pass": 0}
    for r in rows:
        v = vd.get((r["grammar_point_id"], r["japanese"]))
        if not v:
            continue
        codes = v.get("reason_codes") or []
        if v["overall"] != "fail":
            skipped["pass"] += 1; continue
        if codes and all(c in NO_REGEN for c in codes):
            skipped[" ".join(codes)] = skipped.get(" ".join(codes), 0) + 1
            continue
        if any(c in REGEN_ON for c in codes):
            todo.append((r, v))
        else:
            skipped["other"] = skipped.get("other", 0) + 1
    print(f"\nto regenerate: {len(todo)}   not regenerating: {skipped}", flush=True)

    # --- regenerate ----------------------------------------------------
    # Provenance, classified. A single `unknown` catch-all hides three different
    # situations that all require regeneration but carry different diagnostic value:
    #   missing   -- no config field recorded at all
    #   invalid   -- a config field that is malformed, so it cannot be compared
    #   different -- a well-formed config that simply is not this one
    # Only `missing` is eligible for --trust-legacy; a malformed or mismatched field is
    # never an unverifiable claim to accept, it is a defect to surface.
    cfg = config_key(a.model)
    if os.path.exists(a.out):
        prev = []
        for l in open(a.out, encoding="utf-8"):
            try:
                prev.append(json.loads(l))
            except Exception:
                pass                          # unparseable line is not a resume candidate
        done, missing, invalid, incompat, diff_cfg = classify_provenance(prev, cfg)
    else:
        done, missing, invalid, incompat, diff_cfg = set(), set(), set(), set(), 0
    total_prev = len(done) + len(missing) + len(invalid) + len(incompat)
    if total_prev:
        print(f"  resume: {len(done)}/{total_prev} rows carry this exact config",
              flush=True)
        if diff_cfg:
            print(f"          {diff_cfg} rows are a DIFFERENT config "
                  f"-> regenerating", flush=True)
        if incompat:
            print(f"          {len(incompat)} rows were written under another HASH "
                  f"SCHEMA -> regenerating (not comparable, not merely different)",
                  flush=True)
        if invalid:
            print(f"          {len(invalid)} rows have a MALFORMED config "
                  f"-> regenerating (defect, not a claim to accept)", flush=True)
        if missing:
            print(f"          {len(missing)} rows record NO config (unknown provenance)"
                  f" -> regenerating", flush=True)
    if a.trust_legacy and missing:
        pct = 100.0 * len(missing) / max(total_prev, 1)
        print(f"  --trust-legacy: reusing {len(missing)}/{total_prev} rows "
              f"({pct:.1f}%) with UNVERIFIABLE provenance "
              f"(accepting, not verifying)", flush=True)
    elif a.trust_legacy:
        print(f"  --trust-legacy: nothing to accept (no rows with missing config)",
              flush=True)
    log = open(a.out, "a", encoding="utf-8")
    print(f"  config {cfg}", flush=True)
    n = 0; cost = 0.0
    for r, v in todo:
        key3 = (r["japanese"], r["grammar_point_id"], cfg)
        if key3 in done:
            continue
        # only MISSING config is eligible for the opt-in; a malformed or mismatched
        # config is a defect, never an unverifiable claim to accept
        if a.trust_legacy and (r["japanese"], r["grammar_point_id"]) in missing:
            continue
        try:
            # ONE required word per sentence. generate2.py draws N words and asks
            # for N sentences, so each sentence carries one; asking for a single
            # sentence with four words produced cramming instead:
            # 「スリッパ、飛行機、ラジオカセット、肉があります。」
            req = draw_words(r["grammar_point_id"], 1)
            temp = round(random.uniform(*TEMP_RANGE), 2)
            # the request identity is computed from the SAME inputs the request uses
            rid = request_identity(r, req, model=a.model, temperature=temp)
            outs, usage = gen_one(r, v, key, required=req, model=a.model,
                                  temperature=temp)
            cost += float(usage.get("cost", 0) or 0)
            for s in outs[:1]:                      # one fresh candidate per parent
                req_txt = [w["text"] for w in req]
                # bump the ledger so the next parent for this point draws different
                # words -- this is what generate2 does and what keeps 45 noun points
                # from all converging on the same vocabulary
                for w in req_txt:
                    if w in s:
                        _LED.bump(r["grammar_point_id"], w)
                log.write(json.dumps({
                    "parent": r["japanese"],
                    "japanese": s,
                    "grammar_point_id": r["grammar_point_id"],
                    "pattern": r.get("pattern", ""),
                    "frame": r.get("frame", ""),
                    "meaning_en": r.get("meaning_en", ""),
                    "required": req_txt,
                    "gen_model": a.model,          # auditable: which model produced this
                    "config": cfg,                 # GENERATOR identity: resume key
                    "config_schema": HASH_SCHEMA,  # so an old key is never compared
                    "request_id": rid,             # REQUEST identity: this exact input
                    "prompt_sha256": prompt_sha256(),   # evidence, not a convention
                    "regen_because": v["reason_codes"],
                    "failed_dimensions": v["failed_dimensions"],
                    "judge_reason": (v.get("reason") or "")[:200],
                }, ensure_ascii=False) + "\n")
                n += 1
            log.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
        except Exception as e:
            print(f"  regen {type(e).__name__} {str(e)[:100]}", flush=True)
    log.close()
    try:
        if _LED is not None:
            _LED.save()          # persist rotation so the next run draws other words
    except Exception as e:
        print(f"  ledger save failed: {e}", flush=True)
    print(f"\nwrote {n} candidates  ${cost:.5f}", flush=True)


if __name__ == "__main__":
    main()
