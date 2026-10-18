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
    """sha256 of the prompt TEMPLATE (function source + system message), truncated.

    Hashing the source is deliberate: it changes if the wording, the ordering, the
    instruction count or a comment inside the template changes. Hashing a constant
    string would not.
    """
    import hashlib
    import inspect
    src = inspect.getsource(regen_prompt) + "\n--SYS--\n" + REGEN_SYS
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def config_key(model=None, temperature=None):
    """Content-addressed identity of the GENERATION configuration.

    Includes the prompt hash, so a prompt edit produces a new identity automatically.
    Deliberately does NOT include the judge model: a judge change must not force
    expensive regeneration, while a generation change must never reuse old output.
    Those are separate identities and are named separately for that reason.
    """
    import hashlib
    payload = {
        "model": model or MODEL,
        "prompt_sha256": prompt_sha256(),
        "prompt_version_label": PROMPT_VERSION,
        "temperature_range": list(TEMP_RANGE),
        "max_tokens": MAX_TOKENS,
    }
    import json as _json
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def judge_config_key(judge_model):
    """Identity of the JUDGING configuration, separate from generation."""
    import hashlib
    import json as _json
    return hashlib.sha256(_json.dumps(
        {"judge_model": judge_model, "rubric": "rubric_judge-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]


def regen_prompt(item, verdict, required=()):
    """State exactly what was wrong, and ask for a fresh sentence -- not a fix."""
    bad = ", ".join(verdict["reason_codes"])
    dims = ", ".join(verdict["failed_dimensions"])
    words = "、".join(v["text"] for v in required) if required else ""
    return (
        "次の日本語の例文は、JLPT N5 の教材としては不合格でした。"
        "同じ文法を使って、別の新しい例文を1つ書いてください。"
        "前の例文を直すのではなく、まったく別の文を作ること。\n\n"
        f"対象文法: {item.get('pattern','')}（{item.get('meaning_en','')}）\n"
        + (f"構造条件: {item['frame']}\n" if item.get("frame") else "")
        + (f"必ず使う語: {words}\n" if words else "")
        + f"\n前の例文: {item['japanese']}\n"
        f"問題点: {bad}  （観点: {dims}）\n"
        f"校閲者の指摘: {verdict.get('reason','')[:200]}\n\n"
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
    # Two sets, deliberately. A row with no recorded config has UNKNOWN provenance:
    # loading the module default (which is what this did) silently asserts the old row
    # was produced by the current configuration. It may well have been produced by a
    # different model or prompt, so unknown is not a match -- those items regenerate.
    # --trust-legacy accepts them explicitly for a run that knows the history.
    done = set()            # (parent, gp, config) -- config verified identical
    unknown = set()         # (parent, gp) -- provenance not recorded, NOT assumed
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                d = json.loads(l)
                k = (d["parent"], d.get("grammar_point_id"))
                cfg0 = d.get("config")
                if cfg0 is None:
                    unknown.add(k)
                else:
                    done.add(k + (cfg0,))
            except Exception:
                pass
    if unknown:
        print(f"  {len(unknown)} existing rows have NO recorded config (unknown "
              f"provenance)", flush=True)
        print(f"  -> regenerating them, because unknown is not a match", flush=True)
    if a.trust_legacy:
        print(f"  --trust-legacy: accepting them as-is", flush=True)
    log = open(a.out, "a", encoding="utf-8")
    cfg = config_key(a.model)
    print(f"  config {cfg}", flush=True)
    n = 0; cost = 0.0
    for r, v in todo:
        key3 = (r["japanese"], r["grammar_point_id"], cfg)
        if key3 in done:
            continue
        # unknown-provenance rows are accepted only on an explicit opt-in
        if a.trust_legacy and (r["japanese"], r["grammar_point_id"]) in unknown:
            continue
        try:
            # ONE required word per sentence. generate2.py draws N words and asks
            # for N sentences, so each sentence carries one; asking for a single
            # sentence with four words produced cramming instead:
            # 「スリッパ、飛行機、ラジオカセット、肉があります。」
            req = draw_words(r["grammar_point_id"], 1)
            outs, usage = gen_one(r, v, key, required=req, model=a.model)
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
                    "config": cfg,                 # model+prompt identity, part of the resume key
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
