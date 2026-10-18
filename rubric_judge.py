"""Rubric judge: multi-dimensional verdicts instead of a binary ok/true.

Why this replaces a yes/no judge
--------------------------------
A single boolean collapses independent failure modes that we measured to be
genuinely different: 46% of rejected N5 sentences were valid Japanese that simply
did not use their target grammar, 30% used it with the wrong nuance, and only 20%
were broken Japanese. Those need opposite responses, so `ok: false` is not a
usable signal. It also throws away the diagnosis -- recovering defect classes from
free-text reasons needed a keyword classifier with a 19.8% "unclear" bucket.

Design rules (each one follows from a measured failure)
-------------------------------------------------------
1. The model supplies DIMENSION verdicts only. `reason_codes`, `primary_defect`
   and `overall` are DERIVED IN CODE. This structurally prevents the failure we
   hit with the binary judge, where haiku wrote "誤りとは言えない" (not an error)
   in its own reason and marked the sentence invalid anyway -- 40 such cases.
   A summary field the model can contradict is a summary field it will contradict.

2. `uncertain` is a first-class verdict, and the prompt says so explicitly.
   Forcing every sentence into pass/fail is what produces the vague rejections.

3. Target span is requested because it is CHECKABLE: we verify it is a literal
   substring of the sentence and, when the grammar point has one, that it matches
   `check_regex`. That turns "does it really use the pattern?" into evidence
   rather than another opinion. It also lets a `target_used: fail` be graded.

4. Batched by default for throughput, --batch 1 for calibration. Measured caveat:
   verdicts depend on batch composition -- on 44 sentences where two generators
   produced byte-identical text, batched verdicts still disagreed 2-4% of the
   time -- and the model notices within-batch duplication. Both are tolerable at
   our effect sizes but should be known.

5. Reasoning params are per-model. Some endpoints refuse to disable reasoning
   (`claude-sonnet-5.5`, `gpt-6.1-sol`: "Reasoning is mandatory for this
   endpoint"); sending enabled:false to them is a hard 400.
"""
import os, re, json, time, hashlib, argparse, urllib.request, urllib.error

# --- verdict vocabulary ----------------------------------------------------
V3 = {"pass", "fail", "uncertain"}

DIMENSIONS = {
    "grammar_correct":  ("日本語として文法的に正しいか", "BROKEN"),
    "meaning_coherent": ("意味が通り、語の組み合わせが論理的か", "INCOHERENT"),
    "target_used":      ("対象文法が実際に使われているか（単に含まれているだけでは不合格）", "WRONGPAT"),
    "natural":          ("日本語として自然で、母語話者が使うか", "NUANCE"),
    "level_ok":         ("N5の語彙・漢字・文法の範囲に収まっているか", "LEVEL"),
    "register_ok":      ("文体（です・ます／だ）が対象文法の指定と合っているか", "REGISTER"),
}
# dimension -> reason code, and the order a failure is promoted to "primary"
SEVERITY = ["BROKEN", "INCOHERENT", "WRONGPAT", "NUANCE", "REGISTER", "LEVEL"]

# --- per-model reasoning handling ------------------------------------------
# "off"     : disable thinking (cheapest, works on most OpenRouter models)
# "exclude" : endpoint refuses to disable; hide the trace and budget for it
REASONING = {
    "openai/gpt-6-luna":            {"reasoning": {"enabled": False}, "max_tokens": 1500},
    "openai/gpt-6-sol":             {"reasoning": {"enabled": False}, "max_tokens": 2500},
    "anthropic/claude-haiku-5.5":   {"reasoning": {"enabled": False}, "max_tokens": 1500},
    "anthropic/claude-sonnet-5.5":  {"reasoning": {"exclude": True},  "max_tokens": 8000},
    "anthropic/claude-opus-5.5":    {"reasoning": {"exclude": True},  "max_tokens": 8000},
}
DEFAULT_MODEL = "openai/gpt-6-luna"      # best measured recall (0.92) at lowest cost
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

SYS = ("あなたは日本語教材の校閲者です。JLPT N5（初級）の例文を検査します。"
       "各観点を独立に判定し、必要なら「uncertain」を使ってください。"
       "無理に pass / fail に決める必要はありません。")


# --- N5 level evidence -----------------------------------------------------
# level_ok was the LEAST reliable dimension: 3.8% (luna) vs 17.2% (sonnet) on the
# same sentences, a 4.5x spread, because neither model holds a trustworthy N5
# word list. Sonnet flagged 切手 (符) and 風呂 (風) as above N5; both are N5.
#
# So stop asking the model to remember. lexicon.json carries jlpt_all: 8,050
# words mapped to N5..N1. We tokenise the sentence, look each content word up,
# and hand the judge the FACTS. Where the evidence is decisive the verdict is
# derived in code rather than left to the model.
GEN = r"H:\jlpt\gen"
_LEVELS = None
_N5 = None


def level_data():
    global _LEVELS, _N5
    if _LEVELS is None:
        lex = json.load(open(os.path.join(GEN, "data", "lexicon.json"), encoding="utf-8"))
        _LEVELS = lex.get("jlpt_all") or {}
        _N5 = {v["text"] for v in lex.get("n5_vocab", [])}
    return _LEVELS, _N5


def level_evidence(sentence):
    """Content words in `sentence` that sit ABOVE N5, and words we cannot place.

    Returns (above, unknown) where above is [(word, level)] worst-first.

    Two sources, and they disagree, so the precedence matters:
      n5_vocab  (673 curated N5 words) -- AUTHORITATIVE. jlpt_all contradicts it on
                56 of those words (人->N1, 三->N1, 店->N1, 東->N1, 中->N3, 明日->N4),
                because jlpt_all grades a homograph's other sense. 分 is a counter,
                absent from n5_vocab, and jlpt_all calls it N1 -- flagging it would
                fail every counter-fun-minute sentence.
      jlpt_all  (8,050 words, N5..N1)  -- trusted ONLY at the ends of the scale,
                for words >= 2 characters and absent from n5_vocab.

    Absence from both lists is reported as unknown, never as "above level".
    """
    levels, n5 = level_data()
    try:
        from janome.tokenizer import Tokenizer
        toks = Tokenizer().tokenize(sentence)
    except Exception:
        return [], []
    above, unknown = [], []
    for t in toks:
        pos = t.part_of_speech.split(",")[0]
        if pos not in ("名詞", "動詞", "形容詞", "副詞"):
            continue
        forms = []
        if t.base_form and t.base_form != "*":
            forms.append(t.base_form)
        forms.append(t.surface)
        if any(f in n5 for f in forms):          # curated N5 list wins outright
            continue
        lvl = next((levels[f] for f in forms if f in levels), None)
        word = forms[0]
        if lvl in ("N1", "N2") and len(word) >= 2:
            above.append((word, lvl))            # confidently above N5
        elif lvl is None and len(t.surface) >= 2:
            unknown.append(t.surface)            # can't place it; do not accuse
    best = {}
    for w, l in above:
        if w not in best or l < best[w]:
            best[w] = l
    return sorted(best.items(), key=lambda kv: kv[1]), sorted(set(unknown))


def build_prompt(batch, with_frame=True):
    """batch: list of dicts with japanese, pattern, meaning_en, [frame, register_instruction]"""
    dims = "\n".join(f'  "{k}": "pass"|"fail"|"uncertain"   -- {v[0]}'
                     for k, v in DIMENSIONS.items())
    L = []
    for i, r in enumerate(batch):
        s = f'{i+1}. 例文: {r["japanese"]}\n   対象文法: {r.get("pattern","")}'
        if r.get("meaning_en"):
            s += f'（{r["meaning_en"]}）'
        if with_frame and r.get("frame"):
            s += f'\n   構造条件: {r["frame"][:200]}'
        if r.get("register_instruction"):
            s += f'\n   文体の指定: {r["register_instruction"][:80]}'
        # hand over the level FACTS instead of asking the model to recall N5 vocabulary
        if r.get("_above"):
            s += ("\n   【語彙の調査結果】N5を超える語: "
                  + ", ".join(f"{w}({l})" for w, l in r["_above"]))
        elif r.get("_unknown"):
            s += ("\n   【語彙の調査結果】N5リストに無い語（判定不能）: "
                  + ", ".join(r["_unknown"][:6]))
        else:
            s += "\n   【語彙の調査結果】すべてN5の語です。"
        L.append(s)
    return (
        "次の例文を、下の各観点について独立に判定してください。\n\n"
        "観点（各 pass / fail / uncertain）:\n" + dims + "\n\n"
        "判定の注意:\n"
        "- target_used は、対象文法が文の中で「実際に機能している」かをみる。"
        "単に同じ形が含まれているだけ、引用されているだけ、話題として言及されているだけの場合は fail。\n"
        "- 文が正しく自然でも、対象文法が使われていなければ target_used は fail。\n"
        "- level_ok については【語彙の調査結果】を参考にすること。"
        "そこに挙げられた語はN5としては難しい可能性が高いが、機械的な判定ではない。"
        "助詞・助動詞・敬称（さん等）・明らかに基本語の場合は無視してよい。"
        "挙げられた語を根拠に判断し、自分でN5かどうかを推測しないこと。\n"
        "- register_ok は文体の指定がある場合だけ判定し、指定がなければ pass。\n"
        "- 確信が持てない場合は uncertain を選ぶこと（推測で pass にしない）。\n\n"
        "あわせて、対象文法を表している部分（span）を例文からそのまま抜き出してください。\n"
        "該当部分がなければ span は \"\" にしてください。\n\n"
        '出力はJSONのみ:\n{"results":[{"i":1,' 
        + ",".join(f'"{k}":"pass"' for k in DIMENSIONS)
        + ',"span":"","reason":"短い理由"},...]}\n'
        f"全{len(batch)}件について必ず出力すること。\n\n" + "\n".join(L))


def derive(v, sentence="", above=None, unknown=None):
    """Turn dimension verdicts into codes + overall. Deliberately NOT model-supplied."""
    v = dict(v)
    # level_ok is grounded in the word list, but ASYMMETRICALLY.
    #
    # Deriving "fail" would need a trustworthy above-N5 signal, and we do not have
    # one: jlpt_all grades 明るい (plainly N5) as N1, contradicts n5_vocab on 56
    # words, and mis-lemmatises forms like 持てる/年生. A false "above level" would
    # fail valid sentences, so we never manufacture one.
    #
    # Deriving "pass" is safe: it requires every content word to be in the curated
    # n5_vocab, in which case the sentence really is N5 vocabulary. Anything
    # unplaceable is reported as unknown and left to the model.
    level_source = "model"
    if above == [] and unknown == []:
        v["level_ok"], level_source = "pass", "derived"
    codes = [DIMENSIONS[d][1] for d in DIMENSIONS
             if v.get(d) == "fail"]
    codes.sort(key=lambda c: SEVERITY.index(c) if c in SEVERITY else 99)
    dims = [d for d in DIMENSIONS if v.get(d) == "fail"]
    unc = [d for d in DIMENSIONS if v.get(d) == "uncertain"]
    if codes:
        overall = "fail"
    elif unc:
        overall = "uncertain"
    else:
        overall = "pass"

    span = (v.get("span") or "").strip()
    # a span is only evidence if it is literally present
    span_ok = None
    if span:
        span_ok = span in sentence
    out = dict(v)
    out.update({"reason_codes": codes,
                "primary_defect": codes[0] if codes else "",
                "failed_dimensions": dims,
                "uncertain_dimensions": unc,
                "overall": overall,
                "span_ok": span_ok,
                "level_source": level_source,
                "above_level": [list(x) for x in (above or [])],
                "unknown_words": list(unknown or [])})
    return out


def load_key(explicit=None):
    if explicit:
        return explicit
    p = r"C:\Users\greyeye\AppData\Local\hermes\.env"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = re.match(r'\s*OPENROUTER_API_KEY\s*=\s*(.+)', line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return os.environ.get("OPENROUTER_API_KEY", "")


def post(body, key, timeout=280):
    rq = urllib.request.Request(OR_URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(rq, timeout=timeout))


def _parse(txt):
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0:
        return []
    try:
        return json.loads(txt[i:j + 1]).get("results", [])
    except Exception:
        return []


def judge_batch(batch, model=DEFAULT_MODEL, key=None, temperature=0):
    """-> (list of derived verdicts aligned to batch, usage dict)"""
    key = load_key(key)
    # attach the level evidence before building the prompt (mutates copies)
    batch = [dict(r) for r in batch]
    for r in batch:
        above, unknown = level_evidence(r.get("japanese", ""))
        r["_above"], r["_unknown"] = above, unknown
    params = REASONING.get(model, {"reasoning": {"enabled": False}, "max_tokens": 2000})
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": build_prompt(batch)}]}
    body.update(params)
    r = post(body, key)
    txt = r["choices"][0]["message"].get("content") or ""
    usage = r.get("usage", {}) or {}
    finish = r["choices"][0].get("finish_reason")
    out = [None] * len(batch)
    for rr in _parse(txt):
        try:
            idx = int(rr["i"]) - 1
        except Exception:
            continue
        if not (0 <= idx < len(batch)):
            continue
        if any(rr.get(d) not in V3 for d in DIMENSIONS):
            continue                      # malformed: drop, caller counts it
        out[idx] = derive(rr, batch[idx].get("japanese", ""),
                          above=batch[idx].get("_above"),
                          unknown=batch[idx].get("_unknown"))
    return out, usage, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="jsonl with japanese/pattern/...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    # Resume key is the WHOLE input row, not the sentence text. Keying on text
    # collapses duplicate sentences (two arms can emit the same output, and one
    # generator can repeat itself), which makes the line count disagree with the
    # row count and drops the per-row provenance.
    def rowkey(src):
        return hashlib.md5(json.dumps(src, sort_keys=True, ensure_ascii=False)
                           .encode("utf-8")).hexdigest()
    seen = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                seen.add(r.get("_k") or r.get("japanese"))
            except Exception:
                pass
    todo = [r for r in rows if rowkey(r) not in seen]
    print(f"{len(rows)} rows, {len(seen)} already judged, {len(todo)} to go", flush=True)

    log = open(a.out, "a", encoding="utf-8")
    t0 = time.time(); cost = 0.0; nod = 0; n = 0
    for k in range(0, len(todo), a.batch):
        b = todo[k:k + a.batch]
        try:
            res, usage, finish = judge_batch(b, model=a.model, key=a.key)
            cost += float(usage.get("cost", 0) or 0)
            for src, v in zip(b, res):
                if v is None:
                    nod += 1
                    continue
                # carry through every input field (arm, id, provenance...) so the
                # verdict file can be joined and split without a second lookup
                rec = dict(src)
                rec["model"] = a.model
                rec["_k"] = rowkey(src)
                rec.update({d: v[d] for d in DIMENSIONS})
                rec.update({"span": v.get("span", ""), "reason": v.get("reason", ""),
                            "reason_codes": v["reason_codes"],
                            "primary_defect": v["primary_defect"],
                            "failed_dimensions": v["failed_dimensions"],
                            "uncertain_dimensions": v["uncertain_dimensions"],
                            "overall": v["overall"], "span_ok": v["span_ok"],
                            "level_source": v["level_source"],
                            "above_level": v["above_level"],
                            "unknown_words": v["unknown_words"],
                            "finish_reason": finish})
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            log.flush()
            if (k // a.batch) % 20 == 0:
                print(f"  b{k//a.batch+1:>4}  {n} written  {time.time()-t0:.0f}s", flush=True)
        except urllib.error.HTTPError as e:
            print(f"  b{k//a.batch+1}: HTTP {e.code} {e.read().decode()[:160]}", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"  b{k//a.batch+1}: {type(e).__name__} {str(e)[:100]}", flush=True)
            time.sleep(3)
        if a.sleep:
            time.sleep(a.sleep)
    log.close()
    print(f"\nwrote {n}  no_verdict {nod}  cost ${cost:.5f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
