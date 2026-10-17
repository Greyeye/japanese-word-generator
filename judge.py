"""
judge.py -- language judgement, with context.

Replaces the ten hand-tuned rule checks that used to live in validate.py. The point
is not "use a model instead of rules" -- a bare "is this valid?" call is a bad task
and would be inconsistent. The judge sees CONTEXT:

  * the grammar point the sentence was generated for, and its English gloss
  * reference examples of that same grammar point drawn from the corpus
  * the sentence itself

That reframes the question as "does this look like the other sentences we accept for
this pattern", which is easier, more stable, and identical at N5 and N1 -- which is
exactly why it scales where the rules did not.

A MISSING VERDICT IS NOT AN INVALID VERDICT.
The first version defaulted a missing entry to valid=false. When a batch overflowed
max_tokens the JSON arrived truncated, parse() returned [], and every sentence in the
batch was silently scored as defective -- a judge that reported 0 valid out of 10 and
looked confidently wrong. Truncation, an unparseable reply, and "the model says the
sentence is bad" are three different outcomes and are recorded as such.

  python judge.py --model lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF --batch 5
  python judge.py --nothink --batch 10          # reasoning_effort:none, much faster
"""

import json, os, re, sys, time, random, urllib.request, collections

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

_src = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
API   = re.search(r'API\s*=\s*"([^"]+)"', _src).group(1)
TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"', _src).group(1)


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


MODEL   = arg("--model", "unsloth/LFM2.5-24B-A2B-GGUF")
SRC     = arg("--in", os.path.join(RUNS, "strip_sentences.jsonl"))
LIMIT   = arg("--limit")
BATCH   = int(arg("--batch", 5))
NOTHINK = "--nothink" in sys.argv
MAXTOK  = 8000          # thinking needs room: ~500 tokens per sentence

TG = {t["grammar_point_id"]: t for t in
      json.load(open(os.path.join(DATA, "targets.json"), encoding="utf-8"))["targets"]}

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
if LIMIT:
    rows = rows[:int(LIMIT)]
tag = MODEL.split("/")[-1].replace(".gguf", "") + ("_nothink" if NOTHINK else "")
print(f"judging {len(rows)} sentences | {MODEL.split('/')[-1]} | batch={BATCH} "
      f"| thinking={'off' if NOTHINK else 'on'}")

# ---- retrieval: reference examples per grammar point -----------------------
refs = collections.defaultdict(list)
for r in rows:
    if r.get("status") == "validated":
        refs[r["grammar_point_id"]].append(r["japanese"])
for gp in refs:
    random.seed(len(gp))
    refs[gp] = random.sample(refs[gp], min(4, len(refs[gp])))

SYSTEM = """あなたは日本語教材の例文を検査する校閲者です。
与えられた例文が、指定された文法項目の教材として使えるかどうかを判定します。

判定の観点:
1. 文法と語順は正しいか
2. 自然な日本語か（意味が通じるか）
3. 指定された文法項目を実際に使っているか
4. 漢字で書くべき語をひらがなで書いていないか

出力はJSONのみ。説明やマークダウンは書かないこと。"""


def build_prompt(batch):
    parts = []
    for i, r in enumerate(batch):
        gp = r["grammar_point_id"]
        t = TG.get(gp, {})
        ex = refs.get(gp, [])
        block = (f"[{i+1}] 文法項目: {t.get('pattern','')}（{t.get('meaning_en','')}）\n"
                 f"    検査する例文: {r['japanese']}")
        if ex:
            block += "\n    同じ文法項目の既存例文: " + " / ".join(ex)
        parts.append(block)
    return ("次の例文を検査してください。\n\n" + "\n\n".join(parts) +
            '\n\n出力: {"results":[{"n":1,"valid":true,"problems":[],"confidence":0.9}, ...]}')


def call(system, user, temp=0.0):
    body = {"model": MODEL, "max_tokens": MAXTOK,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temp}
    if NOTHINK:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(API + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = r["choices"][0]
    return (time.time() - t0, r["usage"], ch["message"].get("content") or "",
            ch.get("finish_reason"))


def parse(txt):
    """Returns (results, ok). ok=False means we could not read a verdict at all."""
    if not txt:
        return [], False
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0:
        return [], False
    for cand in (txt[i:j + 1], txt[i:] + "]}", txt[i:] + "}"):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and "results" in d:
                return d["results"], True
        except Exception:
            continue
    return [], False


out_path = os.path.join(RUNS, f"judged_{tag}.jsonl")
log = open(out_path, "a", encoding="utf-8")
t0 = time.time()
done = 0
V = collections.Counter()

for b in range(0, len(rows), BATCH):
    batch = rows[b:b + BATCH]
    try:
        el, usage, content, finish = call(SYSTEM, build_prompt(batch))
    except Exception as e:
        print(f"  batch {b//BATCH+1} ERROR {str(e)[:120]}"); continue

    got_list, ok = parse(content)
    got = {g.get("n"): g for g in got_list if isinstance(g, dict)}
    if finish == "length":
        print(f"  batch {b//BATCH+1}: TRUNCATED (finish_reason=length) -- "
              f"raise max_tokens or lower --batch", flush=True)

    for i, r in enumerate(batch, 1):
        g = got.get(i)
        if not g or "valid" not in g:
            verdict = "no_verdict"          # NOT the same as invalid
        else:
            verdict = "valid" if g["valid"] else "invalid"
        V[verdict] += 1
        log.write(json.dumps({
            "japanese": r["japanese"], "grammar_point_id": r["grammar_point_id"],
            "model": MODEL, "thinking": not NOTHINK,
            "verdict": verdict,
            "problems": (g or {}).get("problems") or [],
            "confidence": (g or {}).get("confidence"),
            "finish_reason": finish,
            "sec": round(el / max(len(batch), 1), 2),
        }, ensure_ascii=False) + "\n")
    log.flush()
    done += len(batch)
    print(f"  {done}/{len(rows)}  ({time.time()-t0:.0f}s)  valid={V['valid']} "
          f"invalid={V['invalid']} no_verdict={V['no_verdict']}", flush=True)

log.close()
print(f"\n{out_path}")
print(f"  valid        {V['valid']}")
print(f"  invalid      {V['invalid']}")
print(f"  no verdict   {V['no_verdict']}   <- excluded from any agreement metric")
print(f"  {time.time()-t0:.0f}s total, {len(rows)/max(time.time()-t0,1):.2f} sentences/s")
