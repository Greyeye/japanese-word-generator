"""
generate.py -- generate N5 sentences for each target, gemma-4-26B-A4B, thinking OFF.

Writes attempts.jsonl (IMMUTABLE -- never rewritten; one row per raw model call)
and updates targets.json job status. Resumable: skips jobs already recorded.

  python generate.py --limit 16          # first 16 jobs (the N5 'formal' slice test)
  python generate.py --only desu-polite  # jobs whose id contains this
  python generate.py                     # everything still pending
"""

import json, os, re, sys, time, random, urllib.request

API    = "http://192.168.0.12:8889"
TOKEN  = "sk-unsloth-eb68e3f73915781686d00d2804cecbe8"
MODEL  = "lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF"
OUT    = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

BATCH  = 15                      # sentences per request (ladder optimum was ~20)
MAXTOK = 3000

def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default

TARGETS = json.load(open(os.path.join(DATA, "targets.json"), encoding="utf-8"))
ATT     = os.path.join(RUNS, "attempts.jsonl")

done = set()
if os.path.exists(ATT):
    for line in open(ATT, encoding="utf-8"):
        try:
            done.add(json.loads(line)["job_id"])
        except Exception:
            pass

def build_prompt(t, n, avoid=None):
    hint = "、".join(t["vocab_hint"])
    av = ""
    if avoid:
        av = ("\nすでに次の例文を作りました。これらと同じ文、またはほぼ同じ文は"
              "絶対に作らないでください:\n" + "\n".join(f"  {s}" for s in avoid))
    return f"""日本語のN5例文を{n}個作ってください。
文法: {t['pattern']}（{t['meaning_en']}）
{t['register_instruction']}
語彙: N5レベル以下の語彙のみを使ってください。漢字はN5の漢字だけにしてください。
なるべく次の語を使ってください: {hint}
長さ: 20文字以内。1文のみ。
禁止: 文頭にこの/その/あの/これ/それ/あれ。「彼/彼女/あの人」。
内容はなるべく多様に（話題・主語・場面を変える）。{av}
出力: JSONのみ。説明・思考・マークダウン記法は不要。
{{"sentences":["...","..."]}}"""

def call(prompt, temp):
    body = {"model": MODEL, "max_tokens": MAXTOK,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none", "temperature": temp}
    req = urllib.request.Request(API + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return time.time() - t0, r["usage"], r["choices"][0]["message"].get("content") or ""

def parse(txt):
    if not txt:
        return []
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return []
    try:
        d = json.loads(txt[i:j + 1])
        return [s for s in d.get("sentences", [])
                if isinstance(s, str) and s.strip().lower() not in ("sentences", "sentence", "...")]
    except Exception:
        return [x for x in re.findall(r'"([^"]{2,60})"', txt[i:j + 1]) if x.lower() != "sentences"]

limit = arg("--limit")
only  = arg("--only")
todo = [t for t in TARGETS["targets"]
        if t["job_id"] not in done and (only is None or only in t["grammar_point_id"])]
if limit:
    todo = todo[:int(limit)]

print(f"targets={len(TARGETS['targets'])} already_done={len(done)} todo={len(todo)}", flush=True)
log = open(ATT, "a", encoding="utf-8")
tot = 0
t_all = time.time()

for k, t in enumerate(todo, 1):
    # aim for target_count, in batches
    got_all, attempts = [], 0
    while len(got_all) < t["target_count"] and attempts < 3:
        attempts += 1
        need = min(BATCH, t["target_count"] - len(got_all) + 3)   # +3 slack for rejects
        temp = round(random.uniform(0.7, 1.0), 2)
        try:
            el, usage, content = call(
                build_prompt(t, need, avoid=got_all if attempts > 1 else None), temp)
            got = parse(content)
            log.write(json.dumps({
                "job_id": t["job_id"], "grammar_point_id": t["grammar_point_id"],
                "attempt": attempts, "model": MODEL, "temperature": temp,
                "reasoning_effort": "none", "sec": round(el, 2),
                "usage": usage, "returned": len(got), "sentences": got,
            }, ensure_ascii=False) + "\n")
            log.flush()
            got_all.extend(got)
            print(f"  [{k}/{len(todo)}] {t['grammar_point_id'][:38]:<38} "
                  f"a{attempts} {el:5.1f}s -> {len(got):>2} (total {len(got_all)})", flush=True)
        except Exception as e:
            log.write(json.dumps({"job_id": t["job_id"], "attempt": attempts,
                                  "error": str(e)[:200]}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"  [{k}/{len(todo)}] {t['grammar_point_id']} ERROR {str(e)[:80]}", flush=True)
            break
    tot += len(got_all)
    t["status"] = "generated"

json.dump(TARGETS, open(os.path.join(DATA, "targets.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log.close()
print(f"\nDONE: {tot} sentences from {len(todo)} jobs in {time.time()-t_all:.0f}s "
      f"({(time.time()-t_all)/max(tot,1):.2f} s/sentence)")
