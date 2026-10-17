"""
classify_nouns.py -- semantic typing for N5 nouns.

Classification is a task the local model is genuinely good at (and cheap at), so
this uses it -- unlike generation, where we found it unreliable. Output is a
JSON table we then spot-check by hand and can correct with overrides.

Two orthogonal axes, because counters need both:

  kind  -- what the thing IS        PERSON ANIMAL PLACE TIME UNIT FOOD CLOTHING
                                    VEHICLE NATURE OBJECT SUBSTANCE ABSTRACT EVENT
  shape -- how it is COUNTED        FLAT (枚) LONG (本) ROUND (個) CONTAINER (杯)
                                    MACHINE (台)

A noun can carry several labels: 水 = {FOOD, SUBSTANCE}, 車 = {VEHICLE, MACHINE}.

  python classify_nouns.py            # classify all N5 nouns, write noun_classes.json
  python classify_nouns.py --dry      # classify first 40 only and print
"""

import json, os, re, sys, time, urllib.request

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

_src  = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
API   = re.search(r'API\s*=\s*"([^"]+)"', _src).group(1)
TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"', _src).group(1)
MODEL = re.search(r'MODEL\s*=\s*"([^"]+)"', _src).group(1)

BATCH = 25        # 45 produced JSON long enough to fail parsing once

KINDS = ["PERSON", "ANIMAL", "PLACE", "TIME", "UNIT", "FOOD", "CLOTHING",
         "VEHICLE", "NATURE", "OBJECT", "SUBSTANCE", "ABSTRACT", "EVENT"]
SHAPES = ["FLAT", "LONG", "ROUND", "CONTAINER", "MACHINE"]

SYSTEM = """あなたは日本語名詞の分類器です。与えられた名詞を、種類と形に分類します。

種類 (kind) -- 1つ以上選ぶ:
PERSON 人・職業 / ANIMAL 動物 / PLACE 場所 / TIME 時を表す語 / UNIT 単位・お金
FOOD 食べ物・飲み物 / CLOTHING 衣服 / VEHICLE 乗り物 / NATURE 自然のもの
OBJECT 物 / SUBSTANCE 形のない物質 / ABSTRACT 抽象的なもの / EVENT 出来事

形 (shape) -- 数え方に関わるものだけ。該当しなければ空配列:
FLAT 平らなもの（紙・皿・シャツ）/ LONG 細長いもの（瓶・ペン・傘・木）
ROUND 個で数えるもの（りんご・卵）/ CONTAINER 容器（コップ・カップ）
MACHINE 機械・装置（車・テレビ・カメラ）

出力はJSONのみ: {"単語":{"kind":["..."],"shape":["..."]}}"""


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def call(system, user, temp=0.0):
    body = {"model": MODEL, "max_tokens": 4000,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "reasoning_effort": "none", "temperature": temp}
    req = urllib.request.Request(API + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + TOKEN,
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"].get("content") or ""


def parse(txt):
    if not txt:
        return {}
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return {}
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return {}


lex   = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))
nouns = [v for v in lex["n5_vocab"] if v.get("pos") == "noun"]

# Resumable: keep whatever is already classified and only re-do the gaps. One
# batch can fail to parse (long JSON output) and we do not want to re-classify
# 331 words to recover 45.
NC_PATH = os.path.join(DATA, "noun_classes.json")
out = {}
if os.path.exists(NC_PATH):
    try:
        out = json.load(open(NC_PATH, encoding="utf-8"))
    except Exception:
        out = {}
nouns = [v for v in nouns if v["text"] not in out]

print(f"N5 nouns: {len(lex['n5_vocab'])}   already classified: {len(out)}   "
      f"todo: {len(nouns)}   model: {MODEL.split('/')[-1]}", flush=True)

dry = "--dry" in sys.argv
if dry:
    nouns = nouns[:40]
if not nouns:
    print("nothing to do")
    sys.exit(0)
t0 = time.time()
for b in range(0, len(nouns), BATCH):
    chunk = nouns[b:b + BATCH]
    listing = "\n".join(f"{v['text']} ({v['meaning']})" for v in chunk)
    user = f"次の名詞を分類してください。\n\n{listing}\n\n出力: {{\"単語\":{{\"kind\":[...],\"shape\":[...]}}}}"
    try:
        got = parse(call(SYSTEM, user))
    except Exception as e:
        print(f"  batch {b//BATCH+1} ERROR {str(e)[:80]}", flush=True)
        continue

    for v in chunk:
        w = v["text"]
        c = got.get(w)
        if not isinstance(c, dict):
            # tolerate a bare list, or a {word: "KIND"} string form
            if isinstance(c, list):
                c = {"kind": c, "shape": []}
            else:
                continue
        kind = [k for k in (c.get("kind") or []) if k in KINDS]
        shape = [s for s in (c.get("shape") or []) if s in SHAPES]
        if not kind:
            kind = ["OBJECT"]
        out[w] = {"kind": kind, "shape": shape, "meaning": v["meaning"]}

    print(f"  batch {b//BATCH+1}/{(len(nouns)-1)//BATCH+1}  "
          f"{len(out)}/{len(nouns)} classified  ({time.time()-t0:.0f}s)", flush=True)

print(f"\nclassified {len(out)}/{len(nouns)} in {time.time()-t0:.0f}s")

if dry:
    print(json.dumps(out, ensure_ascii=False, indent=1)[:2500])
    sys.exit(0)

# merge with any hand-written overrides (overrides always win)
ov_path = os.path.join(DATA, "noun_class_overrides.json")
if os.path.exists(ov_path):
    ov = json.load(open(ov_path, encoding="utf-8"))
    for w, c in ov.items():
        out.setdefault(w, {}).update(c)
    print(f"applied {len(ov)} hand overrides")

json.dump(out, open(os.path.join(DATA, "noun_classes.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote noun_classes.json")

import collections
print("\nkind distribution:")
for k, n in collections.Counter(k for c in out.values() for k in c["kind"]).most_common():
    print(f"   {k:<12} {n}")
print("\nshape distribution:")
for k, n in collections.Counter(s for c in out.values() for s in c["shape"]).most_common():
    print(f"   {k:<12} {n}")
