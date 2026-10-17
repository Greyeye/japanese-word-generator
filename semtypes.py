"""
types.py -- semantic typing for N5 nouns, so a pool draw can only put a word into
a slot where it makes sense.

Motivation (measured): with an untyped pool, counter-en-money drew お姉さん and
produced 「お姉さんは五百円です。」, counter-hon-long drew キロメートル and produced
「キロメートルが十あります。」. ~25% of counter sentences were semantically broken,
and no deterministic check can catch it -- there is no rule against it. Nothing in
the pipeline knew that people are not priced in yen.

Two jobs this does:
  1. filter the pool: a 円 slot draws only priceable nouns
  2. make the plausibility check deterministic: 円 x PERSON fails a tag lookup,
     no model call needed

Classification itself is a fixed-label task, which is what small/local models are
good at -- see classify(). Output lands in noun_types.json for review.
"""

import json, os, re, sys, time, random, urllib.request

OUT = r"H:\jlpt\gen"

# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

# ---- the tag vocabulary -----------------------------------------------------
# A noun carries a SET of tags (車 = {MACHINE, OBJECT}); a grammar point declares
# which tags its slot accepts. Overlap is expected and fine.
TAGS = {
    "PERSON":   "a person or people (母, 先生, 友達, お医者さん)",
    "ANIMAL":   "an animal (犬, 猫, 鳥, 魚)",
    "PLACE":    "a location (学校, 家, 駅, 公園, 店)",
    "TIME":     "a time word or point in time (朝, 今日, 毎日, 今年)",
    "MONEY":    "money itself (お金, 円)",
    "FOOD":     "something eaten (パン, ご飯, 肉, りんご)",
    "DRINK":    "something drunk (水, お茶, コーヒー)",
    "CLOTHING": "something worn (シャツ, 靴, 帽子, コート)",
    "MACHINE":  "a device or vehicle (車, テレビ, パソコン, カメラ)",
    "LONGTHIN": "a long thin object countable with 本 (ペン, 傘, 木, ビール)",
    "FLAT":     "a flat object countable with 枚 (紙, 皿, シャツ, 切手)",
    "BODY":     "a body part (手, 目, 頭, 足)",
    "NATURE":   "a natural thing (山, 川, 空, 花, 海)",
    "ABSTRACT": "non-physical (元気, 気持ち, 名前, 仕事)",
    "OBJECT":   "any other concrete inanimate object (本, 机, かばん, 時計)",
}

# Which tags each grammar point's slot accepts.
#   None  -> no constraint (leave the pool alone)
#   set() -> the point needs no noun at all (numbers / pure time frames)
COUNTER_TAGS = {
    # a commodity with a price. NOT people, NOT places, NOT time words,
    # NOT money itself. This is the rule that removes お姉さんは五百円です。
    "counter-en-money": {"FOOD", "DRINK", "CLOTHING", "OBJECT", "MACHINE", "NATURE"},

    # 分/時間 measure how long you do something to an object
    "counter-fun-minute":     {"FOOD", "DRINK", "CLOTHING", "OBJECT", "MACHINE"},
    "jikan-time-duration":    {"FOOD", "DRINK", "CLOTHING", "OBJECT", "MACHINE"},

    "counter-hon-long": {"LONGTHIN", "DRINK"},
    "counter-mai-flat": {"FLAT", "CLOTHING"},
    "counter-people-nin": {"PERSON"},
    "counter-sai-age":    {"PERSON"},
    "counter-tsu":        {"OBJECT", "FOOD", "NATURE", "CLOTHING", "MACHINE"},

    # clock time takes a number and an action; the noun is free but must not be
    # a person or place ("三時に お母さんになりました" is nonsense)
    "counter-ji-oclock": {"OBJECT", "FOOD", "MACHINE", "NATURE", "BODY"},

    "arimasu-existence-inanimate": {"OBJECT", "FOOD", "DRINK", "CLOTHING",
                                    "MACHINE", "NATURE", "FLAT", "LONGTHIN"},
    "imasu-existence-animate":     {"PERSON", "ANIMAL"},

    "ga-hoshii-wanting-thing": {"OBJECT", "FOOD", "MACHINE", "CLOTHING", "NATURE"},

    "mai-every-prefix":  {"TIME"},
    "nanyoubi-day-of-week": None,
    "nanji-what-time":   set(),
    "ikutsu-how-many":   {"OBJECT", "FOOD", "NATURE", "CLOTHING"},
    "toki-ni-when-basic": {"TIME", "OBJECT", "PERSON"},
    "location-nouns-ue-shita": {"OBJECT", "PLACE"},
}


def pool_for(gp, pool):
    """Filter a POS pool down to the tags this grammar point accepts."""
    allowed = COUNTER_TAGS.get(gp, None)
    if allowed is None:
        return pool
    if not allowed:
        return []
    ok = [v for v in pool if set(v.get("tags") or []) & allowed]
    return ok or pool            # never return empty: fall back to untyped


# ---- classification ---------------------------------------------------------
def classify(nouns, api, token, model, batch=40, retries=2):
    """Tag N5 nouns using the local model. Fixed-label classification."""
    done = {}
    for i in range(0, len(nouns), batch):
        chunk = nouns[i:i + batch]
        listing = "\n".join(f"{v['text']}\t{v.get('reading','')}\t{v.get('meaning','')}"
                            for v in chunk)
        prompt = (
            "次の日本語の単語それぞれに、当てはまるタグをすべて付けてください。\n"
            "使えるタグ: " + ", ".join(TAGS) + "\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in TAGS.items())
            + "\n\n単語（表記 / 読み / 意味）:\n" + listing
            + "\n\n出力はJSONのみ。キーは単語の表記、値はタグの配列。\n"
              '例: {"犬":["ANIMAL"],"車":["MACHINE","OBJECT"],"元気":["ABSTRACT"]}'
        )
        for att in range(retries + 1):
            try:
                body = {"model": model, "max_tokens": 2500,
                        "messages": [{"role": "user", "content": prompt}],
                        "reasoning_effort": "none",
                        "temperature": 0.2 if att == 0 else 0.5}
                req = urllib.request.Request(api + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Authorization": "Bearer " + token,
                             "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=300))
                txt = r["choices"][0]["message"].get("content") or ""
                txt = re.sub(r"```(?:json)?", "", txt).strip()
                a, b = txt.find("{"), txt.rfind("}")
                d = json.loads(txt[a:b + 1])
                for k, v in d.items():
                    if isinstance(v, list):
                        done[k.strip()] = [t for t in v if t in TAGS]
                break
            except Exception as e:
                if att == retries:
                    print(f"   batch {i//batch} failed: {str(e)[:80]}", flush=True)
        print(f"   batch {i//batch + 1}/{(len(nouns)+batch-1)//batch} "
              f"-> {len(done)} tagged", flush=True)
    return done


def load_types(path=None):
    path = path or os.path.join(RUNS, "noun_types.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def apply_types(lex, types):
    """Attach a 'tags' list to each n5_vocab entry that has a classification."""
    for v in lex["n5_vocab"]:
        v["tags"] = types.get(v["text"]) or types.get(v.get("reading", "")) or []
    return lex


if __name__ == "__main__":
    lex = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))
    src = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
    api   = re.search(r'API\s*=\s*"([^"]+)"', src).group(1)
    token = re.search(r'TOKEN\s*=\s*"([^"]+)"', src).group(1)
    model = re.search(r'MODEL\s*=\s*"([^"]+)"', src).group(1)

    nouns = [v for v in lex["n5_vocab"] if v.get("pos") == "noun"]
    print(f"classifying {len(nouns)} N5 nouns ...", flush=True)
    t0 = time.time()
    types = classify(nouns, api, token, model)
    json.dump(types, open(os.path.join(RUNS, "noun_types.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(types)}/{len(nouns)} nouns tagged in {time.time()-t0:.0f}s")
    import collections
    c = collections.Counter(t for v in types.values() for t in v)
    for k, n in c.most_common():
        print(f"   {k:<10} {n}")
    print("\nuntagged:", [v['text'] for v in nouns if v['text'] not in types][:30])
