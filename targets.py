"""
targets.py -- build generation jobs for N5 from the japanese-language-data repo.

Reads:
  data/grammar/grammar.json                 -> 77 N5 grammar points (+ formality)
  data/enrichment/jlpt-classifications.json -> N5 vocab (684), N5 kanji (79)
  data/core/words.json                      -> JMdict, for POS + "is this a real word"

Writes (H:\\jlpt\\gen\\):
  lexicon.json   cached N5 vocab w/ POS + kanji sets + known-word set
  targets.json   the job list
"""

import json, os, re, collections, sys

REPO = r"H:\jlpt\japanese-language-data\data"
OUT  = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

os.makedirs(OUT, exist_ok=True)

JP_RUN = re.compile(r'[\u3040-\u30ff\u4e00-\u9fffー]+')

# ---------------------------------------------------------------- load sources
def load(p):
    with open(os.path.join(REPO, p), encoding="utf-8") as f:
        return json.load(f)

print("loading sources ...", flush=True)
grammar = load("grammar/grammar.json")["grammar_points"]
classif = load("enrichment/jlpt-classifications.json")["classifications"]
words   = load("core/words.json")["words"]
print(f"  grammar={len(grammar)}  classifications={len(classif)}  jmdict={len(words)}")

# ---------------------------------------------------------------- lexicon cache
LEX = os.path.join(DATA, "lexicon.json")
if os.path.exists(LEX):
    lex = json.load(open(LEX, encoding="utf-8"))
    print("  using cached lexicon.json")
else:
    # POS by jmdict id
    pos_by_id = {}
    known_words = set()
    for w in words:
        wid = w["id"]
        ps = set()
        for s in w.get("sense", []):
            ps.update(s.get("partOfSpeech", []))
        pos_by_id[wid] = sorted(ps)
        for k in w.get("kana", []):
            known_words.add(k["text"])
        for k in w.get("kanji", []):
            known_words.add(k["text"])

    def pos_class(ps):
        if any(p.startswith("v") for p in ps):            return "verb"
        if "adj-i" in ps or "adj-ix" in ps:               return "i-adj"
        if "adj-na" in ps:                                return "na-adj"
        if any(p == "n" or p.startswith("n-") for p in ps): return "noun"
        if any(p.startswith("adv") for p in ps):          return "adverb"
        return "other"

    n5_vocab = []
    for c in classif:
        if c["kind"] != "vocab" or c["level"] != "N5":
            continue
        ps = pos_by_id.get(c.get("jmdict_seq"), [])
        n5_vocab.append({"text": c["text"], "reading": c.get("reading"),
                         "meaning": c.get("meaning_en"), "pos": pos_class(ps),
                         "pos_raw": ps, "jmdict_seq": c.get("jmdict_seq")})

    n5_kanji = [c["text"] for c in classif if c["kind"] == "kanji" and c["level"] == "N5"]

    # every JLPT-listed word at any level (for "above level" vs "not a word")
    jlpt_all = {}
    for c in classif:
        if c["kind"] == "vocab":
            jlpt_all[c["text"]] = c["level"]

    lex = {"n5_vocab": n5_vocab, "n5_kanji": n5_kanji,
           "known_words": sorted(known_words), "jlpt_all": jlpt_all}
    json.dump(lex, open(LEX, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  wrote lexicon.json  ({len(n5_vocab)} N5 words, {len(n5_kanji)} N5 kanji, "
          f"{len(known_words)} known words)")

V = lex["n5_vocab"]
KANJI = set(lex["n5_kanji"])
print(f"  N5 vocab={len(V)}  N5 kanji={len(KANJI)}")
print("  POS breakdown:", collections.Counter(v["pos"] for v in V))

# ---------------------------------------------------------------- pattern->regex
def derive_check(pattern):
    """Pull the Japanese fragment out of a human pattern like 'Noun + ではありません'.

    Handles alternation ('Noun + 歳 / 才' -> 歳|才) and drops optional markers.
    Returns (regex_fragment, is_loose).
    """
    # work on the part after the last '+' where the Japanese normally lives
    tail = pattern.split("+")[-1]
    parts = [p for p in re.split(r'[/／]', tail)]
    frags = []
    for p in parts:
        p = re.sub(r'[（(][^）)]*[）)]', '', p)          # drop parenthetical notes
        r = JP_RUN.findall(p)
        if r:
            frags.append("".join(r))
    if not frags:
        runs = JP_RUN.findall(pattern)
        frags = [max(runs, key=len)] if runs else []
    if not frags:
        return None, True
    uniq = list(dict.fromkeys(f for f in frags if f))
    frag = "|".join(uniq) if len(uniq) > 1 else uniq[0]
    return frag, (len(uniq) == 1 and len(uniq[0]) <= 2)

# patterns the heuristic gets wrong; hand-corrected
OVERRIDES = {
    "koko-soko-asoko-doko": "(ここ|そこ|あそこ|どこ)",
    "da-plain-copula":      "だ[。！？]?$",
    "i-adjective-nonpast":  "(い|です)[。！？]?$",
    "i-adj-desu-politeness": "い[^。]*です[。！？]?$",
    "counter-tsu":          "つ[。！？]?$",
    "i-adj-adverbial-ku":   "く[^。]*",
}

REGISTER_INSTR = {
    "formal":  "です・ます体（敬体）で書いてください。文末は「です／ます／でした／ました／ません」のいずれかにしてください。",
    "casual":  "普通体（常体）で書いてください。「です・ます」は使わないでください。",
    "neutral": "N5の学習者向けなので、基本的に「です・ます体」で書いてください。",
}

def vocab_for(pattern, i):
    """Rotate through N5 vocab, biased to the POS the pattern needs."""
    wants = None
    p = pattern.lower()
    if "verb" in p:  wants = "verb"
    elif "adjective" in p or "い-adjective" in p or "な-adjective" in p: wants = ("i-adj", "na-adj")
    elif "noun" in p: wants = "noun"

    if wants:
        pool = [v for v in V if v["pos"] in (wants if isinstance(wants, tuple) else (wants,))]
        if len(pool) < 12:
            pool = V
    else:
        pool = V

    stride = max(len(pool) // 12, 5)
    start = (i * stride) % len(pool)
    return [pool[(start + k) % len(pool)]["text"] for k in range(min(12, len(pool)))]

n5 = [g for g in grammar if g["level"] == "N5"]
n5.sort(key=lambda g: g["id"])

targets = []
for i, g in enumerate(n5):
    frag, loose = derive_check(g.get("pattern", ""))
    if g["id"] in OVERRIDES:
        frag, loose = OVERRIDES[g["id"]], False
    targets.append({
        "job_id":            f"n5-{i:03d}-{g['id']}",
        "level":             "N5",
        "grammar_point_id":  g["id"],
        "pattern":           g.get("pattern"),
        "meaning_en":        g.get("meaning_en"),
        "formality":         g.get("formality"),
        "register_instruction": REGISTER_INSTR.get(g.get("formality"), REGISTER_INSTR["neutral"]),
        "check_regex":       frag,
        "check_loose":       bool(loose),
        "vocab_hint":        vocab_for(g.get("pattern") or "", i),
        "target_count":      20,
        "status":            "pending",
    })

json.dump({"level": "N5", "count": len(targets), "targets": targets},
          open(os.path.join(DATA, "targets.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"\nwrote {len(targets)} targets -> {os.path.join(OUT,'targets.json')}")
print("\n-- derived check_regex review (flag any that look too loose) --")
for t in targets[:30]:
    print(f"  {t['grammar_point_id']:<44} {str(t['pattern'])[:30]:<30} -> {t['check_regex']}")
print(f"  ... ({len(targets)-30} more)")
