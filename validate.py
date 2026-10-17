"""
validate.py -- structural checks only.

WHAT WAS REMOVED AND WHY
This file used to run ~15 checks. Ten of them encoded a JUDGEMENT about language
correctness, and every one was found wrong the first time it met real sentences:

    vocab_above_N5                  the level list calls 時 and 日 N3
    kanji_above_N5 / non_jlpt_word  the 79-kanji list rejects 花/赤/机/秋
    pattern_missing                 13 points tested one form of several (ここ vs あそこ)
    register_mismatch               POLITE regex lacked ください/ましょう  (85 false rejects)
    sentence_initial_demonstrative  contradicted two grammar points outright
    pronoun                         bans 彼/彼女, which are ordinary Japanese
    unsuitable_noun                 class table tripped on compound fragments, counter forms
    kana_spelling                   four failed detectors before a curated list half-worked

Each needed a hand-tuned table, a false-positive analysis, and a fix cycle. At N1 the
rule count multiplies and each rule gets harder -- there is no regex for 〜んばかりに
permitting only certain verb forms, or for transitivity pairs like 窓を開ける/窓が開く.

Judgement now belongs to the judge (judge.py), which sees the sentence WITH CONTEXT:
the target grammar point plus the nearest known-good sentences retrieved from the
corpus. That is a far easier and more consistent task than a bare "is this valid?".

WHAT REMAINS, AND WHY IT IS NOT THE SAME KIND OF THING
    duplicate_exact   sha256 of the normalised string. Two identical strings ARE
                      identical -- this is identity, not a threshold that can be tuned.
    duplicate_near    exact content-lemma set comparison. Same.
    not_a_word        does it tokenise to anything at all. Structural.
    no_predicate      is there a verb or adjective. Structural.
    length>20         a display constraint for the corpus, not a language judgement.
    no_pool_word      a COVERAGE query -- did generation use the word it was told to.
                      No model can answer "which N5 words still lack an example".

None of these can be wrong about the language, so none of them need recalibrating when
the corpus moves to N1.

  python validate.py --att attempts8.jsonl --out-prefix x_
"""

import json, os, re, sys, hashlib, collections
from janome.tokenizer import Tokenizer

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

sys.path.insert(0, OUT)
import guardrails as G

tok = Tokenizer()
lex = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))
TG  = json.load(open(os.path.join(DATA, "targets.json"), encoding="utf-8"))
TGT = {t["grammar_point_id"]: t for t in TG["targets"]}
POOLS = G.build_pools(lex)

n5set = {v["text"] for v in lex["n5_vocab"]}
norm = lambda s: re.sub(r'[。．！？、…\s「」『』（）・]', '', s)
sha  = lambda s: hashlib.sha256(norm(s).encode("utf-8")).hexdigest()
KANJI_RX = re.compile(r'[\u4e00-\u9fff]')
ASCII_RX = re.compile(r'^[0-9A-Za-z０-９]+$')
CONTENT = ("名詞", "動詞", "形容詞", "副詞")
LIGHT = {"為る", "有る", "居る", "成る", "する", "ある", "いる", "なる"}


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


ATT    = arg("--att") or os.path.join(RUNS, "attempts.jsonl")
PREFIX = arg("--out-prefix") or ""


def content_set(s):
    out = set()
    for t in tok.tokenize(s):
        p = t.part_of_speech.split(",")
        if p[0] not in CONTENT:
            continue
        if p[0] == "名詞" and p[1] in ("数", "固有名詞"):
            continue
        l = t.base_form if t.base_form not in ("*", "") else t.surface
        if l in LIGHT:
            continue
        out.add(l)
    return out


def structural(s):
    """Returns (hard, soft, note). Nothing here judges the language."""
    hard, soft, note = [], [], []
    toks = list(tok.tokenize(s))

    if len(s) > 20:
        note.append(f"length>{20}:{len(s)}")

    # not_a_word: every content token is ASCII/numeric, or nothing tokenised
    content = [t for t in toks if t.part_of_speech.split(",")[0] in CONTENT]
    if not toks or (content and all(ASCII_RX.match(t.surface) for t in content)):
        hard.append("not_a_word")

    # no_predicate: no verb, adjective or copula anywhere
    if not any(t.part_of_speech.split(",")[0] in ("動詞", "形容詞", "助動詞") for t in toks):
        hard.append("no_predicate")

    return hard, soft, note


if not os.path.exists(ATT):
    print(f"no {os.path.basename(ATT)}"); sys.exit(1)

print(f"validating {ATT}")
print("checks: exact/near duplicate, not_a_word, no_predicate (structural only)")
print("        language judgement is the judge's job -- see judge.py\n")

seen_exact, seen_cs = set(), []
clean, review, rejected = [], [], []
stats = collections.Counter()

for line in open(ATT, encoding="utf-8"):
    try:
        a = json.loads(line)
    except Exception:
        continue
    rows_s = a.get("kept") if a.get("kept") is not None else a.get("sentences")
    if not rows_s:
        continue
    gp = a["grammar_point_id"]
    tgt = TGT.get(gp, {})
    for s in rows_s:
        s = s.strip()
        if not s:
            continue

        h = sha(s)
        if h in seen_exact:
            rejected.append({"japanese": s, "grammar_point_id": gp,
                             "hard": ["duplicate_exact"]})
            stats["dup_exact"] += 1
            continue

        hard, soft, note = structural(s)

        cs = content_set(s)
        best = max((len(cs & t) / len(cs | t) for t in seen_cs[-4000:] if cs or t),
                   default=0.0)
        if best >= 0.8:
            hard.append(f"duplicate_near:{best:.2f}")
            stats["dup_near"] += 1

        req = a.get("required") or []
        if req and not G.required_present(s, [{"text": w, "reading": None} for w in req], tok):
            note.append("no_pool_word")

        row = {"japanese": s, "grammar_point_id": gp,
               "register": tgt.get("formality"), "source": "generated",
               "model": a.get("model"), "job_id": a.get("job_id"),
               "dup_content_max": round(best, 2),
               "hard": hard, "soft": soft, "note": note}
        seen_exact.add(h)
        seen_cs.append(cs)

        if hard:
            rejected.append(row); stats["rejected"] += 1
        elif soft:
            row["status"] = "needs_review"; review.append(row); stats["needs_review"] += 1
        else:
            row["status"] = "validated"; clean.append(row); stats["validated"] += 1


def dump(name, rows):
    with open(os.path.join(RUNS, PREFIX + name), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


dump("sentences.jsonl", clean + review)
dump("rejected.jsonl", rejected)

total = stats["validated"] + stats["needs_review"] + stats["rejected"]
print(f"candidates: {total}")
for k in ("validated", "needs_review", "rejected"):
    print(f"  {k:<14} {stats[k]:>5} ({100*stats[k]/max(total,1):5.1f}%)")
print(f"  duplicates     {stats['dup_exact']} exact, {stats['dup_near']} near")

print("\n-- why rejected --")
c = collections.Counter()
for r in rejected:
    for x in r["hard"]:
        c[x.split(":")[0]] += 1
for k, v in c.most_common():
    print(f"   {k:<24} {v}")

# ---- coverage: a query, not a judgement -------------------------------------
print("\n-- coverage --")
byg = collections.Counter(r["grammar_point_id"] for r in clean + review)
print(f"   grammar points with >=1 accepted: {len(byg)}/{len(TG['targets'])}")

used = set()
for r in clean + review:
    for t in tok.tokenize(r["japanese"]):
        used.add(t.base_form); used.add(t.surface)
hit = n5set & used
print(f"   N5 vocabulary used: {len(hit)}/{len(n5set)} ({100*len(hit)/len(n5set):.1f}%)")

ledger_words = set()
try:
    led = json.load(open(os.path.join(RUNS, "ledger.json"), encoding="utf-8"))
    ledger_words = {w for w, c in (led.get("__global__") or {}).items() if c}
except Exception:
    pass
print(f"   words generation was told to use: {len(ledger_words)}")
print(f"   of those, present in accepted output: {len(ledger_words & used)}")
