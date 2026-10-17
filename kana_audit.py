"""
kana_audit.py -- report N5 words written in kana, and emit repairs.

The detection logic lives in guardrails.kana_violations() so validate.py and this
report can never drift apart. An earlier version kept its own copy of the curated
list and silently lost entries (すずしい), which produced two different answers for
the same corpus.

Detection history, recorded so it is not retried:
  1. substring match on readings              -> 84% of the corpus flagged. は is both
     the topic particle and the reading of 歯.
  2. token readings via janome                -> precise, but blind to はこ, which
     the tokeniser splits into は + こ.
  3. substring match, readings >= 2 chars     -> 24%. した inside でした, きた inside
     弾きたい, くる inside くるま.
  4. joining adjacent kana tokens, sub-runs   -> 72%. まし + た joins to ました which
     contains した, recreating failure 1.
  What works: a curated list, guarded by (a) sitting inside a known kana word and
  (b) token boundaries, so は|これ is not read as はこ.

  python kana_audit.py            # report only
  python kana_audit.py --write    # also write kana_fix.json (with source spans)
"""

import json, os, re, sys, collections

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

sys.path.insert(0, OUT)
import guardrails as G

rows = [json.loads(l) for l in open(os.path.join(RUNS, "v11_sentences.jsonl"), encoding="utf-8")]
curated, safe, allowed = G._ortho_data()

print(f"auditing {len(rows)} sentences")
print(f"   curated words that require kanji : {len(curated)} of {len(G.CURATED_KANA)}"
      f" ({len(G.CURATED_KANA)-len(curated)} excluded: kanji outside the N5 set)")
print(f"   legitimately-kana guard words    : {len(safe)}")
print(f"   N5-permitted kanji               : {len(allowed)}\n")

hits = []
per_word = collections.Counter()
for r in rows:
    v = G.kana_violations(r["japanese"])
    if v:
        hits.append((r["japanese"], v))
        for kana, kanji, _a, _b in v:
            per_word[(kana, kanji)] += 1

print(f"sentences affected: {len(hits)} of {len(rows)} ({100*len(hits)/max(len(rows),1):.1f}%)\n")
for (kana, kanji), n in per_word.most_common(40):
    ex = next(s for s, f in hits if any(k == kana and j == kanji for k, j, _a, _b in f))
    print(f"   {kana:<10} -> {kanji:<6} x{n:<3}  {ex}")

print(f"\ndistinct words : {len(per_word)}")
print(f"occurrences    : {sum(per_word.values())}")

if "--write" in sys.argv:
    patch = []
    for s, found in hits:
        fixed = s
        for kana, kanji, a, b in sorted(found, key=lambda x: -x[2]):
            assert fixed[a:b] == kana, (fixed, kana, a, b)
            fixed = fixed[:a] + kanji + fixed[b:]
        patch.append({"before": s, "after": fixed,
                      "spans": [[a, b, kana, kanji] for kana, kanji, a, b in found]})
    json.dump(patch, open(os.path.join(RUNS, "kana_fix.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {len(patch)} repairs to kana_fix.json (with source spans)")
