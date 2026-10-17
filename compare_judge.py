"""
compare_judge.py -- measure the judge against hand labels.

NOTE ON THE GROUND TRUTH: the labels in calib_labels.json were produced by reading
each sentence, not by an external Japanese teacher or a published benchmark. So this
measures agreement with one informed reader. It is enough to catch a judge that is
random or systematically lenient; it is not enough to certify the judge. Treat the
numbers as a floor, and re-measure against real expert labels before shipping.

  python compare_judge.py judged_gemma-4-26B-A4B-it-QAT-GGUF.jsonl
"""

import json, os, sys, collections

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

labels = json.load(open(os.path.join(RUNS, "calib_labels.json"), encoding="utf-8"))
# combined truth: valid Japanese AND actually uses the target grammar point.
L, R = labels["combined_ok"], labels["reasons"]
PAT = labels.get("pattern_reason", {})
order = [json.loads(l)["japanese"] for l in
         open(os.path.join(RUNS, "calib_sample.jsonl"), encoding="utf-8")]
idx_of = {s: str(i + 1) for i, s in enumerate(order)}

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path or not os.path.isabs(path):
    path = os.path.join(OUT, path or "")
judged = [json.loads(l) for l in open(path, encoding="utf-8")]
print(f"judge: {os.path.basename(path)}   {len(judged)} verdicts   "
      f"labels: {len(L)}\n")

tp = tn = fp = fn = missing = 0
agree = []
disagree = []
for j in judged:
    if j.get("verdict") == "no_verdict":
        missing += 1; continue
    n = idx_of.get(j["japanese"])
    if n is None or n not in L:
        missing += 1
        continue
    truth_valid = L[n] == 1
    said_valid = j.get("verdict") == "valid"
    if truth_valid and said_valid:
        tp += 1; agree.append((n, j))
    elif not truth_valid and not said_valid:
        tn += 1; agree.append((n, j))
    else:
        disagree.append((n, j, truth_valid))
        if truth_valid and not said_valid:
            fp += 1        # judge rejected something good
        else:
            fn += 1        # judge accepted something bad

n = tp + tn + fp + fn
prec = tp / max(tp + fp, 1)
rec = tn / max(tn + fn, 1)
print(f"agreement      {n - len(disagree)}/{n}  ({100*(n-len(disagree))/max(n,1):.1f}%)")
print(f"  both valid            {tp}")
print(f"  both invalid          {tn}")
print(f"  judge said invalid, actually valid   {fp}   (false reject)")
print(f"  judge said valid,  actually invalid  {fn}   (missed defect)")
print(f"\n  precision (of 'invalid' verdicts) {prec:.2f}")
print(f"  recall    (of true defects)       {rec:.2f}")
if missing:
    print(f"  ({missing} verdicts had no matching label)")

print("\n-- MISSED DEFECTS (judge said valid, my label says defective) --")
for n_, j, tv in disagree:
    if tv:
        continue
    print(f"  [{n_}] {j['japanese']}")
    print(f"        my reason : {R[n_] or PAT.get(n_, '')}")
    print(f"        judge     : conf={j.get('confidence')} problems={j.get('problems')}")

print("\n-- FALSE REJECTS (judge said invalid, I marked valid) --")
for n_, j, tv in disagree:
    if not tv:
        continue
    print(f"  [{n_}] {j['japanese']}")
    print(f"        judge     : conf={j.get('confidence')} problems={j.get('problems')}")

probs = collections.Counter()
for j in judged:
    for p in (j.get("problems") or []):
        probs[p[:60]] += 1
print(f"\n-- problems the judge reported most --")
for p, c in probs.most_common(12):
    print(f"   {c:>3}  {p}")
