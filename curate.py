#!/usr/bin/env python3
"""select.py -- turn judged sentences into rows the loader can accept.

The loader wants jsonl with a `status` of "validated" or "needs_review". This decides
which is which from the judges' verdicts, and is deliberately explicit about the policy
because "what enters the corpus" is a judgement call, not a formatting step.

POLICY
------
  both judges pass                     -> validated
  neither fails, at least one uncertain -> needs_review
  exactly one fails                    -> needs_review   (disagreement is not a pass)
  both fail                            -> EXCLUDED       (known-bad does not load)

Rationale: disagreement between judges is the signal that a sentence needs a human, not
the signal that it is good. Loading it as "validated" would launder a coin-flip into a
claim. Excluding both-fail keeps known-bad sentences out of a corpus whose whole purpose
is to tell learners what is correct.

The status is a claim about EVIDENCE, not about Japanese. Every row here is still a
model's opinion; the human gold set is what turns any of it into ground truth.

  python select.py --in runs/n5_5k/input.jsonl \
                   --luna runs/n5_5k/verdicts_luna.jsonl \
                   --haiku runs/n5_5k/verdicts_haiku.jsonl \
                   --out runs/n5_5k/load_input.jsonl
"""

import argparse
import collections
import json
import os
import sys

GEN = os.path.dirname(os.path.abspath(__file__))


def load(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def keyed(rows, fields):
    return {(r.get("grammar_point_id"), r.get("japanese")): r for r in rows}


def decide(a, b):
    """(status_or_None, note) from two verdicts."""
    if a is None or b is None:
        return "needs_review", ["unjudged: one judge returned no verdict"]
    oa, ob = a.get("overall"), b.get("overall")
    fa = (oa == "fail")
    fb = (ob == "fail")
    ua = a.get("uncertain_dimensions") or []
    ub = b.get("uncertain_dimensions") or []
    if fa and fb:
        return None, ["both judges fail"]
    # "validated" requires an unambiguous pass from BOTH. derive() cannot produce
    # overall=pass alongside uncertain dimensions, but a row that arrived that way
    # (older prompt version, hand edit, a future change to derive) must not be
    # laundered into a clean claim, so require the absence of uncertainty too.
    if oa == "pass" and ob == "pass" and not ua and not ub:
        return "validated", []
    if not fa and not fb:
        unc = sorted(set(ua + ub))
        return "needs_review", ["uncertain: " + ",".join(unc) if unc else "uncertain"]
    # exactly one fails
    who = "luna" if fa else "haiku"
    dims = sorted(set(a.get("failed_dimensions") or []) if fa
                  else set(b.get("failed_dimensions") or []))
    return "needs_review", [f"judges disagree, {who} fails: " + ",".join(dims)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="judge input jsonl")
    ap.add_argument("--luna", required=True)
    ap.add_argument("--haiku", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = load(a.inp)
    A = keyed(load(a.luna), None)
    B = keyed(load(a.haiku), None)
    print(f"input {len(src)}  luna {len(A)}  haiku {len(B)}")

    stat = collections.Counter()
    per = collections.Counter()
    out = []
    for r in src:
        k = (r.get("grammar_point_id"), r.get("japanese"))
        status, note = decide(A.get(k), B.get(k))
        stat[status or "EXCLUDED"] += 1
        if status is None:
            continue
        per[(r["grammar_point_id"], status)] += 1
        row = dict(r)
        row["status"] = status
        row["note"] = note
        # the loader reads `note` as a list of notes; keep the judge evidence with the
        # row so a later reader can see WHY it is needs_review rather than re-judging
        if A.get(k):
            row["judge_luna_overall"] = A[k].get("overall")
            row["judge_luna_target_used"] = A[k].get("target_used")
        if B.get(k):
            row["judge_haiku_overall"] = B[k].get("overall")
            row["judge_haiku_target_used"] = B[k].get("target_used")
        out.append(row)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {'validated (both pass)':<26}{stat['validated']:>6}"
          f"  {stat['validated']/len(src):>6.1%}")
    print(f"  {'needs_review':<26}{stat['needs_review']:>6}"
          f"  {stat['needs_review']/len(src):>6.1%}")
    print(f"  {'EXCLUDED (both fail)':<26}{stat['EXCLUDED']:>6}"
          f"  {stat['EXCLUDED']/len(src):>6.1%}")
    print(f"\n  wrote {len(out)} rows -> {a.out}")

    # per-point spread, so a lopsided corpus is visible before loading
    pts = sorted({k[0] for k in per})
    thin = [(g, sum(v for (gg, s), v in per.items() if gg == g and s == "validated"))
            for g in pts]
    thin = [x for x in thin if x[1] < 5]
    print(f"  points with <5 validated: {len(thin)}/{len(pts)}"
          + (f"  {[g for g, _ in thin[:6]]}" if thin else ""))


if __name__ == "__main__":
    main()
