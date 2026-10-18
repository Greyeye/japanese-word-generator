#!/usr/bin/env python3
"""goldsample.py -- build a stratified human-labelling set from judged sentences.

WHY STRATIFIED, NOT RANDOM
--------------------------
A random sample of 5,162 would be ~45% both-pass and would tell you little about the
cases that matter: where the judges DISAGREE. The set is built in cells so each
judgement buys information:

  S1  both pass                      -- low-yield, kept small; a sanity check on the
                                        judges agreeing for the right reasons
  S2  haiku fails natural/coherence,
      luna passes                    -- THE cell. This is where the 5.4x gap lives and
                                        where the loader decision turns
  S3  luna fails, haiku passes       -- the reverse direction, to catch luna being
                                        right where haiku missed it
  S4  both fail                      -- checks the exclusion rule is not throwing away
                                        sentences that are actually fine

Each cell is spread across grammar points, and the point-level naturalness gap is
sampled deliberately rather than left to chance.

WHO LABELS IT MATTERS
---------------------
Every row carries a `labeller` field, and the output says plainly what it is. A gold set
labelled by an informed learner is not native-speaker ground truth, and recording the
distinction is the whole point of this project's approach to evidence -- a claim must not
outrun what supports it. Fill `labeller` honestly; the analysis reports it back.

  python goldsample.py --out runs/n5_5k/gold --per-cell 60
"""

import argparse
import collections
import json
import os
import random
import sys

GEN = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def keyed(rows):
    return {(r.get("grammar_point_id"), r.get("japanese")): r for r in rows}


def cell_of(l, h):
    """Which stratification cell does this sentence fall in?"""
    def dims(v):
        return set((v or {}).get("failed_dimensions") or [])
    lo = (l or {}).get("overall")
    ho = (h or {}).get("overall")
    lf, hf = lo == "fail", ho == "fail"
    if not lf and not hf:
        return "S1_both_pass"
    if lf and hf:
        return "S4_both_fail"
    if hf and not lf and (dims(h) & {"natural", "meaning_coherent"}):
        return "S2_haiku_natural_luna_ok"
    if lf and not hf:
        return "S3_luna_fail_haiku_ok"
    if hf and not lf:
        return "S2b_haiku_fail_other"
    return "S5_uncertain"


def spread(rows, n, seed=0):
    """Take up to n rows, spread round-robin across grammar points.

    Round-robin rather than random so no single point dominates the sample; that is the
    failure mode the clustered statistics already flagged elsewhere in this project.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r["grammar_point_id"]].append(r)
    rnd = random.Random(seed)
    for v in by.values():
        rnd.shuffle(v)
    out, gp = [], sorted(by)
    i = 0
    while len(out) < n and any(by[g] for g in gp):
        g = gp[i % len(gp)]
        if by[g]:
            out.append(by[g].pop())
        i += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=r"H:\jlpt\runs\n5_5k\input.jsonl")
    ap.add_argument("--luna", default=r"H:\jlpt\runs\n5_5k\verdicts_luna.jsonl")
    ap.add_argument("--haiku", default=r"H:\jlpt\runs\n5_5k\verdicts_haiku.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = load(a.inp)
    L, H = keyed(load(a.luna)), keyed(load(a.haiku))

    cells = collections.defaultdict(list)
    for r in src:
        k = (r["grammar_point_id"], r["japanese"])
        cells[cell_of(L.get(k), H.get(k))].append(r)

    picked = []
    for name in sorted(cells):
        got = spread(cells[name], a.per_cell, a.seed)
        print(f"  {name:<28} avail {len(cells[name]):>5}  sampled {len(got):>3}")
        picked += [(name, r) for r in got]

    rnd = random.Random(a.seed)
    rnd.shuffle(picked)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jsonl = a.out + ".jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, (cell, r) in enumerate(picked, 1):
            k = (r["grammar_point_id"], r["japanese"])
            f.write(json.dumps({
                "id": i,
                "cell": cell,
                "japanese": r["japanese"],
                "grammar_point_id": r["grammar_point_id"],
                "pattern": r.get("pattern", ""),
                "meaning_en": r.get("meaning_en", ""),
                "luna_overall": (L.get(k) or {}).get("overall"),
                "luna_dims": (L.get(k) or {}).get("failed_dimensions"),
                "luna_reason": ((L.get(k) or {}).get("reason") or "")[:200],
                "haiku_overall": (H.get(k) or {}).get("overall"),
                "haiku_dims": (H.get(k) or {}).get("failed_dimensions"),
                "haiku_reason": ((H.get(k) or {}).get("reason") or "")[:200],
                # ---- for the labeller ----
                "verdict": "",          # OK | WRONG | UNSURE
                "correction": "",       # if WRONG, what it should say
                "defect": "",           # grammar | naturalness | vocabulary | counter | other
                "labeller": "",         # WHO judged this. fill it in.
                "notes": "",
            }, ensure_ascii=False) + "\n")

    md = a.out + ".md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Gold set — human labelling\n\n")
        f.write("Fill `verdict` (OK / WRONG / UNSURE) for each row. "
                "Add `correction` when WRONG.\n\n")
        f.write("**Record who is labelling** — in the `labeller` column of the jsonl, "
                "or state it here: `labeller: ______`.\n\n")
        f.write("A set labelled by an informed learner is evidence, but it is not "
                "native-speaker ground truth, and the analysis reports which it was. "
                "That distinction is the point.\n\n")
        f.write(f"{len(picked)} sentences, stratified:\n\n")
        c = collections.Counter(cell for cell, _ in picked)
        for k, v in sorted(c.items()):
            f.write(f"- `{k}` — {v}\n")
        f.write("\n---\n\n")
        for i, (cell, r) in enumerate(picked, 1):
            k = (r["grammar_point_id"], r["japanese"])
            lv, hv = L.get(k) or {}, H.get(k) or {}
            f.write(f"### {i}. {r['japanese']}\n\n")
            f.write(f"- point: `{r['grammar_point_id']}` — {r.get('pattern','')}\n")
            f.write(f"- cell: `{cell}`\n")
            f.write(f"- luna: **{lv.get('overall')}** "
                    f"{'fail:' + ','.join(lv.get('failed_dimensions') or []) if lv.get('failed_dimensions') else ''}\n")
            f.write(f"  - {(lv.get('reason') or '')[:180]}\n")
            f.write(f"- haiku: **{hv.get('overall')}** "
                    f"{'fail:' + ','.join(hv.get('failed_dimensions') or []) if hv.get('failed_dimensions') else ''}\n")
            f.write(f"  - {(hv.get('reason') or '')[:180]}\n")
            f.write(f"- **verdict: ______**  correction: ______  defect: ______\n\n")
    print(f"\n  {len(picked)} rows -> {jsonl}")
    print(f"                  -> {md}")


if __name__ == "__main__":
    main()
