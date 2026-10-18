#!/usr/bin/env python3
"""Frozen regression corpus: make a change measurable instead of arguable.

THE PROBLEM THIS SOLVES
-----------------------
Every measurement in this project has had the same weakness. A run draws a fresh
random set of required words per grammar point and generates ~12 sentences from them,
so a per-point failure rate at n=12 moves 8.3pp per sentence. Comparing two runs means
comparing two fresh draws, and the difference you see is dominated by which words
happened to be drawn. Three times now a change was correctly motivated and then
unmeasurable -- judged rates moved within noise and the two judges did not even agree
which run was better.

WHAT CAN AND CANNOT BE FROZEN
-----------------------------
- Sentences: freezing SENTENCES only tests the judge. Re-judging identical text is
  the clean case, and that is what `run` does.
- Generation: cannot be frozen -- sampling at temperature is the point. But the
  largest single source of cross-run variance is the WORD DRAW, not the sampling.
  `gen` holds the words fixed, so before/after is paired on vocabulary per word.

So: `freeze` records the draw; `run` re-judges frozen sentences (judge changes);
`gen` regenerates from frozen words (pool/frame/regex/POS changes).

USAGE
-----
  python regress.py freeze                      # record the draw (~5s, no API)
  python regress.py run --label base --judge luna     # baseline verdicts
  python regress.py run --label cand --judge luna     # after a change
  python regress.py diff base cand                    # paired comparison
  python regress.py gen --label cand --judge luna     # frozen-word generation
"""

import argparse
import collections
import json
import os
import random
import sys
import time

GEN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GEN)
RUNS = os.path.join(os.path.dirname(GEN), "runs")
REG = os.path.join(RUNS, "regress")
FREEZE_DIR = os.path.join(GEN, "data", "regression")

# how many required words to freeze per grammar point. 12 matches the observed
# per-point sample size, so a frozen run is cost-comparable with a live one.
K_WORDS = 12


# ---------------------------------------------------------------- freeze

def freeze(force=False):
    """Record a deterministic required-word draw per grammar point.

    Deterministic in the grammar point id, so re-running `freeze` reproduces the same
    draw. It does NOT re-run the ledger rotation: the point is to hold vocabulary
    fixed across time, which is the opposite of what the ledger does.
    """
    import guardrails as G

    targets = json.load(open(os.path.join(GEN, "data", "targets.json"),
                             encoding="utf-8"))["targets"]
    lex = json.load(open(os.path.join(GEN, "data", "lexicon.json"), encoding="utf-8"))
    pools = G.build_pools(lex)
    rows = []
    for t in targets:
        gp = t["grammar_point_id"]
        need = G.pos_need(t["pattern"], t.get("meaning_en", ""), gp)
        pool = G.filter_pool(gp, pools.get(need) or pools["noun"])
        if not pool:
            print(f"  {gp:<38} NO POOL (skipped)")
            continue
        # deterministic per point, independent of run order and of the ledger
        rnd = random.Random(f"regress:{gp}")
        words = sorted({(w["text"] if isinstance(w, dict) else str(w)) for w in pool})
        chosen = rnd.sample(words, min(K_WORDS, len(words)))
        rows.append({
            "grammar_point_id": gp,
            "pattern": t.get("pattern", ""),
            "meaning_en": t.get("meaning_en", ""),
            "frame": t.get("frame", ""),
            "check_regex": t.get("check_regex"),
            "pos_check": t.get("pos_check"),
            "required": chosen,
        })
    os.makedirs(FREEZE_DIR, exist_ok=True)
    out = os.path.join(FREEZE_DIR, "inputs.jsonl")

    # The reference set is IMMUTABLE by default. A pool change (the transitive filter
    # added 77 points' worth of edits) would otherwise silently produce a different
    # draw on the next `freeze`, and every baseline comparison against it would be
    # measuring a changed reference rather than a changed code path. Changing the
    # reference is a deliberate act: --force, and the hash is recorded so the change
    # is visible in the diff.
    if os.path.exists(out) and not force:
        old = open(out, encoding="utf-8").read()
        import hashlib
        new = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
        if old != new:
            oh = hashlib.sha256(old.encode()).hexdigest()[:12]
            nh = hashlib.sha256(new.encode()).hexdigest()[:12]
            print(f"REFUSING to overwrite the frozen reference set.\n")
            print(f"  existing : {oh}  ({len(old.splitlines())} points)")
            print(f"  recomputed: {nh}  ({len(rows)} points)")
            print(f"\n  The draw changed -- most likely a pool or CLASS_RULES edit. That is")
            print(f"  expected after a rule change, but it means the baseline no longer")
            print(f"  refers to the same reference set, so it must be a deliberate act:")
            print(f"      python regress.py freeze --force")
            print(f"  and the old baseline should be re-taken, not silently inherited.")
            return
        print(f"frozen set unchanged ({len(rows)} points) -- nothing written")
        return

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # A reference set is only useful if the thing it was frozen AGAINST is recorded
    # with it. Store the fixture hash, the identity of the curated data that produced
    # the draw, and the commit -- otherwise a future reader cannot tell whether a
    # baseline and a fixture belong to the same world.
    import hashlib
    import subprocess as _sp
    raw = open(out, "rb").read()
    try:
        commit = _sp.run(["git", "rev-parse", "HEAD"], cwd=GEN, capture_output=True,
                         text=True, timeout=20).stdout.strip() or None
    except Exception:
        commit = None
    man = {
        "frozen_sha256": hashlib.sha256(raw).hexdigest(),
        "points": len(rows),
        "word_slots": sum(len(r["required"]) for r in rows),
        "k_words_per_point": K_WORDS,
        "schema": 1,
        "creation_commit": commit,
        "note": ("Replaced by `freeze --force` only. A forced replacement archives the "
                 "previous manifest under replaced/ so the change is recorded rather "
                 "than only printed."),
    }
    # the data identity lives in regen_loop; reuse it so the two cannot disagree
    try:
        import regen_loop as _R
        man["data_sha256"] = _R.data_sha256()
        man["prompt_sha256"] = _R.prompt_sha256()
    except Exception as e:
        man["identity_error"] = str(e)[:120]
    mpath = os.path.join(FREEZE_DIR, "MANIFEST.json")
    if os.path.exists(mpath):
        os.makedirs(os.path.join(FREEZE_DIR, "replaced"), exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.replace(mpath, os.path.join(FREEZE_DIR, "replaced", f"MANIFEST-{stamp}.json"))
        print(f"  archived the previous manifest -> replaced/MANIFEST-{stamp}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"  manifest -> {mpath}")
    print(f"    frozen_sha256 {man['frozen_sha256'][:16]}  "
          f"data_sha256 {man.get('data_sha256','?')}  commit {commit or '?'}")
    tot = sum(len(r["required"]) for r in rows)
    print(f"froze {len(rows)} points, {tot} word slots -> {out}")
    print(f"  re-running freeze reproduces this exactly (seeded by grammar point id)")


def load_frozen():
    p = os.path.join(FREEZE_DIR, "inputs.jsonl")
    if not os.path.exists(p):
        sys.exit("no frozen draw -- run: python regress.py freeze")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- run (judge)

def _out_dir(label):
    d = os.path.join(REG, label)
    os.makedirs(d, exist_ok=True)
    return d


def run_judge(label, judge, limit=0, reuse=None):
    """Judge a fixed sentence set and store verdicts for `diff`.

    With --reuse, judges an existing verdict file's sentences (a judge-change test).
    Otherwise judges the highest-fidelity corpus available, so the SET is stable.
    """
    import rubric_judge as RJ
    import guardrails as G

    d = _out_dir(label)
    if reuse:
        src = reuse
    else:
        # the current corpus, flattened to unique (gp, sentence) pairs
        src = os.path.join(RUNS, "vtfix", "input.jsonl")
        if not os.path.exists(src):
            sys.exit(f"no corpus at {src}")
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("japanese")]
    if limit:
        rows = rows[:limit]
    print(f"{label}: judging {len(rows)} frozen sentences with {judge}")

    out = os.path.join(d, f"verdicts_{judge}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                r = json.loads(l); done.add((r["grammar_point_id"], r["japanese"]))
            except Exception:
                pass
    todo = [r for r in rows if (r["grammar_point_id"], r["japanese"]) not in done]
    print(f"  {len(done)} already judged, {len(todo)} to go")

    tg = {t["grammar_point_id"]: t for t in
          json.load(open(os.path.join(GEN, "data", "targets.json"),
                         encoding="utf-8"))["targets"]}
    log = open(out, "a", encoding="utf-8")
    t0, cost, n, nod = time.time(), 0.0, 0, 0
    for k in range(0, len(todo), 6):
        b = todo[k:k + 6]
        try:
            res, usage, finish = RJ.judge_batch(b, model=judge)
            cost += float(usage.get("cost", 0) or 0)
            for src_r, v in zip(b, res):
                if v is None:
                    nod += 1; continue
                rec = dict(src_r)
                rec["model"] = judge
                rec.update({k2: v[k2] for k2 in RJ.DIMENSIONS})
                rec.update({"span": v.get("span", ""), "reason": v.get("reason", ""),
                            "reason_codes": v["reason_codes"],
                            "failed_dimensions": v["failed_dimensions"],
                            "overall": v["overall"],
                            "span_ok": v["span_ok"],
                            "span_pattern_ok": v.get("span_pattern_ok"),
                            "evidence_conflict": v.get("evidence_conflict", False)})
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        except Exception as e:
            print(f"  {type(e).__name__}: {str(e)[:110]}", flush=True)
            time.sleep(3)
        log.flush()
        if (k // 6) % 20 == 0:
            print(f"    b{k//6+1:>4}  {n} written  {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"  wrote {n}  no_verdict {nod}  ${cost:.5f}  {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- gen (frozen words)

def run_gen(label, judge, model=None, points=0, per_word=1):
    """Regenerate from the FROZEN words, so before/after is paired on vocabulary.

    Shells out to generate2.py with --fixwords, which is the flag built for exactly
    this: it freezes the required-word list per point so two runs -- e.g. two models --
    see an identical vocabulary. The first version of this function reimplemented the
    call locally and got it wrong (it passed required words without the `meaning` key
    build_user needs), which is what happens when tested code is duplicated instead of
    reused. --fixwords also means the generator's own pool/ledger/retry logic is what
    runs, so a generator change is measured on the real path.
    """
    import subprocess

    frozen = load_frozen()
    if points:
        frozen = frozen[:points]
    d = _out_dir(label)
    fw = os.path.join(d, "fixwords.json")
    with open(fw, "w", encoding="utf-8") as f:
        json.dump({r["grammar_point_id"]: r["required"] for r in frozen},
                  f, ensure_ascii=False, indent=1)
    attempts = os.path.join(d, "attempts.jsonl")
    onlylist = os.path.join(d, "points.json")
    with open(onlylist, "w", encoding="utf-8") as f:
        json.dump([r["grammar_point_id"] for r in frozen], f, ensure_ascii=False)

    cmd = [sys.executable, os.path.join(GEN, "generate2.py"),
           "--fixwords", fw, "--onlylist", onlylist, "--out", attempts, "--append"]
    if model:
        cmd += ["--model", model]
    print(f"  {' '.join(cmd[-6:])}")
    r = subprocess.run(cmd, cwd=GEN)
    if r.returncode != 0:
        sys.exit(f"generate2 exited {r.returncode}")
    print(f"\nwrote {attempts}")
    print(f"next: judge it and `diff` against another label")


def judge_gen(label, judge, limit=0):
    """Judge the sentences a `gen` run produced, into the same label dir."""
    import rubric_judge as RJ

    d = _out_dir(label)
    src = os.path.join(d, "attempts.jsonl")
    if not os.path.exists(src):
        sys.exit(f"no attempts at {src} -- run `gen` first")
    tg = {t["grammar_point_id"]: t for t in
          json.load(open(os.path.join(GEN, "data", "targets.json"),
                         encoding="utf-8"))["targets"]}
    rows, seen = [], set()
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        gp = r["grammar_point_id"]
        t = tg.get(gp, {})
        for s in r.get("kept", []):
            s = s.strip()
            if not s or (gp, s) in seen:
                continue
            seen.add((gp, s))
            rows.append({"japanese": s, "grammar_point_id": gp,
                         "pattern": t.get("pattern", ""),
                         "meaning_en": t.get("meaning_en", ""),
                         "frame": t.get("frame", "")})
    if limit:
        rows = rows[:limit]
    print(f"{label}: judging {len(rows)} generated sentences with {judge}")
    out = os.path.join(d, f"verdicts_{judge}.jsonl")
    log = open(out, "a", encoding="utf-8")
    t0, cost, n, nod = time.time(), 0.0, 0, 0
    for k in range(0, len(rows), 6):
        b = rows[k:k + 6]
        try:
            res, usage, finish = RJ.judge_batch(b, model=judge)
            cost += float(usage.get("cost", 0) or 0)
            for src_r, v in zip(b, res):
                if v is None:
                    nod += 1; continue
                rec = dict(src_r)
                rec["model"] = judge
                rec.update({k2: v[k2] for k2 in RJ.DIMENSIONS})
                rec.update({"span": v.get("span", ""), "reason": v.get("reason", ""),
                            "reason_codes": v["reason_codes"],
                            "failed_dimensions": v["failed_dimensions"],
                            "overall": v["overall"], "span_ok": v["span_ok"],
                            "span_pattern_ok": v.get("span_pattern_ok"),
                            "evidence_conflict": v.get("evidence_conflict", False)})
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        except Exception as e:
            print(f"  {type(e).__name__}: {str(e)[:110]}", flush=True)
            time.sleep(3)
        log.flush()
        if (k // 6) % 20 == 0:
            print(f"    b{k//6+1:>4}  {n} written  {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"  wrote {n}  no_verdict {nod}  ${cost:.5f}  {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- diff

def _key(r):
    return (r.get("grammar_point_id"), r.get("japanese"))


def _load_verdicts(d):
    p = os.path.join(REG, d, "verdicts_luna.jsonl")
    if not os.path.exists(p):
        cands = [f for f in os.listdir(os.path.join(REG, d)) if f.startswith("verdicts_")]
        if not cands:
            sys.exit(f"no verdicts in {d}")
        p = os.path.join(REG, d, cands[0])
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()], p


def diff(a, b):
    """PAIRED comparison: only sentences present in both, verdicts compared per sentence.

    This is the whole point. A paired test on identical text has no draw variance, so
    a small effect is resolvable -- unlike two fresh runs, where one sentence at n=12
    is worth 8.3pp. McNemar's exact test on the discordant pairs, computed here so the
    number does not depend on trusting a summary.
    """
    A, pa = _load_verdicts(a)
    B, pb = _load_verdicts(b)
    am = {_key(r): r for r in A}
    bm = {_key(r): r for r in B}
    both = sorted(set(am) & set(bm))
    print(f"A = {pa}  ({len(A)} verdicts)")
    print(f"B = {pb}  ({len(B)} verdicts)")
    print(f"paired on {len(both)} identical sentences "
          f"(A-only {len(set(am)-set(bm))}, B-only {len(set(bm)-set(am))})")
    if not both:
        sys.exit("nothing paired -- the two labels judged different sentence sets")

    # each field has its own "bad" encoding -- comparing them all as == "fail" silently
    # reported span_ok as 0.0%/0.0%, because span_ok is True/False/None, not a string.
    # span_ok=False means the judge QUOTED A SPAN THAT IS NOT IN THE SENTENCE, which is
    # the one signal here that is unambiguous judge error rather than a policy call.
    def _bad(dim, val):
        if dim == "overall":
            return val == "fail"
        if dim == "target_used":
            return val == "fail"
        if dim == "span_ok":
            return val is False
        if dim == "evidence_conflict":
            return bool(val)
        return val == "fail"

    DIMS = ("target_used", "overall", "span_ok", "evidence_conflict")
    for dim in DIMS:
        a_fail = b1 = b2 = b3 = b4 = 0
        for k in both:
            x = _bad(dim, am[k].get(dim)); y = _bad(dim, bm[k].get(dim))
            if x and not y: b1 += 1
            if y and not x: b2 += 1
            if x and y: b3 += 1
            if not x and not y: b4 += 1
        n = len(both)
        ra, rb = (b1 + b3) / n, (b2 + b3) / n
        print(f"\n  {dim:<20} A {ra:>6.1%}  ->  B {rb:>6.1%}   ({rb-ra:+.1%})")
        print(f"     {'A ok -> B fail':<22}{b2:>5}   (regressions)")
        print(f"     {'A fail -> B ok':<22}{b1:>5}   (fixes)")
        print(f"     {'both fail / both ok':<22}{b3:>5} / {b4}")
        if b1 + b2:
            p = _mcnemar_exact(b1, b2)
            print(f"     McNemar exact p = {p:.2e}   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    # per-point movement, and a CLUSTERED analysis.
    #
    # The 938 sentences are not 938 independent samples: they come from 77 grammar
    # points that share regexes, frames and vocabulary constraints, so failures within
    # a point are correlated and a sentence-level p-value overstates the evidence for
    # generalising across points. Measured on luna-vs-haiku, clustering costs
    # target_used about an order of magnitude (p=1.2e-02 -> 4.1e-02) and overall
    # almost nothing (2.8e-14 -> 1.2e-06). Both survive; the honest number is the
    # clustered one.
    #
    # Direction convention, which is easy to get backwards:
    #   v[0] = luna fails, haiku passes     (luna is HARSHER here)
    #   v[1] = haiku fails, luna passes     (haiku is HARSHER here)
    # so a point counts toward "A better" when v[1] > v[0].
    per = collections.defaultdict(lambda: [0, 0, 0])
    for k in both:
        gp = k[0]
        x = am[k].get("target_used") == "fail"
        y = bm[k].get("target_used") == "fail"
        per[gp][0] += x and not y
        per[gp][1] += y and not x
        per[gp][2] += 1

    a_better = sorted((gp for gp, v in per.items() if v[1] > v[0]),
                      key=lambda g: -(per[g][1] - per[g][0]))
    b_better = sorted((gp for gp, v in per.items() if v[0] > v[1]),
                      key=lambda g: -(per[g][0] - per[g][1]))
    tied = [gp for gp, v in per.items() if v[0] == v[1]]
    print(f"\n  target_used by grammar point: A better on {len(a_better)}, "
          f"B better on {len(b_better)}, tied {len(tied)}  (of {len(per)})")
    m = len(a_better) + len(b_better)
    if m:
        print(f"     sign test over the {m} differing points: "
              f"p = {_sign_test(len(a_better), m):.3e}")
    if a_better:
        print(f"     A better: " + ", ".join(f"{g} +{per[g][1]-per[g][0]}"
                                             for g in a_better[:6]))
    if b_better:
        print(f"     B better: " + ", ".join(f"{g} +{per[g][0]-per[g][1]}"
                                             for g in b_better[:6]))

    # clustered bootstrap: resample POINTS, not sentences.
    #
    # ESTIMAND, stated explicitly because an interval without units is uninterpretable:
    #   statistic  = (number of sentences where A fails and B passes)
    #              - (number of sentences where B fails and A passes)
    #   unit       = discordant SENTENCES (a count, not a percentage point)
    #   sign       = negative means A has fewer failures than B, i.e. A is better
    #   scale      = resampling N points WITH replacement, N = number of points present,
    #                so each replicate covers the same number of points as the data
    #   replicates = 5000, seed = 0 (deterministic: same input -> same interval)
    #   interval   = percentile method, 2.5th and 97.5th percentiles of the replicates
    #   p-value    = fraction of replicates with statistic <= 0 (one-sided; a
    #                two-sided equivalent is 2x that when the mean is on one side)
    #
    # It is NOT a percentage-point difference and NOT a per-sentence rate. The
    # magnitude scales with corpus size: the same effect on a 2x larger corpus gives a
    # roughly 2x wider interval in these units.
    #
    # Resampling points supports generalising ACROSS COMPARABLE POINTS, not across all
    # Japanese grammar: these 77 are a defined curriculum, not a random sample.
    N_BOOT, SEED = 5000, 0
    pts = list(per.values())
    rnd = random.Random(SEED)
    d = []
    for _ in range(N_BOOT):
        s = [rnd.choice(pts) for _ in pts]
        d.append(sum(v[0] for v in s) - sum(v[1] for v in s))
    d.sort()
    lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
    p_le0 = sum(1 for x in d if x <= 0) / len(d)
    print(f"\n  clustered bootstrap")
    print(f"     estimand : discordant SENTENCE count, A-fail-minus-B-fail "
          f"(negative = A better)")
    print(f"     procedure: resample the {len(per)} grammar points with replacement, "
          f"{N_BOOT} replicates, seed {SEED}")
    print(f"     interval : percentile 2.5-97.5 of the replicates")
    print(f"     result   : mean {sum(d)/len(d):+.1f}  95% CI [{lo:+d}, {hi:+d}]  "
          f"P(diff<=0) = {p_le0:.4f}")
    print(f"     relates to : {len(a_better)} vs {len(b_better)} points, not "
          f"{len(both)} sentences")
    print(f"     does NOT say: anything about all Japanese grammar -- these 77 points "
          f"are a curriculum")


def _sign_test(k, n):
    """Two-sided exact sign test: k successes out of n."""
    from math import comb
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def _mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant pairs b and c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze", help="record the deterministic required-word draw")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="judge a fixed sentence set")
    p.add_argument("--label", required=True)
    p.add_argument("--judge", default="openai/gpt-6-luna")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reuse", default=None, help="judge these sentences instead of the corpus")

    p = sub.add_parser("gen", help="regenerate from the frozen words")
    p.add_argument("--label", required=True)
    p.add_argument("--judge", default="openai/gpt-6-luna")
    p.add_argument("--model", default=None)
    p.add_argument("--points", type=int, default=0)
    p.add_argument("--per-word", type=int, default=1)

    p = sub.add_parser("judge-gen", help="judge a gen run's output")
    p.add_argument("--label", required=True)
    p.add_argument("--judge", default="openai/gpt-6-luna")
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("diff", help="paired comparison of two labels")
    p.add_argument("a"); p.add_argument("b")

    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.force)
    elif a.cmd == "run":
        run_judge(a.label, a.judge, a.limit, a.reuse)
    elif a.cmd == "gen":
        run_gen(a.label, a.judge, a.model, a.points, a.per_word)
    elif a.cmd == "judge-gen":
        judge_gen(a.label, a.judge, a.limit)
    elif a.cmd == "diff":
        diff(a.a, a.b)


if __name__ == "__main__":
    main()
