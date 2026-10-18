#!/usr/bin/env python3
"""Tests for regress.py -- the statistics, and import/runtime isolation.

Two things this guards:

  1. The statistics have KNOWN answers on synthetic input. An interval or p-value that
     is merely plausible is not evidence. Every expectation here is computed by hand.
  2. Importing or dry-running the tooling has NO side effects. `generate2` used to run a
     full paid generation at import; the AST test in test_regen_model.py pins the
     structure, this pins the observable behaviour (an AST can pass while behaviour
     still surprises, and vice versa).

Run:  python test_regress.py
"""

import json
import subprocess
import sys
import shutil
import tempfile
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import regress as R

FAILS = []


def check(name, got, want, tol=None):
    if tol is not None and isinstance(got, float):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILS.append(name)


print("exact McNemar (hand-computed expectations)")
print()

# two-sided exact McNemar: p = 2 * sum_{i=0}^{min(b,c)} C(n,i) / 2^n
# b=0, c=10 -> 2 * C(10,0)/1024 = 2/1024
check("mcnemar(0, 10)", R._mcnemar_exact(0, 10), 2 / 1024)
check("mcnemar(10, 0)", R._mcnemar_exact(10, 0), 2 / 1024)
# perfectly balanced -> the two tails cover everything -> probability capped at 1
check("mcnemar(5, 5)", R._mcnemar_exact(5, 5), 1.0)
# no discordance at all -> nothing to test
check("mcnemar(0, 0)", R._mcnemar_exact(0, 0), 1.0)
# b=1, c=9 -> 2 * (C(10,0)+C(10,1))/1024 = 2*11/1024
check("mcnemar(1, 9)", R._mcnemar_exact(1, 9), 2 * 11 / 1024)
# b=2, c=8 -> 2 * (1+10+45)/1024
check("mcnemar(2, 8)", R._mcnemar_exact(2, 8), 2 * 56 / 1024)

print()
print("exact sign test (hand-computed expectations)")
print()

check("sign(5, 10) all one side is impossible", R._sign_test(5, 10), 1.0)
check("sign(10, 10)", R._sign_test(10, 10), 2 / 1024)
check("sign(0, 10)", R._sign_test(0, 10), 2 / 1024)
check("sign(0, 0)", R._sign_test(0, 0), 1.0)
# 7 vs 3 on 10 -> 2 * (C(10,0..3))/1024 = 2 * (1+10+45+120)/1024
check("sign(7, 10)", R._sign_test(7, 10), 2 * 176 / 1024)

print()
print("clustered bootstrap: known synthetic answer")
print()

# Construct points with a KNOWN discordance structure. Six points:
#   point A: 3 A-only, 0 B-only
#   point B: 0 A-only, 4 B-only
#   points C-F: no discordance
# Total statistic on the observed data = (3) - (4) = -1, i.e. A better by 1.
# Resampling points changes which points are drawn, so the replicated statistic varies
# with multiplicity; the mean over replicates should sit near -1.
import random as _r


def boot(per_like, n=5000, seed=0):
    pts = list(per_like)
    rnd = _r.Random(seed)
    d = []
    for _ in range(n):
        s = [rnd.choice(pts) for _ in pts]
        d.append(sum(v[0] for v in s) - sum(v[1] for v in s))
    d.sort()
    return d


pts = [[3, 0], [0, 4], [0, 0], [0, 0], [0, 0], [0, 0]]
d = boot(pts)
observed = sum(v[0] for v in pts) - sum(v[1] for v in pts)
mean = sum(d) / len(d)
check("observed statistic on synthetic data", observed, -1)

# The resampled mean is unbiased IN EXPECTATION, but a finite replicate count carries
# Monte Carlo error -- asserting exact equality here was my error, not the code's
# (it returned -1.016 on 5000 replicates). Bound it by the standard error of the mean
# over replicates, so the test states the real property: unbiased within MC error.
import math as _m
se = (_m.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1)) / _m.sqrt(len(d)))
print(f"        (MC standard error of the bootstrap mean = {se:.4f})")
check("bootstrap mean is unbiased within 4 MC standard errors",
      abs(mean - observed) <= 4 * se, True)

# determinism: the seed must make the interval reproducible
d2 = boot(pts)
check("bootstrap is deterministic under a fixed seed", d == d2, True)

# a degenerate case: every point identical -> every replicate identical
pts_deg = [[2, 1]] * 5
dd = boot(pts_deg)
check("degenerate case has a zero-width interval",
      len({x for x in dd}), 1)
check("degenerate value is the observed statistic",
      dd[0], 2 * 5 - 1 * 5)

# and with no discordance at all, the interval must straddle zero
pts_zero = [[0, 0]] * 8
dz = boot(pts_zero)
check("no discordance -> interval is exactly [0, 0]", (min(dz), max(dz)), (0, 0))

print()
print("import and run isolation")
print()


def run(code, cwd=None, timeout=120):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd or HERE,
                          capture_output=True, text=True, timeout=timeout)


# 1. importing the generator must not generate, print a run banner, or write output
before = {}
for f in ("attempts2.jsonl",):
    p = os.path.join(os.path.dirname(HERE), "runs", f)
    before[f] = os.path.getsize(p) if os.path.exists(p) else None

r = run("import sys; sys.path.insert(0, r'%s'); import generate2; print('CLEAN')"
        % HERE.replace("\\", "/"))
check("importing generate2 exits 0", r.returncode, 0)
check("importing generate2 prints only its marker", r.stdout.strip(), "CLEAN")
check("importing generate2 emits no run banner", "DONE" not in r.stdout
      and "points=" not in r.stdout, True)
check("importing generate2 stderr is empty", r.stderr.strip(), "")

# 2. importing must not create or modify the default output file
after = {}
for f in ("attempts2.jsonl",):
    p = os.path.join(os.path.dirname(HERE), "runs", f)
    after[f] = os.path.getsize(p) if os.path.exists(p) else None
check("importing generate2 does not touch the dataset", after, before)

# 3. no network: import with sockets disabled must still succeed
guard = (
    "import socket\n"
    "class Blocked(Exception): pass\n"
    "def deny(*a, **k): raise Blocked('network access at import')\n"
    "socket.socket.connect = deny\n"
    "socket.create_connection = deny\n"
    "import sys; sys.path.insert(0, r'%s')\n"
    "import generate2, regress, rubric_judge, regen_loop, guardrails\n"
    "print('NO_NETWORK_NEEDED')\n" % HERE.replace("\\", "/")
)
r = run(guard)
check("importing every module needs no network", r.stdout.strip(), "NO_NETWORK_NEEDED")
check("no network attempt raised", "Blocked" not in r.stderr, True)

# 4. importing the regression tool must not freeze, judge or generate
r = run("import sys; sys.path.insert(0, r'%s'); import regress; print('CLEAN')"
        % HERE.replace("\\", "/"))
check("importing regress prints only its marker", r.stdout.strip(), "CLEAN")

print()
print("frozen reference set is immutable by default")
print()

# A pool change would otherwise silently produce a different draw on the next freeze,
# and every baseline comparison would then measure a changed reference rather than a
# changed code path. Freeze must refuse unless --force.
fsrc = open(os.path.join(HERE, "regress.py"), encoding="utf-8").read()
check("freeze refuses to overwrite silently", "REFUSING to overwrite" in fsrc, True)
check("freeze takes --force", '"--force"' in fsrc, True)
check("freeze reports the old and new hashes",
      "existing :" in fsrc and "recomputed:" in fsrc, True)
check("freeze is a no-op when unchanged",
      "frozen set unchanged" in fsrc, True)

# and the reference set on disk must still hash to the value the baseline was taken
# against, so committed baselines and fixtures can never silently disagree
import hashlib as _h
ref = os.path.join(HERE, "data", "regression", "inputs.jsonl")
base = os.path.join(HERE, "data", "regression", "baseline.json")
check("the frozen set exists", os.path.exists(ref), True)
check("the baseline exists", os.path.exists(base), True)
if os.path.exists(ref) and os.path.exists(base):
    raw = open(ref, "rb").read()
    check("frozen set has 77 points",
          len([l for l in raw.decode("utf-8").splitlines() if l.strip()]), 77)
    # both freeze runs must agree: re-freezing a stable tree writes nothing
    r = subprocess.run([sys.executable, "regress.py", "freeze"], cwd=HERE,
                       capture_output=True, text=True, timeout=120)
    check("re-freezing a stable tree reports no change",
          "unchanged" in r.stdout, True)
    check("re-freezing does not rewrite the file",
          _h.sha256(open(ref, "rb").read()).hexdigest(),
          _h.sha256(raw).hexdigest())

print()
print("the baseline is auditable")
print()

# A reference set is only useful if what it was frozen AGAINST is recorded with it,
# or a later reader cannot tell whether a baseline and a fixture belong to one world.
man_p = os.path.join(HERE, "data", "regression", "MANIFEST.json")
check("a manifest is written", os.path.exists(man_p), True)
if os.path.exists(man_p):
    man = json.loads(open(man_p, encoding="utf-8").read())
    import hashlib as _hh
    actual = _hh.sha256(open(ref, "rb").read()).hexdigest()
    check("the manifest records the fixture hash",
          man.get("frozen_sha256"), actual)
    check("the manifest records the data identity",
          bool(man.get("data_sha256")), True)
    check("the manifest records the creation commit",
          bool(man.get("creation_commit")), True)
    check("the manifest records the schema", man.get("schema"), 1)
    check("the manifest counts match the fixture",
          (man.get("points"), man.get("word_slots")), (77, 913))

# a forced replacement must leave a record, not just console output
rsrc = open(os.path.join(HERE, "regress.py"), encoding="utf-8").read()
check("a forced replacement archives the old manifest",
      "archived the previous manifest" in rsrc, True)
check("the archive lives under replaced/", "replaced" in rsrc, True)

print()
print("the manifest invariant is enforced, not just recorded")
print()

# A manifest is only worth having if its relationship to the fixture is checked. These
# are the reviewer's invariants: stale hashes, line-ending sensitivity, forced
# replacement archiving, and the draw-vs-baseline distinction.
import importlib.util as _ilu2
_s2 = _ilu2.spec_from_file_location("rg2", os.path.join(HERE, "regress.py"))
rg = _ilu2.module_from_spec(_s2)
_s2.loader.exec_module(rg)

check("a consistent manifest verifies", rg.verify_manifest(verbose=False), [])

# 1. a tampered fixture must be detected
fixture = os.path.join(HERE, "data", "regression", "inputs.jsonl")
raw = open(fixture, "rb").read()
try:
    open(fixture, "wb").write(raw + b"\n")
    probs = rg.verify_manifest(verbose=False)
    check("a tampered fixture fails verification", len(probs) > 0, True)
    check("...and the problem names the fixture hash",
          any("sha256 mismatch" in x for x in probs), True)
finally:
    open(fixture, "wb").write(raw)

# 2. line endings must NOT change the fingerprint -- the hash is over content
try:
    open(fixture, "wb").write(raw.replace(b"\n", b"\r\n"))
    recorded = json.loads(open(os.path.join(HERE, "data", "regression", "MANIFEST.json"),
                               encoding="utf-8").read())["frozen_sha256"]
    check("CRLF does not change the canonical hash",
          rg.fixture_sha256(fixture), recorded)
    check("CRLF still verifies", rg.verify_manifest(verbose=False), [])
finally:
    open(fixture, "wb").write(raw)

# 3. the hash must be over content, and say so
mjson = json.loads(open(os.path.join(HERE, "data", "regression", "MANIFEST.json"),
                        encoding="utf-8").read())
check("the manifest states the hash is content-normalised",
      "normalised" in (mjson.get("hash_note") or ""), True)

# 4. writing is atomic, and the force path archives -- tested in a TEMP dir so the
#    suite never mutates the repository it is checking
import glob as _g
_real_freeze = rg.FREEZE_DIR
_tmp_freeze = tempfile.mkdtemp()
try:
    shutil.copy(fixture, os.path.join(_tmp_freeze, "inputs.jsonl"))
    shutil.copy(os.path.join(HERE, "data", "regression", "MANIFEST.json"),
                os.path.join(_tmp_freeze, "MANIFEST.json"))   # so a force has one to archive
    rg.FREEZE_DIR = _tmp_freeze
    rg.freeze(force=True)
    leftovers = [x for x in _g.glob(os.path.join(_tmp_freeze, ".tmp-*"))]
    check("no temp files left after an atomic write", leftovers, [])
    check("the manifest verifies in the temp dir", rg.verify_manifest(verbose=False), [])
    check("a force archived the previous manifest",
          os.path.isdir(os.path.join(_tmp_freeze, "replaced")), True)
    check("freeze is deterministic after a force",
          rg.fixture_sha256(os.path.join(_tmp_freeze, "inputs.jsonl")),
          mjson["frozen_sha256"])
finally:
    rg.FREEZE_DIR = _real_freeze
    shutil.rmtree(_tmp_freeze, ignore_errors=True)

# 5. the two consequences are distinguished in the message
src_ = open(os.path.join(HERE, "regress.py"), encoding="utf-8").read()
check("a data change is reported as a stale DRAW", "DRAW" in src_, True)
check("a prompt change is reported as a stale BASELINE", "re-take the baseline" in src_, True)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("all passed")
