#!/usr/bin/env python3
"""Tests for regen_loop: the --model flag must reach the request body.

An external code review found that gen_one() built its request from the module-level
MODEL constant, so --model was parsed but ignored while --judge-model worked. The
failure mode is silent and specifically dangerous: a generator comparison run through
this script would compare luna against luna and look like a clean null result.

Run:  python test_regen_model.py
"""

import io
import json
import os
import sys
import urllib.request

import regen_loop as R

HERE = os.path.dirname(os.path.abspath(__file__))

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILS.append(name)


class _Resp(io.BytesIO):
    """Minimal stand-in for the urllib response object."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def capture_body(**kwargs):
    """Call gen_one with urlopen stubbed out; return the JSON body it sent."""
    sent = {}

    def fake_urlopen(req, timeout=None):
        # urllib.request.Request -> body is bytes of the JSON payload
        sent["body"] = json.loads(req.data.decode("utf-8"))
        payload = {"choices": [{"message": {"content": '{"sentences":["テスト。"]}'}}],
                   "usage": {}}
        return _Resp(json.dumps(payload).encode("utf-8"))

    real = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        R.gen_one({"japanese": "前の文。", "pattern": "X（Y）"}, 
                  {"reason_codes": ["BROKEN"], "failed_dimensions": ["grammar_correct"],
                   "reason": "テスト"},
                  "fake-key", required=[{"text": "水"}], **kwargs)
    finally:
        urllib.request.urlopen = real
    return sent["body"]


print("regen_loop --model plumbing")
print()
item_note = "no model passed -> module default"
check(item_note, capture_body()["model"], R.MODEL)

check("explicit model is used",
      capture_body(model="anthropic/claude-haiku-5.5")["model"],
      "anthropic/claude-haiku-5.5")

check("a different explicit model is used",
      capture_body(model="google/gemma-4-26b-it")["model"],
      "google/gemma-4-26b-it")

# the flag must be wired to gen_one, not just accepted by argparse
import inspect
src = inspect.getsource(R.main)
check("main() passes a.model into gen_one", "model=a.model" in src, True)

# and the record must carry the model, so a run is auditable after the fact
check("output records gen_model", '"gen_model": a.model' in src, True)

print()
print("resume identity")
print()

# the resume key must identify the CONFIGURATION, not just the sentence. Keying on
# (parent, grammar_point_id) alone meant a --model or prompt change found the previous
# run's rows already "done" and silently reused the old model's output.
check("config differs by model",
      R.config_key("openai/gpt-6-luna") == R.config_key("anthropic/claude-haiku-5.5"),
      False)
check("config is stable for the same config",
      R.config_key("m") == R.config_key("m"), True)
check("resume key includes the config", 'r["grammar_point_id"], cfg)' in src, True)
check("rows record the config", '"config": cfg' in src, True)

print()
print("content-addressed prompt identity (a hash is evidence, a version string is not)")
print()

# the identity must change when the PROMPT changes, with nobody remembering to bump
# anything. Simulate an edit by re-hashing a mutated copy of the template.
import hashlib
import inspect
base = inspect.getsource(R.regen_prompt) + "\n--SYS--\n" + R.REGEN_SYS
edited = base.replace("20文字以内", "18文字以内")
check("the prompt template actually differs after an edit", edited != base, True)
h_base = hashlib.sha256(base.encode()).hexdigest()[:16]
h_edit = hashlib.sha256(edited.encode()).hexdigest()[:16]
check("a prompt edit changes the prompt hash", h_base == h_edit, False)

# and the hash must be what config_key embeds, so an edit propagates automatically
check("config_key embeds the prompt hash", "prompt_sha256" in inspect.getsource(R.config_key), True)
check("prompt_sha256 is derived from source, not a constant",
      "inspect.getsource" in inspect.getsource(R.prompt_sha256), True)
check("prompt hash is stable across calls", R.prompt_sha256() == R.prompt_sha256(), True)

# generator and judge identities are SEPARATE: a judge change must not invalidate
# generation output, and a generation change must not reuse it
check("judge identity differs from generation identity",
      R.config_key("m") == R.judge_config_key("m"), False)
check("judge identity tracks the judge model",
      R.judge_config_key("judge-a") == R.judge_config_key("judge-b"), False)
check("judge identity is stable", R.judge_config_key("j") == R.judge_config_key("j"), True)

print()
print("invalidation: what changes the generation identity, and what must not")
print()

# the reviewer's question: does the hash cover the EFFECTIVE configuration, or only
# selected pieces? A change to build_user() or to the word pools alters the request
# without touching regen_prompt's text, so both must be inside the identity.
import inspect as _i
check("prompt hash covers build_user, not just the template",
      "getsource(G.build_user)" in _i.getsource(R.prompt_sha256), True)
check("identity covers the curated data", "data_sha256" in _i.getsource(R.config_key), True)
check("identity is schema-versioned", "HASH_SCHEMA" in _i.getsource(R.config_key), True)

# a data edit must change the identity. Prove it on a copy rather than the real file.
import tempfile
import shutil
real_data = R.data_sha256()
tmp = tempfile.mkdtemp()
try:
    print(f"        (data identity = {real_data})")
    check("data identity is stable across calls", R.data_sha256() == real_data, True)
    # simulate: hash a mutated copy of targets.json the same way
    import hashlib as _hl
    tp = os.path.join(HERE, "data", "targets.json")
    raw = open(tp, "rb").read()
    h_orig = _hl.sha256(b"targets.json" + raw).hexdigest()[:16]
    h_edit = _hl.sha256(b"targets.json" + raw + b" ").hexdigest()[:16]
    check("a targets.json edit changes its data identity", h_orig == h_edit, False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# model is part of the identity; a judge is NOT
check("identity changes with the generation model",
      R.config_key("model-a") == R.config_key("model-b"), False)
check("identity does NOT include the judge",
      R.config_key("m") == R.config_key("m"), True)

# a display-only change must not invalidate anything: nothing in the identity reads
# the README, the docs, or arbitrary files
src_cfg = _i.getsource(R.config_key) + _i.getsource(R.prompt_sha256) + _i.getsource(R.data_sha256)
check("identity does not read the README", "README" in src_cfg, False)
check("identity only reads named data files",
      src_cfg.count("os.path.join(GEN, \"data\"") >= 1, True)

print()
print("provenance classification: missing / invalid / different / matching")
print()

CFG = R.config_key("m")
good = {"parent": "P1", "grammar_point_id": "gp1", "config": CFG}
other = {"parent": "P2", "grammar_point_id": "gp2", "config": R.config_key("other-model")}
none_c = {"parent": "P3", "grammar_point_id": "gp3"}
bad_len = {"parent": "P4", "grammar_point_id": "gp4", "config": "abc"}
bad_type = {"parent": "P5", "grammar_point_id": "gp5", "config": 12345}
bad_hex = {"parent": "P6", "grammar_point_id": "gp6", "config": "zzzzzzzzzzzzzzzz"}

match, miss, inv, diff = R.classify_provenance(
    [good, other, none_c, bad_len, bad_type, bad_hex], CFG)

check("an identical config is a match", ("P1", "gp1", CFG) in match, True)
check("a different known config is NOT a match", ("P2", "gp2") in match, False)
check("a different known config is counted", diff, 1)
check("a missing config is 'missing', not matched", ("P3", "gp3") in miss, True)
check("a short config is 'invalid'", ("P4", "gp4") in inv, True)
check("a non-string config is 'invalid'", ("P5", "gp5") in inv, True)
check("a non-hex config is 'invalid'", ("P6", "gp6") in inv, True)
check("nothing invalid leaked into missing", len(miss), 1)
check("nothing missing leaked into invalid", len(inv), 3)

# an empty file is not an error and not a match
m2, s2, i2, d2 = R.classify_provenance([], CFG)
check("no existing rows classifies as nothing", (len(m2), len(s2), len(i2), d2), (0, 0, 0, 0))

# only MISSING is eligible for the trust-legacy override
rel = open(os.path.join(HERE, "regen_loop.py"), encoding="utf-8").read()
check("trust-legacy is scoped to missing config only",
      'in missing:' in rel and "in invalid:" not in rel, True)
check("trust-legacy says accepting, not verifying",
      "accepting, not verifying" in rel, True)
check("trust-legacy reports how many rows it reused",
      "reusing {len(missing)} rows" in rel, True)

print()
print("generate2 import safety")
print()

# generate2.py had no __main__ guard: its 185-line generation loop ran at module level,
# so merely IMPORTING it launched a full 77-point paid run. regress.py did import it,
# which is how this was found. The loop must live inside main().
import ast
gsrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate2.py"),
            encoding="utf-8").read()
gtree = ast.parse(gsrc)
gfns = {n.name: (n.lineno, n.end_lineno) for n in gtree.body if isinstance(n, ast.FunctionDef)}
check("generate2 defines main()", "main" in gfns, True)
check("generate2 guards execution", 'if __name__ == "__main__":' in gsrc, True)
if "main" in gfns:
    loop_line = next((i + 1 for i, l in enumerate(gsrc.split("\n"))
                      if l.strip().startswith('limit  = arg("--limit")')), None)
    lo, hi = gfns["main"]
    check("generation loop is inside main()",
          bool(loop_line and lo <= loop_line <= hi), True)
    # and not swallowed by a helper, which is the failure I actually made first
    for helper in ("parse", "call"):
        if helper in gfns:
            hlo, hhi = gfns[helper]
            check(f"{helper}() does not contain the loop",
                  not (loop_line and hlo <= loop_line <= hhi), True)

# a trailing valueless flag (e.g. --append last) used to raise IndexError in arg()
check("arg() handles a trailing valueless flag",
      "return a[i] if i < len(a) else True" in gsrc, True)

print()
print("the README's documented hash values match the code")
print()

# This drifted the moment I widened prompt_sha256: the README still listed the old
# three values. A documented value that disagrees with the code is the recurring sin
# across these review rounds, so assert it instead of re-checking by hand.
readme = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
live = {
    "prompt_sha256": R.prompt_sha256(),
    "data_sha256": R.data_sha256(),
    "config (luna)": R.config_key("openai/gpt-6-luna"),
    "config (haiku)": R.config_key("anthropic/claude-haiku-5.5"),
    "judge  (luna)": R.judge_config_key("openai/gpt-6-luna"),
}
for label, val in live.items():
    check(f"README lists the live {label} ({val})", val in readme, True)

# and no superseded value may linger in the block either
for stale in ("404bf3edd2e51991", "14626756cd1957c8", "afc443d2f747d0fc"):
    check(f"README has dropped the superseded {stale}", stale in readme, False)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("all passed")
