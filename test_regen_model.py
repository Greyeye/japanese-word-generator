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
check("config differs by prompt version",
      R.config_key("m", "regen-v1") == R.config_key("m", "regen-v2"),
      False)
check("config is stable for the same config",
      R.config_key("m", "regen-v1") == R.config_key("m", "regen-v1"),
      True)
check("resume key includes the config", 'r["grammar_point_id"], cfg)' in src, True)
check("rows record the config", '"config": cfg' in src, True)
check("old rows without a config still load",
      'd.get("config") or config_key(MODEL)' in src, True)

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

if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("all passed")
