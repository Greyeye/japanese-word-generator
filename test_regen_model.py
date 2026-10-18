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
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("all passed")
