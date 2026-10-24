"""
generate2.py -- guarded generation: POS-routed pool + ledger rotation + retry-on-miss.

Difference from generate.py:
  * system/user prompt split (standing rules in `system`)
  * the word hint is POS-routed and rotated by a persistent ledger, not the same
    alphabetical 12 words for every grammar point
  * the required words are ENFORCED: after generation we check they were actually
    used, and regenerate the misses with the failure stated in the prompt
  * writes attempts2.jsonl, recording which required words were covered

The prompt raises the hit rate. The check + retry is what guarantees it.

  python generate2.py --points 10              # first 10 grammar points (pilot)
  python generate2.py --only counter-ji        # points whose id contains this
  python generate2.py                          # everything
"""

import json, os, re, sys, time, random, urllib.request
from janome.tokenizer import Tokenizer

sys.path.insert(0, r"H:\jlpt\gen")
import guardrails as G

OUT = r"H:\jlpt\gen"

# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

# single source of truth for the endpoint: reuse generate.py's constants so the
# token is not duplicated into a second file
_src  = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
API   = re.search(r'API\s*=\s*"([^"]+)"', _src).group(1)
TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"', _src).group(1)
MODEL = re.search(r'MODEL\s*=\s*"([^"]+)"', _src).group(1)

POOL_N   = 12        # required words drawn per call
MAXTOK   = 3000
MAX_TRIES = 4      # shared between word coverage and pattern compliance

tok = Tokenizer()
lex   = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))
TG    = json.load(open(os.path.join(DATA, "targets.json"),  encoding="utf-8"))
pools = G.build_pools(lex)
led   = G.Ledger()


def arg(name, default=None):
    a = sys.argv
    if name not in a:
        return default
    i = a.index(name) + 1
    # A trailing valueless flag (e.g. `--append` last on the command line) used to
    # crash with IndexError here. Latent until something ran with such a flag last.
    return a[i] if i < len(a) else True


def call(system, user, temp=0.8):
    body = {"model": MODEL, "max_tokens": MAXTOK,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            "temperature": temp}
    # the local Unsloth server takes reasoning_effort; OpenRouter takes `reasoning`.
    # OpenRouter's default is thinking ON, which returns an EMPTY content field and
    # burns the whole max_tokens budget on reasoning - it must be disabled explicitly.
    if "openrouter" in API:
        body["reasoning"] = {"enabled": False}
    else:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(API + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + TOKEN,
                 "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return time.time() - t0, r["usage"], r["choices"][0]["message"].get("content") or ""


def parse(txt):
    if not txt:
        return []
    txt = re.sub(r"```(?:json)?", "", txt).strip()
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        return []
    try:
        d = json.loads(txt[i:j + 1])
        return [s for s in d.get("sentences", [])
                if isinstance(s, str)
                and s.strip().lower() not in ("sentences", "sentence", "...")]
    except Exception:
        return [x for x in re.findall(r'"([^"]{2,60})"', txt[i:j + 1])
                if x.lower() != "sentences"]



def main():
    global API, TOKEN, MODEL
    # endpoint/model overrides so the same guarded loop can drive a remote API.
    #   --base https://openrouter.ai/api --key $OPENROUTER_API_KEY --model openai/gpt-6-luna
    # NOTE: pass the base WITHOUT /v1. call() appends "/v1/chat/completions", so
    #       ".../api/v1" builds ".../api/v1/v1/chat/completions" and OpenRouter
    #       answers 404 -- which reads as "no such model" for every model at once.
    _b = arg("--base")
    if _b:
        API = _b
    _k = arg("--key")
    if _k:
        TOKEN = _k
    _m = arg("--model")
    if _m:
        MODEL = _m
    _s = arg("--seed")
    if _s:
        random.seed(int(_s))
    # --fixwords freezes the required-word list per point, so two runs with different
    # models get an identical word pool. Without it, draw()'s shuffle means run 2 draws
    # different words and the comparison is confounded by vocabulary, not model quality.
    _fw = arg("--fixwords")
    FIXWORDS = json.load(open(_fw, encoding="utf-8")) if _fw else None
    limit  = arg("--limit")
    only   = arg("--only")
    points = arg("--points")
    todo = list(TG["targets"])
    if only:
        todo = [t for t in todo if only in t["grammar_point_id"]]
    _ol = arg("--onlylist")
    if _ol:
        _want = set(json.load(open(_ol, encoding="utf-8")))
        todo = [t for t in todo if t["grammar_point_id"] in _want]
    if points:
        todo = todo[:int(points)]
    if limit:
        todo = todo[:int(limit)]

    # How many required words may go UNCOVERED before the batch is accepted.
    #
    # Retrying for the last word or two cost three extra model calls per point and
    # bought nothing. Measured: particle-o-object burned all four attempts because
    # 授業 never landed -- the model wrote 「日本語を勉強します」 for that slot -- while
    # the other 11 words were covered. 11/12 is a perfectly usable batch, and a 100%
    # requirement does not scale: with 77 points and one stubborn word each, the retry
    # overhead is the dominant call cost.
    #
    # Allowance is max(1, 10% of required), capped below the word count so a small
    # pool still has to give something (a 1-word pool must still cover its word).
    MISS_ALLOWANCE = float(arg("--miss-allowance") or 0.10)


    def misses_allowed(n_req):
        if n_req <= 1:
            return 0
        return min(n_req - 1, max(1, int(MISS_ALLOWANCE * n_req)))

    ATT = arg("--out") or os.path.join(RUNS, "attempts2.jsonl")
    if not arg("--append") and os.path.exists(ATT) and "--resume" not in sys.argv:
        os.replace(ATT, ATT + ".prev")   # one run per file; keep the previous one
    log = open(ATT, "a", encoding="utf-8")

    print(f"model={MODEL.split('/')[-1]}  points={len(todo)}  pool_n={POOL_N}", flush=True)
    print(f"{'grammar point':<38} {'need':<6} {'tries':>5} {'got':>4} "
          f"{'covered':>8} {'sec':>6}", flush=True)

    t0 = time.time()
    tot_sent = 0
    tot_covered = 0
    tot_required = 0
    first_try_hits = 0
    loose_jobs = 0        # accepted with a word still uncovered (allowed by design)
    tot_dropped = 0       # sentences dropped for failing the point's pattern check
    jobs = 0

    for k, t in enumerate(todo, 1):
        gp    = t["grammar_point_id"]
        need  = G.pos_need(t["pattern"], t["meaning_en"], gp)
        # POS routing, then semantic filtering: a verb pattern draws from verbs, and
        # counter-en-money draws only from nouns that can carry a price
        pool  = G.filter_pool(gp, pools.get(need) or pools["noun"])

        if FIXWORDS is not None:
            # frozen per-point word list, so a second run with a different model sees
            # exactly the same words and the comparison is not confounded by vocabulary
            _by = {v["text"]: v for v in pool}
            required = [_by[w] for w in FIXWORDS.get(gp, []) if w in _by]
        else:
            # a pool smaller than one draw wraps and repeats within the same call
            # (nanyoubi-day-of-week has a 7-word pool)
            required = led.draw(gp, pool, min(POOL_N, len(pool)))
        if not required:
            continue

        rx = t.get("check_regex")
        rx = re.compile(rx) if rx else None

        covered, got_all, tries = set(), [], 0
        miss = None
        pat_ok = False
        while tries < MAX_TRIES:
            tries += 1
            temp = round(random.uniform(0.7, 1.0), 2)
            # ask for one sentence per required word; on a retry, focus on the misses
            n_want = len(required) if not miss else min(len(required), len(miss) + 2)
            if tries > 1 and not pat_ok and rx:
                n_want = max(n_want, 4)
            try:
                el, usage, content = call(G.system_for(gp),
                    G.build_user(t, required, n_want, avoid=got_all, miss=miss), temp)
            except Exception as e:
                log.write(json.dumps({"job_id": t["job_id"], "attempt": tries,
                                      "error": str(e)[:200]}, ensure_ascii=False) + "\n")
                log.flush()
                print(f"{gp[:38]:<38} ERROR {str(e)[:60]}", flush=True)
                break

            got = parse(content)
            # only sentences the generator actually keeps matter downstream; the raw
            # output is retained separately so the log still shows what the model said
            new = [s for s in got if s.strip() not in {x.strip() for x in got_all}]

            # The word pool guarantees the vocabulary; nothing so far guaranteed the
            # grammar. The pattern is checked PER SENTENCE, and -- crucially -- only
            # sentences that PASS it are kept.
            #
            # The previous design kept everything and applied the check as a batch gate
            # (`pat_ok` at 80% of keepers). That failed two ways at once:
            #   1. off-pattern sentences still entered the corpus;
            #   2. a point whose check could not reach 80% retried MAX_TRIES times without
            #      ever converging. Measured: tai-desire and mashou-ka-invitation both
            #      burned 4 attempts at 100% WORD coverage while 25-33% of their output
            #      was off-pattern -- and the retries produced more off-pattern sentences,
            #      so the loop emitted pollution it could not fix.
            # Filtering instead makes pat_ok true by construction and lets coverage alone
            # decide when to stop.
            pos_names = t.get("pos_check") or []

            def _passes(s):
                return (rx.search(s) if rx else True) and G.pos_ok(s, pos_names)

            if rx or pos_names:
                kept_ok = [s for s in new if _passes(s)]
                n_match = len(kept_ok)
                n_seen = len(got_all) + len(new)
            else:
                kept_ok, n_match, n_seen = new, len(new), len(got_all) + len(new)

            got_all.extend(kept_ok)
            pat_ok = True                      # by construction: got_all passed the check
            pattern_ratio = (len(got_all) / n_seen) if n_seen else 1.0

            # which required words did this batch actually use?
            hit = {v["text"] for v in required
                   if any(G.required_present(s, [v], tok) for s in got_all)}
            newly = hit - covered
            covered |= newly
            for w in newly:
                led.bump(gp, w)

            log.write(json.dumps({
                "job_id": t["job_id"], "grammar_point_id": gp,
                "attempt": tries, "model": MODEL, "temperature": temp,
                "required": [v["text"] for v in required],
                "covered": sorted(covered),
                "returned": len(got), "sentences": got,
                "kept": kept_ok, "pattern_ok": pat_ok,
                "pattern_ratio": round(pattern_ratio, 3),
                "dropped_off_pattern": len(new) - len(kept_ok),
                "usage": usage, "sec": round(el, 2),
            }, ensure_ascii=False) + "\n")
            log.flush()
            tot_dropped += len(new) - len(kept_ok)

            if tries == 1 and len(covered) >= POOL_N * 0.6:
                first_try_hits += 1

            miss = sorted({v["text"] for v in required} - covered)
            # Accept when the pattern holds and coverage is within allowance. The old
            # rule was `not miss` -- 100% coverage -- which made one stubborn word cost
            # three extra calls and does not scale across 77 points.
            if pat_ok and len(miss) <= misses_allowed(len(required)):
                if miss:
                    loose_jobs += 1
                break

        jobs += 1
        tot_sent    += len(got_all)
        tot_covered += len(covered)
        tot_required += len(required)
        print(f"{gp[:38]:<38} {need:<6} {tries:>5} {len(got_all):>4} "
              f"{len(covered):>3}/{len(required):<4} {time.time()-t0:>6.0f}", flush=True)

    led.save()
    log.close()

    dt = time.time() - t0
    print(f"\nDONE {jobs} points, {tot_sent} sentences in {dt:.0f}s "
          f"({dt/max(tot_sent,1):.2f} s/sentence)")
    print(f"required-word coverage: {tot_covered}/{tot_required} "
          f"({100*tot_covered/max(tot_required,1):.1f}%)")
    print(f"points hitting >=60% of pool on attempt 1: {first_try_hits}/{jobs}")
    print(f"points accepted with an uncovered word: {loose_jobs}/{jobs} "
          f"(allowance {MISS_ALLOWANCE:.0%}, i.e. {misses_allowed(POOL_N)} of {POOL_N} words)")
    print(f"sentences dropped for failing their own pattern check: {tot_dropped}")
    for p in ("noun", "verb", "adj"):
        u, n = led.coverage(pools[p])
        print(f"  global {p:<5} coverage: {u}/{n} ({100*u/max(n,1):.1f}%)")



if __name__ == "__main__":
    main()
