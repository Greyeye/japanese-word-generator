# JLPT sentence corpus — generation pipeline

Generates and validates Japanese example sentences for N5–N3 grammar points, and
loads them into Supabase as a **matching target**: a learner builds a phrase, and
the corpus decides what it is.

The corpus exists to *validate learner-authored sentences*, not to be read. That
constrains everything below.

---

## Layout

```
gen/                        the repo — code only
  *.py                      the pipeline (see Stages)
  data/                     curated inputs the code READS
    lexicon.json            775 words, POS-tagged, from japanese-language-data
    targets.json            77 N5 grammar points + per-point rules
    noun_classes.json       semantic classes (kind/shape) per noun
    noun_class_overrides.json   hand corrections layered over the above
  supabase/
    config.toml
    migrations/
      20261017000000_jlpt_corpus_schema.sql
  .gitignore

H:\jlpt\runs\               everything GENERATED (outside the repo)
  attempts*.jsonl           raw model output, immutable
  *_sentences.jsonl         validated / rejected candidates
  ledger*.json              word-rotation state
  load/                     generated INSERT batches (~12 MB)
  *.log, *_val.txt
```

Path constants in every script:

```python
OUT  = r"H:\jlpt\gen"        # this repo: the code
DATA = os.path.join(OUT, "data")   # curated inputs
RUNS = r"H:\jlpt\runs"       # generated artefacts
```

Nothing generated is written into the repo, and `.gitignore` catches strays.

---

## Stages

| stage | script | what it does |
|---|---|---|
| 0 | `targets.py` | build `targets.json` from the cloned corpus repo |
| 0b | `classify_nouns.py` | model-assisted noun typing → `noun_classes.json` (resumable, `BATCH=25`) |
| 1 | `generate2.py` | guarded generation: POS-routed word pool + ledger rotation + retry |
| 2 | `validate.py` | structural checks only (identity, length, dedup, predicate) |
| 3 | `judge.py` | retrieval-grounded model judge → `valid` / `invalid` / `no_verdict` |
| 4 | `kana_audit.py` → `repair_kana.py` | curated kana-spelling audit and repair |
| 5 | `loader.py` → `push.py` | emit INSERT batches to `runs/load/`, then load |

Supporting: `guardrails.py` (pools, ledger, prompts — imported by `generate2`),
`ab_retry.py` (measured the retry-context fix), `compare_judge.py` (judge agreement),
`semtypes.py`, `demo_match.py`.

### Generate

```powershell
python generate2.py                          # all 77 points
python generate2.py --points 10              # first 10 (pilot)
python generate2.py --only counter-ji        # id substring
python generate2.py --onlylist pts.json      # explicit list
python generate2.py --out $RUNS\attempts.jsonl
```

Remote API instead of the local server (note: the base is `.../api`, because the
script appends `/v1/chat/completions` — passing `.../api/v1` yields a double `/v1`
and a 404):

```powershell
python generate2.py --base https://openrouter.ai/api --key $env:OR_KEY --model openai/gpt-6-luna
```

`--fixwords FILE` freezes the required-word list per point so two runs with
different models see **identical** words. Without it `Ledger.draw()` shuffles and
the second run draws different vocabulary, confounding any model comparison.

---

## Things that bit, and the rules they produced

These are measured findings, not preferences. Each cost a debugging cycle.

**The guardrail is what guarantees coverage, not the prompt.** The prompt raises the
hit rate; the check-and-retry is what enforces it. Coverage is 100% only because
misses are detected and regenerated.

**Retry must be told what already happened.** Without restating the prior output,
43% of generated sentences were duplicates. With it, 0%.

**A word pool is only as good as its routing.** `pos_need()` picks a POS from the
pattern text and `filter_pool()` narrows by semantic class. When either is wrong the
model produces *confident nonsense*, because it faithfully uses the wrong words:

```
counter-ji-oclock  given 切手 箸 ペン  →  「切手を買うのは三時です」  (buying stamps is 3 o'clock)
kara-cause         given bare verbs    →  「忘れるから、書きます」     (because I'll forget, I write)
```

Two traps: `CLASS_RULES` used to admit words via a **union** of axes, so
`shape:{ROUND}` alone let 人/二人 (PERSON) into a countable-things point — hence the
`not_text` / `not_kind` exclusion axes. And a duplicate dict key silently overrides
an earlier one; `ikutsu-how-many` was defined twice.

Sometimes the fix is not the pool but the sentence *shape* — that goes in the
target's `frame` field, which `build_user()` appends to the user prompt.

**Verb-first POS routing wins.** Patterns list alternatives; the first POS mentioned
is the primary frame. Picking another hands the model a pool it cannot use.

**Demote a check rather than fight it.** `sentence_initial_demonstrative` was
rejecting the two grammar points whose subject *is* demonstratives. Demoting it to a
note moved the yield from 73.4% to 76%.

**Judgement checks were net-harmful.** Ten hand-tuned rule checks took the effective
yield from 89% down to 66% and were removed. `validate.py` is now identity and
structure only; judgement belongs in `judge.py`.

**Turn thinking OFF, and check it actually is.** OpenRouter's default is thinking ON,
which burns the whole `max_tokens` budget on reasoning and returns an **empty
`content` field** — 8,158 reasoning tokens and nothing usable. Set
`reasoning: {"enabled": false}` explicitly. `gpt-6.1-sol` refuses to disable it at
all and 400s.

**Don't judge a model by how compliant it feels.** Judge it by which of its failures
are undetectable. Thinking mode bought constraint-following (34 sentence-initial
彼/彼女 → 0) but at 2.9× the cost and *more* templating (33/50 repeated openings), and
its errors were the kind only a human catches.

**Self-correction does not work; external retry does.** Two models diagnosed their
own repetition correctly in their reasoning and failed to fix it in the output. Only
restating the failure in the *next* prompt changed a measured number.

**Dedupe by exact string, not by vector.** Duplicates here are character-identical.
Embeddings are tier 4, not tier 1 — they are insensitive to particles, so
「食べます」and「食べません」sit almost on top of each other with opposite meanings.

**Pick a judge by recall, and never rely on one judge.** Measured against hand
labels: `gpt-6-luna` recall 0.92 / precision 0.77, local gemma 0.65 / 0.85,
`claude-haiku-5.5` 0.77 / 0.65. Two independent judges agree on only ~77% of
sentences, so a single judge is wrong roughly a quarter of the time — use consensus
of ≥2 and quarantine the disagreement.

**Not every flag is a defect.** Of 439 flags, 33% were "pattern not used" (valid
Japanese, wrong grammar tag — re-labelable, not garbage), 9% had a reason that
conceded the sentence was correct, and 3% were register complaints. Classify the
reason before counting it.

---

## Supabase

Schema lives at `supabase/migrations/<version>_<name>.sql` — the layout the Supabase
GitHub integration reads. The project ref is `wilmybwwljzbozotopbhx`.

To connect this repo:

1. `git init` and push to GitHub.
2. In the Supabase dashboard, **Project Settings → Integrations → GitHub** and point
   it at the repo. It reads `supabase/migrations/` on push to the configured branch.
3. For CLI work, `supabase link --project-ref wilmybwwljzbozotopbhx`, then
   `supabase db push`.

The schema is a matching target: `norm_hash` (exact, normalised), `content_lemmas` +
`form_signature` (same words and grammar, via GIN), and `embedding` (HNSW, tier 4)
back a five-tier `match_sentence()` function. Match keys are materialised at load
time, never computed at query time.

---

## Known state

- **N5 corpus: 1,230 sentences.** Two-judge consensus: 42% clean, 35% rejected by
  both, 23% disputed. Real defect rate ~6–8% strict, ~9–16% loose.
- **`runs/n5_corpus_verdicts.jsonl`** holds per-sentence verdicts from both judges.
- Points still needing a routing rule: `nanji-what-time` (92% defective — the same
  time-word bug fixed in `counter-ji-oclock`), `nai-form` (100%), `da-plain-copula`,
  `particle-e-direction`, `i-adj-adverbial-ku`, `particle-made-until`,
  `particle-de-means`.
- `gpt-6.1-sol` was tested and rejected: 20× the price of `gpt-6-luna` for a 2.5×
  higher real error rate, and it writes N1-literary Japanese when the product targets
  N5–N3.
