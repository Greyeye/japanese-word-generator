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
      20261017040810_jlpt_corpus_initial_schema.sql
      20261017042247_jlpt_embedding_dim_1024.sql
      20261018000000_jlpt_match_embedding_dim.sql
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
| 3 | `judge.py` | binary model judge → `valid` / `invalid` / `no_verdict` (superseded) |
| 3b | `rubric_judge.py` | **multi-dimensional** judge: 6 verdicts + target span, codes derived |
| 4 | `kana_audit.py` → `repair_kana.py` | curated kana-spelling audit and repair |
| 5 | `loader.py` → `push.py` | emit INSERT batches to `runs/load/`, then load |

Supporting: `guardrails.py` (pools, ledger, prompts — imported by `generate2`),
`ab_retry.py` (measured the retry-context fix), `compare_judge.py` (judge agreement),
`semtypes.py`, `demo_match.py`.

### Judge

```powershell
python rubric_judge.py --in sentences.jsonl --out verdicts.jsonl `
  --model openai/gpt-6-luna --batch 6
```

Six dimensions, each `pass` / `fail` / `uncertain`:

```
grammar_correct  meaning_coherent  target_used  natural  level_ok  register_ok
```

`reason_codes`, `primary_defect` and `overall` are **derived in code from the
dimensions**, never supplied by the model. That is deliberate: the binary judge
produced 40 flags whose own written reason conceded the sentence was correct
(「誤りとは言えない」/「問題はない」) while marking it invalid. A summary field the
model can contradict is a summary field it will contradict. Across 2,554 verdicts
this design produced **0** `fail`-without-a-code and **0** `pass`-with-a-code.

Every verdict also carries the **target span**, which is verified as a literal
substring of the sentence — a checkable artefact instead of another opinion.


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

## What the rubric judge measured

2,554 verdicts over two models (`gpt-6-luna`, `claude-sonnet-5.5`) on 426 sentences
both binary judges had already rejected:

| dimension | luna | sonnet |
|---|---|---|
| **target_used** | **59.9%** | **55.4%** |
| natural | 28.4% | 34.9% |
| meaning_coherent | 15.5% | 22.4% |
| level_ok | 3.8% | **17.2%** |
| register_ok | 14.6% | 9.0% |
| grammar_correct | 7.3% | 5.0% |

**The dominant defect is that the sentence does not use its target grammar — 55–60%.
Broken Japanese is 5–7%.** The corpus's problem is pattern compliance, not Japanese
quality. A binary verdict reported all of that as a single `ok: false`.

`level_ok` is the **least reliable** dimension: 3.8% vs 17.2% is a 4.5× spread,
by far the widest. Neither model holds a trustworthy N5 kanji/vocabulary list, and
sonnet flagged 切符 (符) and 風呂 (風) as above N5. Inject an explicit list or treat
`LEVEL` as non-blocking — filtering `LEVEL`/`REGISTER` out of `reason_codes` moves
arm A from 7.7% to 3.5%.

Against an independent single-label classification the derived codes agree on
50–54% exactly, but recover `WRONGPAT` at **92–93%** — reliable where it matters
most, divergent only in the tail, where the two schemes aren't doing the same task.

### Regeneration beats correction (measured twice)

426 consensus-rejected sentences, two arms with **matched prompts** (same model,
same system prompt, same context; only the instruction differs):

| defect class (n) | A regenerate | B correct |
|---|---|---|
| WRONGPAT — right Japanese, wrong pattern (197) | 11.2% | 22.8% |
| NUANCE — pattern used, wrong nuance (128) | 9.4% | 15.6% |
| BROKEN — Japanese itself wrong (84) | 6.0% | 17.9% |
| **near-miss pool (336)** | **11.6%** | **19.6%** |
| **broken pool (84)** | **6.0%** | **17.9%** |

Paired McNemar p=2.2e-4 and p=4.8e-9. `gpt-6-sol` judged independently and agreed.
The rubric judge then reconfirmed it against two more judges.

**Correction is not a weaker version of regeneration — it is worse, everywhere.**
A minimal edit is anchored to the rejected sentence, and the rejection was usually
about the content *frame*, so the edit preserves the defect:
「魚はとても大きかったです」→「魚は今日のお昼でした」 (nonsense), where regeneration
produced 「昼ご飯はカレーでした」.

Arm B's signature failure is **NUANCE** (4.9–9.2% vs A's 0.9–1.6%): the pattern is
present but used with the wrong nuance — patched surface, unchanged frame. Arm A's
signature weakness is **LEVEL** (4.9%): valid Japanese above N5 (`十分` for ten
minutes is ambiguous じゅっぷん/じゅうぶん; `絵を書く` should be 描く).

Note the constraint fear was **unfounded**: arm B preserved the pattern (97.7% vs
98.1% for A) and length (100% both) — the damage was semantic, invisible to
mechanical checks, and showed up as vocabulary like 一部 / 見つける / 答え.

### Judge noise floor

On the 44 sentences where both arms emitted **byte-identical** text, batched
verdicts still disagreed 2–4% of the time. Batching makes a verdict depend on batch
composition, and the model notices within-batch duplication. Effect sizes here are
8–13pp, well above it, but any comparison should be paired and per-item, not
aggregate. `--batch 1` for calibration.

### Reasoning cannot be disabled everywhere

`claude-sonnet-5.5` and `gpt-6.1-sol` both 400 on `reasoning: {"enabled": false}`
with *"Reasoning is mandatory for this endpoint and cannot be disabled."* Use
`reasoning: {"exclude": true}` and give the call an 8k budget. This matters for
cost: luna judges 2,554 verdicts for **$0.06**, sonnet for **$3.30** — ~45×, because
sonnet's thinking is billed as output. At 100k sentences that is ~$7 versus ~$390.

---

## Supabase

Project ref **`wilmybwljzbozotopbhx`**. Schema lives at
`supabase/migrations/<version>_<name>.sql` — the layout the GitHub integration
reads, and the layout `supabase db push` reads.

To apply the repo to the database (no linking needed if `SUPABASE_DB_URL` is set;
the CLI records `version` from the filename, so the repo filenames become the
project's migration versions):

```powershell
cd H:\jlpt\gen
npx supabase db push --db-url $env:SUPABASE_DB_URL
```

To wire up the GitHub integration: push this repo, then **Project Settings →
Integrations → GitHub** in the dashboard and install the app for this repository.
It reads `supabase/migrations/` on push to `main`. That step is dashboard-only —
the MCP server exposes 29 Supabase tools but none manage the GitHub app install.

### Migrations

| version | name | what |
|---|---|---|
| 20261017040810 | jlpt_corpus_initial_schema | 5 tables + indexes + `match_sentence()`; embedding starts at 768 |
| 20261017042247 | jlpt_embedding_dim_1024 | widens `embedding` to 1024 (Qwen3-Embedding-0.6B) |
| 20261018000000 | jlpt_match_embedding_dim | declares `match_sentence`'s embedding arg unconstrained |

Version 3 exists because version 2 widened the *column* but left the *function
argument* at 768 — a database rebuilt from 1+2 would have rejected the embeddings
its own column stores. Never edit an applied migration; add one.

The schema is a matching target: `norm_hash` (exact, normalised),
`content_lemmas` + `form_signature` (same words and grammar, via GIN), and
`embedding` (HNSW, tier 4) back the five-tier `match_sentence()`. Match keys are
materialised at load time, never computed at query time.

## Known state

- **The database is empty by design.** Rebuilt from these migrations, so
  `schema_migrations` holds exactly the three rows above and the five tables have 0
  rows. Nothing has been loaded since the rebuild.
- **N5 corpus quality, two ways of measuring.** Binary two-judge consensus over the
  1,230-sentence corpus: 42% clean, 35% rejected by both, 23% disputed; real defect
  ~6–8% strict, ~9–16% loose. The rubric judge over 426 of those rejects says the
  dominant defect is `target_used` at 55–60%. The corpus predates several routing
  fixes, so its rates are inflated relative to a fresh run.
- **Of the 426 consensus-rejected, 232 are valid Japanese tagged to a grammar point
  they do not exercise** — a re-labelling job, not a discard (520 → ~750 usable).
- **Points known to need a routing rule:** `nanji-what-time` (~92% — the same
  time-word bug fixed in `counter-ji-oclock`; needs both `CLASS_RULES` and a
  `targets.json` `frame`), `nai-form` (100%), `da-plain-copula` (80%),
  `particle-e-direction` (75%), `i-adj-adverbial-ku` (74%), `particle-made-until`
  (73%), `particle-de-means` (71%). The remaining ~70 points are unranked.
- **Fixed and verified:** `counter-ji-oclock`, `kara-cause`, `jikan-time-duration`,
  `location-nouns-ue-shita`, `na-adjective-negative`, `ikutsu-how-many`.
- **`gpt-6.1-sol` rejected:** 20× the price of `gpt-6-luna` for a 2.5× higher real
  error rate, and it writes N1-literary Japanese when the product targets N5–N3.
- **ChatGPT/Codex subscriptions do not help here.** Plus includes Codex, but Codex
  is an agent allowance, not API access — on Plus the GPT-6 Luna lane is 350–3,000
  *local messages* per 5h with an unpublished weekly cap, so 200k verdicts would need
  ~2.3 days of continuous best-case allowance with none left for real work. The whole
  100k corpus costs ~$15 at API rates.
- **Untested (deliberately):** `loader.py` and `push.py` against the rebuilt schema.
  They compile and their paths resolve, but nothing has been loaded.
