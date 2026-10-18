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
| 4b | `regen_loop.py` | judge `fail` -> regenerate with the failure stated (not correction) |
| 4 | `kana_audit.py` → `repair_kana.py` | curated kana-spelling audit and repair |
| 5 | `loader.py` → `push.py` | emit INSERT batches to `runs/load/`, then load |

### Regeneration loop

```powershell
python regen_loop.py --in sentences.jsonl --out regen.jsonl `
  --verdicts verdicts.jsonl --key $env:OR_KEY
```

Judges the input, then regenerates everything that failed on a structural or
linguistic dimension — stating which dimensions failed, the reason codes and the
judge's own words. It does **not** correct; correction lost to regeneration in
every stratum of the A/B (see below).

Two policies worth knowing:

- **`LEVEL`/`REGISTER` are not regenerated.** They are valid Japanese sitting above
  (or beside) the target; the right response is a re-label, not a re-roll. On 1,012
  sentences this declined 34 of 162 candidates.
- It draws a required word per sentence from the same pool + ledger as
  `generate2.py`. Without that the loop collapses to templates: **33% of its output
  was a duplicate of another candidate** (86 distinct among 128). With it, 128/128
  distinct — but also more failures, because a word that cannot take the pattern is
  now forced into it (「箸を二つください」 is on the pool, not the model).

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

- **The database is empty by design.** Rebuilt from these migrations; nothing has
  been loaded since (the load is deliberately untested — `loader.py`/`push.py`
  compile and resolve their paths but have not been run).
- **Fresh N5 sample, current guardrails, unbiased:** 1,012 sentences over all 77
  points. `target_used` fails on **8.0%**, overall-fail **16.0%** (gpt-6-luna,
  rubric judge). The old corpus's 55–60% figure was measured *conditional on
  already being rejected*, so it is not comparable.
- **Six points fixed and re-measured, under BOTH judges** (same prompts on both
  samples, so the comparison isolates the fixes; `target_used` fail rate):

  | point | luna before | luna after | haiku before | haiku after |
  |---|---|---|---|---|
  | location-nouns-ue-shita | 56% | 0% | 50% | 14% |
  | na-adj-adverbial-ni | 58% | 9% | 58% | 9% |
  | masenka-invitation | 58% | 8% | 75% | 17% |
  | deshita-past-polite-copula | 42% | 0% | 50% | 0% |
  | particle-o-object | 48% | 9% | 43% | 18% |
  | kono-sono-ano-dono-attributive | 58% | 17% | 58% | 8% |

  All six improve substantially under both judges, so they are not artefacts of one
  judge. Corpus-wide `target_used` 11.4% -> 8.0% (luna) and 13.7% -> 11.5% (haiku);
  agreement 81.6% -> 82.9%. Caveat: n≈12 per point, so one sentence is 8.3pp;
  deltas under ~17pp are noise, and the two runs drew different vocabulary.
- **Next worklist — CONSENSUS of both judges (luna and haiku, current prompt), 13
  points where they agree the pattern is missed on >=15% of sentences:**

  | point | luna | haiku | n |
  |---|---|---|---|
  | na-adj-te-joining-de | 17% | 58% | 12 |
  | question-words-basic | 36% | 36% | 36 |
  | particle-de-means | 25% | 42% | 12 |
  | da-plain-copula | 17% | 50% | 12 |
  | na-adjective-attributive | 25% | 33% | 12 |
  | na-adjective-past | 33% | 20% | 15 |
  | masen-polite-negative-verb | 25% | 25% | 12 |
  | particle-to-and-with | 17% | 33% | 12 |
  | te-kudasai-request | 17% | 33% | 12 |
  | mada-not-yet | 17% | 25% | 12 |
  | mashou-volitional | 17% | 25% | 12 |
  | counter-tsu | 18% | 18% | 28 |
  | particle-kara-from | 17% | 17% | 12 |

  `counter-tsu` independently corroborated by the regeneration loop, which found
  the pool admits words counted with 膳/匹/本 rather than つ (「箸を二つください」).
  `question-words-basic` burns all four retries every run.
- **Points the judges irreconcilably disagree on — do not trust a single-judge rate:**
  `nanyoubi-day-of-week` (luna 0% / haiku 100%, n=7), `particle-no-possession`
  (67% / 0%), `dewa-arimasen-negative-polite-copula` (25% / 0%). These need more
  sentences, and the weekday point may need human data — judges split 14% on it.
- **Regeneration loop:** pooled over two runs, **233/256 (91%)** of sentences that
  were failing by construction came back passing, for ~$0.03.
- **POS checks (`predicate_is_noun`, `no_suru_setphrase`) are correct but INERT —
  they have never been observed to fire.** Attached to four points
  (`deshita`/`desu`/`da-plain-copula` via `predicate_is_noun`; `particle-o-object`
  via `no_suru_setphrase`), run as generation-time gates feeding the retry.

  Unit-tested: `predicate_is_noun` 20/20 on good/bad pairs including no-copula
  sentences (which must not fire), `no_suru_setphrase` 8/8. Both separate cases a
  regex cannot, because the tokens differ only in **subcategory**:

  ```
  「今日は木曜日でした。」 木曜日[名詞/副詞可能] でし      -> noun predicate, OK
  「村は静かでした。」   静か[名詞/形容動詞語幹] でし    -> not Noun + でした
  「たばこを吸います。」  たばこ[名詞/一般] + を + 吸う    -> real object, OK
  「買い物をします。」    買い物[名詞/サ変接続] + を + する -> verbal noun, reject
  ```

  But on 54 generated sentences across those four points the checks rejected **0**.
  The model avoids the blocked realisations rather than producing them, so the
  upstream fixes (frame + pool + regex) still do all the work and these are a cheap
  backstop for the case where a `frame` hint is ignored. A `frame` is a prompt; a
  check is a gate. They are not a measured improvement and should not be counted
  as one.

- **A proposed improvement that measurement rejected: trimming the
  `particle-o-object` deny-list.** The argument was that the POS check generalises
  the hand list, so the list could shrink to the words with no non-する realisation
  and 電話 would become drawable again, making 「友だちに電話を掛けます」 reachable.
  Measured: `particle-o-object` got **worse** — luna 9% -> 39%, haiku 18% -> 39% —
  and the pool filter turns out to be the mechanism that actually works, because
  re-admitting 電話/仕事/勉強 makes the model drop required words or reach for
  awkward realisations. Reverted to the full 10-word list; the POS check stays as a
  backstop.
- **Beware draw variance when re-measuring a single point.** `da-plain-copula` moved
  17% -> 0% (luna) and 50% -> 0% (haiku) across the same two runs, while nothing
  about that point changed and its POS check never fired. With ~12 sentences per
  point, one sentence is 8.3pp and a whole-point swing can be nothing at all. Only
  the POS-rejection count (0) is a clean signal; the rates are not.
- **Coverage acceptance is 90%, not 100% (`--miss-allowance`, default 0.10).** The
  old rule accepted a batch only when *every* required word was covered
  (`if not miss`), so one stubborn word cost three extra model calls for the whole
  point — `particle-o-object` burned all four attempts because 授業 never landed
  while the other 11 words were. Measured full-corpus effect:

  | | 100% required | 90% allowance |
  |---|---|---|
  | total attempts | 98 | **84** |
  | extra calls (above 1/point) | 21 | **7** |
  | points retrying | 9 | **3** |
  | final coverage | 99.6% | 99.1% |
  | sentences produced | 1012 | 965 |

  A 100% requirement does not scale: with one stubborn word per point, retry
  overhead becomes the dominant call cost. The allowance is `max(1, 10% of
  required)` capped below the word count, so a 1-word pool must still cover its
  word.

  **Does it cost quality? Not measurably — and the test is worth keeping.** Corpus
  `target_used` read 8.0% -> 10.8% (luna) and 11.5% -> 14.2% (haiku), which looks
  like a cost. Splitting by whether a point's retry behaviour actually changed
  dissolves it:

  | group | luna | haiku |
  |---|---|---|
  | STOPPED retrying (the 7 points the change can act on) | 14.9% -> 16.7% (+1.8) | 14.9% -> 13.1% (**-1.8**) |
  | everything else (68 points, unaffected by the change) | 6.6% -> 10.2% (**+3.6**) | 10.8% -> 14.3% (**+3.5**) |

  The control group moved more than the treatment group under both judges, so the
  corpus movement is **draw variance**, not a cost of the relaxation — and under
  haiku the group the change acts on got better. Never compare two whole-corpus
  rates without a control group.
- **Only sentences that pass their own point's pattern check are kept.** The loop
  used to keep everything and apply the check as a *batch* gate (`pat_ok` at 80% of
  keepers). That failed two ways at once: off-pattern sentences entered the corpus,
  and a point whose check could not reach 80% retried `MAX_TRIES` times without ever
  converging — `tai-desire` and `mashou-ka-invitation` both burned 4 attempts at
  **100% word coverage** while 25–33% of their output was off-pattern. Filtering
  makes `pat_ok` true by construction, lets coverage alone decide when to stop, and
  gives a hard guarantee: **0 kept sentences fail their own pattern check** (67+
  before).

  Measured effect, both judges, three generations of the same code path:

  | generation | luna `target_used` | haiku `target_used` |
  |---|---|---|
  | 100% coverage required | 8.0% (1012) | 11.5% (1012) |
  | 90% allowance | 10.8% (965) | 14.2% (965) |
  | **90% allowance + per-sentence filter** | **5.9%** (928) | **7.2%** (928) |

  Filter vs allowance alone: **-4.9pp** (luna) and **-7.0pp** (haiku), with 28 and 32
  of ~77 points improving against 13 and 14 worsening. The mechanism is checkable:
  the points that improve most are exactly the ones with the most off-pattern output
  (`particle-o-object` 50% -> 0%, `mashou-ka-invitation` 35% -> 0%, `tai-desire`
  31% -> 8%).

  **The two changes are needed together.** Filtering alone lowers coverage, because
  dropped sentences no longer count toward it, so the loop retries longer — attempts
  went 84 -> 94. The 90% allowance is what stops that from becoming thrashing. The
  allowance alone looked like a quality regression; with the filter it is the best
  configuration measured.

  Also fixed a regex gap found by this work: `tai-desire`'s `たい` does not match
  `たくない` (the substring is `た` `く` `な` `い`), so «時間が掛かりたくないです» — a
  perfectly good 〜たい example — was rejected. Now `た(い|くない|かった|くなかった)`,
  taking its match rate 67% -> 78%.

  **Coverage reads 97.0% here, not 99.1% — that is a stricter denominator, not lost
  words.** A required word used only in an off-pattern sentence no longer counts
  toward coverage. Like-for-like, counting every generated sentence, the same run is
  **98.9%** against the loose run's 99.1% — a 0.2pp difference, which is the
  allowance accepting one uncovered word on 13 points. Verify with
  `attempts.jsonl`: compare `covered` against required words found in `sentences`
  rather than `kept`.

  Where the 67 dropped sentences came from — `particle-o-object` 18,
  `koko-soko-asoko-doko` 10, `te-form-basic` 10, `ta-form` 9, across 17 points.
  `particle-o-object` dominates because its を covers paths and departures
  (「公園を散歩します」, 「バスを降ります」) as well as objects, so its regex admits
  sentences the judge calls off-pattern. Removing them is why its judged
  `target_used` went 50% -> 0%.
- **`particle-o-object`'s pool is restricted to TRANSITIVE verbs, from the lexicon's
  JMdict transitivity (`pos_raw_any: {vt}`).** を must mark a direct object, but 60 of
  the point's 136 pool verbs were vi-only, so を ended up marking a path or departure
  (「公園を散歩します」/「バスを降ります」/「道を走ります」). One mistake caused both of
  its symptoms — off-pattern sentences the filter then dropped, and the coverage
  collapse that followed. Pool 126 -> 69. Measured on a full run:

  | | before | after |
  |---|---|---|
  | attempts | 4 | **1** |
  | word coverage | 7/12 (58%) | **11/12 (92%)** |
  | on-pattern ratio | ~0.58 | **0.92** |
  | sentences dropped corpus-wide | 67 | **34** |

  The transitivity filter also removed the lexicon's mis-tagged nouns (お茶, 帽子,
  授業, 作文, 暇, マッチ), which were reaching a verb pool.

  **It is a CORRECTNESS fix, not a quality improvement, and the judged rates say so.**
  Both judges show no separation: luna 5.9% -> 5.3% with 19 points improved and 18
  worsened; haiku 7.2% -> 7.8% with 24 improved and 25 worsened. They do not even
  agree on which run is better (luna prefers the vt pool, haiku prefers the filter).
  The target point itself reads 0% -> 18%, which is 2 sentences out of 11. At ~12
  sentences per point and ~80% judge agreement, an effect this size is not resolvable
  — and it should not be claimed as one. What justifies keeping it is the
  deterministic argument: an intransitive verb cannot take a direct object, so a pool
  containing them guarantees off-pattern output regardless of what the model does.

  This is a new rule axis (`pos_raw_any` / `pos_raw_none` in `noun_ok`), reading
  JMdict vt/vi rather than `noun_classes`, so it applies to words the noun classifier
  never saw. Transitivity is the one property a surface regex cannot recover.
- **The consensus worklist shrank from 13 points to 8** after the per-sentence filter.
  Re-derived from the current run, both judges agreeing the pattern is missed on >=15%:

  | point | luna | haiku |
  |---|---|---|
  | particle-de-means | 42% | 25% |
  | counter-fun-minute | 33% | 17% |
  | mai-every-prefix | 33% | 17% |
  | particle-e-direction | 25% | 33% |
  | na-adjective-nonpast | 27% | 18% |
  | na-adjective-attributive | 21% | 21% |
  | na-adj-te-joining-de | 17% | 33% |
  | masenka-invitation | 17% | 25% |

  Still irreconcilable and not to be acted on from one judge: `ichiban-superlative`
  (0% / 50%) and `particle-kara-from` (42% / 8%).
- **Generator precision — all four generations, coverage on a like-for-like standard.**
  The reported coverage figure changed meaning when the per-sentence filter landed (a
  word used only in an off-pattern sentence stopped counting), so the two columns
  differ. The old-standard column counts every generated sentence, as the pre-filter
  runs did:

  | generation | points | attempts | kept | dropped | cov (reported) | cov (old std) |
  |---|---|---|---|---|---|---|
  | 100% coverage required | 77 | 98 | 1021 | – | 99.6% | **99.0%** |
  | 90% allowance | 77 | 84 | 965 | – | 99.1% | **98.8%** |
  | + per-sentence filter | 77 | 94 | 931 | 67 | 97.0% | **98.9%** |
  | + transitive pool | 77 | 92 | 939 | 34 | 97.6% | **98.6%** |

  **Word coverage is flat — 98.6–99.0% — across every change.** The generator was
  never the constraint on coverage; what moved was the attempt count (98 -> 92) and
  how much off-pattern text gets dropped (67 -> 34). Any statement that coverage
  "improved" or "regressed" over this work is a measurement-standard artefact. The
  drop column starts at the filter change because nothing was filtered before it.

- **Generator precision, measured on the after-fixes run (77 points, 1,021 kept
  sentences, 210s, 0.21 s/sentence):** required-word coverage **909/913 = 99.6%**,
  not 100% — four points missed one word each (`kara-cause`, `masu-polite-verb`,
  `mou-ta-already`, `tai-desire`). Nine sentences were exact duplicates of another
  (1,012 unique of 1,021). 75/77 points covered >=60% of their pool on attempt 1.
  Retries cost 30 attempts across 9 points; the other 68 points needed exactly one.
  Coverage was never 100%: the before-fixes run was 917/919 = **99.8%** with 76/77
  on attempt 1. So 99.6% is not a regression from the fixes — the guardrail's
  guarantee is "checked and retried", not "always satisfied".
- **`counter-people-nin`'s check_regex was broken (41% match).** It read
  `(にん|一人|二人|ひとり|ふたり)`, so 三人–十人 — written in kanji, containing no
  kana `にん` — never matched, and the point burned all four retries on every run
  while reporting failure. Fixed to `([一二三四五六七八九十]+人|何人|にん|ひとり|ふたり)`;
  now 100% on the sentences it produced, with no false positives (bare 人 alone does
  not match, so 「人がいます。」 is correctly rejected). The `+` matters: `*` accepted
  bare 人.
- **Three more points never satisfy `pattern_ok` and are model drift, not regex
  bugs** — `question-words-basic` (61% match), `counter-tsu` (82%), `particle-ni-target`
  (88%). The regex is satisfiable; the model simply produces off-pattern sentences
  (「明日は晴れですか」 for a question-word point), burning all four retries. These
  want a `frame`, not a regex change.
- **`level_ok` is now derived in BOTH directions, and the reason it could not be
  before was a bug in the lookup.** The above-N5 signal was unusable for three
  separate reasons, all now fixed:

  1. **The tokeniser's base form was not preferred.** `行きます` yields base `行く`
     (N5) but surface `行き`, which `jlpt_all` grades N3. Falling back to the surface
     flagged every ます-stem verb in the corpus: 行き n=39, 読み n=13, 帰り n=11. This
     was the single largest source of false "above level" hits — 54 distinct flagged
     words, now 18.
  2. **Sub-POS was not filtered.** Suffixes (さん, graded N1) and non-independent
     helpers (あり/くださる) are not vocabulary a learner chooses.
  3. **`jlpt_all` contradicts the curated list on 56 words** (人->N1, 三->N1, 店->N1,
     中->N3, 明日->N4) because it grades a homograph's other sense. `data/n5_extra.json`
     records the 27 corrections needed beyond that, each an inspectable word.

  Measured against hand-verified sets: **0 false positives on 27 N5 words, 0 false
  negatives on 26 above-N5 words**. Corpus flags fell from 7.5% of sentences to 2.5%.
  A tenth case (`友だち`, absent from both lists) correctly defers to the model rather
  than deriving — both directions require the evidence to be complete, and any
  unplaceable word leaves `level_ok` to the model.

  Residual risk is bounded by policy: a false `LEVEL` failure does not trigger
  regeneration (`regen_loop.NO_REGEN`), so the cost is a re-label decision, never a bad
  sentence entering the corpus.
- **Of the old 1,230-sentence corpus, 232 rejects are valid Japanese tagged to a
  grammar point they do not exercise** — a re-labelling job (520 -> ~750 usable).
- **`gpt-6.1-sol` rejected:** 20x the price of `gpt-6-luna` for a 2.5x higher real
  error rate; it writes N1-literary Japanese when the product targets N5–N3.
- **ChatGPT/Codex subscriptions do not help here.** Plus includes Codex, but Codex
  is an agent allowance, not API access — on Plus the GPT-6 Luna lane is 350–3,000
  *local messages* per 5h with an unpublished weekly cap, so 200k verdicts would need
  ~2.3 days of continuous best-case allowance with none left for real work. The whole
  100k corpus costs ~$15 at API rates.
