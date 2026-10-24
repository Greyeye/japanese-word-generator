# Assistant-authored generation and review

Use `agent_pipeline.py` for new sentence work. The current assistant writes and
reviews Japanese directly through files. **Do not call a model API, launch Codex,
read subscription credentials, import legacy generation/loader modules, or upload
anything to a database.** Existing API generators remain legacy opt-in tools.
A subscription is not an API credential and this workflow does not need one.

## File handoff

Run from `H:/jlpt/gen`. Choose a NEW directory for every output bundle under
`H:/jlpt/runs` (tests must use the Hermes scratch directory). Never modify old runs,
source data, migrations, the global ledger, or sealed output files. The shared
SQLite sentence registry is the deliberate mutable exception. Commands refuse
existing output paths; there is intentionally no `--force` option.

## Shared exact-string corpus registry

Before production ingestion, the operator must explicitly initialize and seed
`H:/jlpt/runs/sentence_registry.sqlite3` from ALL authoritative corpus JSONLs.
Do not silently start with an empty registry or discover sources by glob. These
are templates, not authorization to seed a production registry:

```bash
python sentence_registry.py check --sources H:/jlpt/runs/CORPUS_A.jsonl H:/jlpt/runs/CORPUS_B.jsonl
python sentence_registry.py init --registry H:/jlpt/runs/sentence_registry.sqlite3
python sentence_registry.py seed --registry H:/jlpt/runs/sentence_registry.sqlite3 --sources H:/jlpt/runs/CORPUS_A.jsonl H:/jlpt/runs/CORPUS_B.jsonl
python sentence_registry.py status --registry H:/jlpt/runs/sentence_registry.sqlite3
```

`seed` requires strict UTF-8 JSONL objects with a nonempty string `japanese`
field (other corpus metadata is allowed). It validates every supplied source
before one transactional mutation, returns inserted/existing counts, preserves
first source path/SHA-256/line, and is safe to reseed. `init` refuses existing
paths. `status` is read-only. `check` never opens/writes a registry: it reports
within/across-source exact duplicates, first origins, row and unique counts;
exit 0 means no duplicates, 1 means duplicates, 2 means invalid input.

Ingest defaults to that shared registry; `--registry PATH` selects an explicitly
initialized scratch/test registry, not a dedupe opt-out. Missing/uninitialized
registries fail closed. The registry uses one table with a binary UNIQUE primary
key on raw Japanese text: punctuation, whitespace and composed/decomposed Unicode
remain distinct. No normalization or semantic dedupe occurs. All workers/runs must
use the SAME local registry; separate databases cannot enforce global uniqueness.
Use a local filesystem with working SQLite locks, not a remote/network share.

Example commands below are templates; replace the run directory and provenance
labels with actual values. Do not invent a model version, request ID, or request
hash. If model identity is unavailable, explicitly declare `unknown` rather than
claiming a precise model. Actor should identify the assistant session/operator.

```bash
python agent_pipeline.py prepare --out H:/jlpt/runs/agent-new/jobs --gp arimasu-existence-inanimate --slots 3 --seed 17
```

- Omit `--gp` for all targets. `--slots` is the number of sentences per target,
  defaults to 1, and deliberately does not inherit legacy `target_count`.
- Optional `--targets PATH` and `--lexicon PATH` select local input files.
- Read `jobs/jobs.jsonl`. Every slot has a stable `job_id`, full target, exactly
  one required vocabulary record, system/user prompt, and content SHA-256.
- Follow the target's meaning, frame, register instruction, prompt and vocabulary.
  Write a genuinely natural, short N5 sentence, not a template filled mechanically.
  The prompt's legacy `sentences` envelope is replaced ONLY at transport level:
  write UTF-8 JSONL with exactly `job_id` and `japanese`, one row per job:

```json
{"job_id":"COPY-THE-EXACT-SLOT-ID","japanese":"机の上に本があります。"}
```

This example is a schema illustration, not a generated answer for an unseen job.
Do not add reasoning, English translations, Markdown fences, or extra fields.
Write raw responses to a new file outside the sealed bundle, e.g.
`H:/jlpt/runs/agent-new/responses.jsonl`.

```bash
python agent_pipeline.py ingest --jobs H:/jlpt/runs/agent-new/jobs --source H:/jlpt/runs/agent-new/responses.jsonl --out H:/jlpt/runs/agent-new/ingested --registry H:/jlpt/runs/sentence_registry.sqlite3 --actor ACTUAL_SESSION --model ACTUAL_OR_UNKNOWN_MODEL --provider ACTUAL_PROVIDER
python agent_pipeline.py review-export --ingested H:/jlpt/runs/agent-new/ingested --out H:/jlpt/runs/agent-new/blind
```

Ingestion preserves the exact source bytes and SHA-256. `audit.jsonl` records each
source line's errors, plus synthetic missing-job rows; `accepted.jsonl` and
`rejected.jsonl` separate mechanical results. Bad JSON, duplicate JSON keys, extra
fields, unknown/missing/duplicate job IDs, empty/non-Japanese text, target regex,
POS and required-word failures are rejected. All occurrences of a duplicate job ID
are rejected. Exact-character duplicates are rejected among otherwise accepted
rows in this handoff; punctuation/whitespace variants are NOT normalized away.
Otherwise-valid candidates atomically claim the shared registry BEFORE bundle
publication; invalid candidates never reserve text. Cross-run/corpus collisions
are `registry_collision` with prior origin in the audit. Manifests/audits record
registry path, claim identities and source provenance. SQLite serializes writers
with a 30-second busy timeout; lock/I/O errors fail closed.

Claims remain durable if publication fails or the process crashes: NEVER release
them automatically, including after review failure. This intentionally may retain
unused sentences rather than allow duplicate acceptance. An identical retry with
the same jobs, job/slot, source path/bytes, output path and provenance can recover
its own claim if the output does not exist. Different outputs/runs cannot recover
it. Existing output paths are refused before registry writes and remain immutable;
a crash leaving an incomplete directory requires operator investigation, not an
automatic overwrite or claim release.

There is no automatic model retry: duplicates are rejected AFTER generation,
BEFORE review. The assistant must regenerate rejected candidates into a NEW source
and NEW output bundle. This does not magically prevent a model emitting duplicates.
Previously claimed texts (including accepted siblings) cannot be republished under
the new output identity; regenerate only the rejected work, using fresh jobs as
needed for coverage.

A successful ingest command means the audit was written, NOT that every row passed.
Mechanical acceptance does not establish naturalness, meaning, level, or all prompt
constraints. Check accepted/rejected files before review; no accepts means export
refuses to create a review job.

## Blind six-axis review

The reviewer must read **only `blind/review.jsonl`**, not generation prompts,
generator reasoning, audits, or other reviewers' judgments. It contains sentence
ID, Japanese, target and required vocabulary context, with no generator reasoning
or previous judge reasons. Prefer a fresh independent reviewer context. Re-reading
your own sentence is still self-review even if this handoff hides provenance.

Write UTF-8 review JSONL to a new file, with exactly one row for EVERY exported ID:

```json
{"sentence_id":"COPY-THE-EXACT-SENTENCE-ID","axes":{"grammar_correct":"pass","meaning_coherent":"pass","target_used":"pass","natural":"pass","level":"pass","register":"pass"},"reason":"Specific short explanation based on the sentence and target.","target_span":"あります"}
```

Judge each axis independently:

- `grammar_correct`: grammatical Japanese, including particles and conjugation.
- `meaning_coherent`: a coherent, plausible situation, not merely matching tokens.
- `target_used`: the intended construction and meaning, not just a regex substring.
- `natural`: an idiomatic standalone example a native speaker would use.
- `level`: suitable elementary N5 vocabulary, grammar and sentence complexity.
- `register`: the target's requested register and formality.

Each value MUST be `pass`, `fail`, or `uncertain`. Use uncertainty honestly; never
turn uncertainty into a pass. `reason` must be nonempty. `target_span` is a literal
substring of the Japanese; it must be nonempty for `target_used: pass`, and may be
empty when the construction is absent. Do not submit `overall`: code derives it
(`fail` overrides `uncertain`, otherwise all six must pass). Missing axes/rows,
duplicate or unknown IDs, invalid spans, extra fields and fabricated overall
verdicts reject the whole review import without publishing a bundle.

```bash
python agent_pipeline.py review-import --exported H:/jlpt/runs/agent-new/blind --source H:/jlpt/runs/agent-new/reviews.jsonl --out H:/jlpt/runs/agent-new/reviewed --actor ACTUAL_REVIEW_SESSION --model ACTUAL_OR_UNKNOWN_MODEL --provider ACTUAL_PROVIDER
```

`reviews.jsonl` stores all accepted review records, six axes, code-derived overall,
source hash, generation/review stage and identity labels. Same actor OR same
model/provider is conservatively `self_review`; only a different actor AND a
different model/provider pair is `independent_model_review`. These labels are
explicit declarations, not authenticated proof of independence or human truth.
Do not relabel a session/model to manufacture independence.

Only all-pass records emit `loader.jsonl` rows with `japanese`,
`grammar_point_id`, `pattern`, `meaning_en`, `status`, `note`, sentence ID and review
evidence. **Status is always `needs_review`, never `validated`, even for one
independent-model review.** This is a file export, not a DB operation. Do not run
legacy `loader.py`/`push.py` as part of this workflow; they can read secrets or make
network calls. Later approval/upload is separate work requiring authorization.

## Integrity and development

Every bundle has `manifest.json` containing configuration/source SHA-256s and file
hashes, plus a `seal.json` containing its manifest hash. Consumers check hashes and
Janome version before processing. Targets, lexicon, noun classes/overrides,
guardrails, pipeline and registry code are pinned: changes require preparing a fresh job.
Hashes detect accidental edits, not an adversary who can rewrite all hashes.
The private local ledger is snapshotted in the job bundle; the global ledger is
never loaded or updated. Temporary state lives beside the requested output.
Output files are staged, the destination is reserved exclusively, and the seal is
published last; an incomplete bundle cannot be consumed.

Run tests before reporting success:

```bash
python -W error -m unittest test_sentence_registry test_agent_pipeline -v
```

Use test-driven vertical slices for changes: write one failing behavior test, run
it RED, implement only that behavior, rerun GREEN, then add the next slice. Test
outputs belong only in `C:/Users/greyeye/AppData/Local/hermes/cache/scratch`, never
in repository data or existing runs. Do not change `generate2.py`'s user edits.
