"""Offline workflow regression tests. Run: python -m unittest test_agent_pipeline -v."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import sentence_registry as R

import agent_pipeline as A

HERE = Path(__file__).resolve().parent


class PipelineTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "AppData/Local/hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry.sqlite3"
        R.initialize(self.registry)
        connect = R._connect
        def scratch_only(path, **kwargs):
            self.assertTrue(Path(path).resolve().is_relative_to(self.root.resolve()),
                            "tests must explicitly use scratch registry, never production")
            return connect(path, **kwargs)
        self.registry_guard = patch.object(R, "_connect", side_effect=scratch_only)
        self.registry_guard.start()
        self.addCleanup(self.registry_guard.stop)

    def cleanup(self):
        # Windows scanners can briefly hold deletion-pending files/directories.
        for attempt in range(5):
            try:
                self.tmp.cleanup()
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.05)

    def prepare(self, name="jobs", slots=1):
        out = self.root / name
        A.prepare(out, gp="arimasu-existence-inanimate", slots=slots, seed=17)
        return out

    def rows(self, folder, name):
        return [json.loads(x) for x in (folder / name).read_text(encoding="utf-8").splitlines()]

    def ingest(self, jobs, rows, name="ingest"):
        source = self.root / (name + "-raw.jsonl")
        source.write_bytes(A.lines(rows))
        out = self.root / name
        A.ingest(jobs, source, out, registry=self.registry, actor="assistant-session", model="declared-model", provider="subscription")
        return out, source

    def test_registry_rejects_other_runs_records_origin_and_pins_code(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "があります。"
        first, source = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": text}])
        second, _ = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": text}], "other-run")
        self.assertEqual(self.rows(second, "accepted.jsonl"), [])
        rejected = self.rows(second, "rejected.jsonl")[0]
        self.assertEqual(rejected["errors"], ["registry_collision"])
        accepted = self.rows(first, "accepted.jsonl")[0]
        self.assertEqual(rejected["registry"]["origin"], accepted["registry"]["origin"])
        self.assertEqual(accepted["source_path"], str(source.resolve()))
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["registry_path"], str(self.registry.resolve()))
        self.assertEqual(manifest["registry_claim_ids"], [accepted["registry"]["claim_id"]])
        self.assertTrue(any(Path(s["path"]).name == "sentence_registry.py" for s in manifest["config"]["sources"]))

    def test_seeded_corpus_collision_keeps_first_source_provenance(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "があります。"
        corpus = self.root / "corpus.jsonl"
        corpus.write_bytes(A.lines([{"japanese": text, "status": "existing"}]))
        R.seed(self.registry, [corpus])
        before = self.registry.read_bytes()
        output, _ = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": text}])
        rejected = self.rows(output, "rejected.jsonl")[0]
        self.assertEqual(rejected["errors"], ["registry_collision"])
        self.assertEqual(rejected["registry"]["origin"]["source_path"], str(corpus.resolve()))
        self.assertEqual(self.registry.read_bytes(), before)
        self.assertEqual(A.ingest.__kwdefaults__["registry"], Path("H:/jlpt/runs/sentence_registry.sqlite3"))
        with self.assertRaisesRegex(AssertionError, "scratch registry"):
            A.ingest(jobs, corpus, self.root / "no-default-test", actor="a", model="m", provider="p")

    def test_publish_failure_keeps_claim_and_identical_retry_recovers(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "があります。"
        source = self.root / "retry.jsonl"
        source.write_bytes(A.lines([{"job_id": job["job_id"], "japanese": text}]))
        output = self.root / "recovered"
        labels = dict(actor="a", model="m", provider="p", registry=self.registry)
        # Inject a real publish I/O failure after the durable SQLite commit.
        with patch.object(A.os, "replace", side_effect=OSError("publish unavailable")):
            with self.assertRaisesRegex(OSError, "publish unavailable"):
                A.ingest(jobs, source, output, **labels)
        self.assertFalse(output.exists())
        self.assertEqual(R.status(self.registry)["count"], 1)
        owner = R.lookup(self.registry, text)
        other = self.root / "different-output"
        A.ingest(jobs, source, other, **labels)
        self.assertEqual(self.rows(other, "accepted.jsonl"), [])
        A.ingest(jobs, source, output, **labels)
        self.assertEqual(len(self.rows(output, "accepted.jsonl")), 1)
        self.assertEqual(R.lookup(self.registry, text), owner)
        before = self.registry.read_bytes()
        with self.assertRaises(FileExistsError):
            A.ingest(jobs, source, output, **labels)
        self.assertEqual(self.registry.read_bytes(), before)

    def test_existing_output_preflight_never_reserves_new_candidate(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        source = self.root / "new.jsonl"
        source.write_bytes(A.lines([{"job_id": job["job_id"], "japanese": job["required"][0]["text"] + "があります。"}]))
        output = self.root / "occupied"
        output.mkdir()
        before = self.registry.read_bytes()
        with self.assertRaises(FileExistsError):
            A.ingest(jobs, source, output, registry=self.registry, actor="a", model="m", provider="p")
        self.assertEqual(self.registry.read_bytes(), before)
        self.assertEqual(R.status(self.registry)["count"], 0)

    def test_invalid_candidates_never_reserve_and_missing_registry_fails_closed(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        ingested, source = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": "あります。"}])
        self.assertEqual(self.rows(ingested, "accepted.jsonl"), [])
        self.assertEqual(R.status(self.registry)["count"], 0)
        missing = self.root / "missing.sqlite3"
        with self.assertRaises(R.sqlite3.Error):
            A.ingest(jobs, source, self.root / "closed", registry=missing, actor="a", model="m", provider="p")
        self.assertFalse(missing.exists())
        self.assertFalse((self.root / "closed").exists())

    def test_ingest_checks_schema_coverage_and_exact_dedupe(self):
        jobs = self.prepare(slots=5)
        records = self.rows(jobs, "jobs.jsonl")
        # The existence target can naturally contain any of its required nouns.
        words = [r["required"][0]["text"] for r in records]
        text = "と".join(words) + "があります。"
        out, source = self.ingest(jobs, [
            {"job_id": records[0]["job_id"], "japanese": text},
            {"job_id": records[1]["job_id"], "japanese": text},
            {"job_id": records[2]["job_id"], "japanese": text[:-1] + "！"},
            {"job_id": records[3]["job_id"], "japanese": words[3] + "です。"},
            {"job_id": records[4]["job_id"], "japanese": "あります。"}])
        self.assertEqual((out / "source.jsonl").read_bytes(), source.read_bytes())
        accepted = self.rows(out, "accepted.jsonl")
        self.assertEqual([r["japanese"] for r in accepted], [text, text[:-1] + "！"])
        rejected = self.rows(out, "rejected.jsonl")
        self.assertEqual([r["errors"] for r in rejected], [["duplicate_japanese"], ["regex_mismatch"], ["required_word_missing"]])
        self.assertEqual(rejected[0]["registry"]["origin"], accepted[0]["registry"]["origin"])
        self.assertEqual(len(self.rows(out, "audit.jsonl")), 5)
        self.assertEqual(accepted[0]["provenance"]["stage"], "generation")
        self.assertNotIn("request_hash", accepted[0]["provenance"])
        with self.assertRaises(FileExistsError):
            A.ingest(jobs, source, out, registry=self.registry, actor="a", model="m", provider="p")

    def test_blind_review_roundtrip_keeps_self_review_needs_review(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "があります。"
        ingested, _ = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": text}])
        exported = self.root / "blind"
        A.review_export(ingested, exported)
        blind = self.rows(exported, "review.jsonl")[0]
        self.assertEqual(set(blind), {"sentence_id", "japanese", "target", "required"})
        self.assertNotIn("provenance", blind)
        source = self.root / "reviews.jsonl"
        row = {"sentence_id": blind["sentence_id"], "axes": {axis: "pass" for axis in A.AXES},
               "reason": "Fits target and elementary register.", "target_span": "あります"}
        source.write_bytes(A.lines([row]))
        out = self.root / "reviewed"
        A.review_import(exported, source, out, actor="assistant-session", model="declared-model", provider="subscription")
        result = self.rows(out, "reviews.jsonl")[0]
        self.assertEqual(result["overall"], "pass")
        self.assertEqual(result["review_relationship"], "self_review")
        loader = self.rows(out, "loader.jsonl")[0]
        self.assertEqual(loader["status"], "needs_review")
        self.assertEqual(loader["japanese"], text)
        self.assertEqual(loader["grammar_point_id"], job["target"]["grammar_point_id"])
        self.assertTrue(loader["note"])
        self.assertIsInstance(loader["note"], list, "legacy loader stores note as a text[]")
        self.assertIn("evidence", loader)
        self.assertEqual((out / "source.jsonl").read_bytes(), source.read_bytes())

    def test_unknown_pos_check_is_rejected_before_publish(self):
        targets = json.loads((HERE / "data/targets.json").read_text(encoding="utf-8"))
        target = targets["targets"][0]
        target["pos_check"] = ["invented_check"]
        path = self.root / "targets.json"
        path.write_text(json.dumps({"targets": [target]}), encoding="utf-8")
        out = self.root / "unknown-pos"
        with self.assertRaisesRegex(ValueError, "unknown POS"):
            A.prepare(out, targets=path)
        self.assertFalse(out.exists())

    def test_ingest_actual_pos_check_rejects_na_adjective_predicate(self):
        out = self.root / "copula"
        A.prepare(out, gp="deshita-past-polite-copula", seed=17)
        job = self.rows(out, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "は静かでした。"
        ingested, _ = self.ingest(out, [{"job_id": job["job_id"], "japanese": text}])
        self.assertIn("pos_check_failed", self.rows(ingested, "rejected.jsonl")[0]["errors"])

    def test_tampered_manifest_jobs_and_configuration_fail_closed(self):
        for name in ("manifest.json", "jobs.jsonl"):
            with self.subTest(name=name):
                jobs = self.prepare(name.replace(".", "-"))
                (jobs / name).write_bytes((jobs / name).read_bytes() + b" ")
                source = self.root / "empty.jsonl"
                source.write_bytes(b"")
                out = self.root / (name + "-out")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    A.ingest(jobs, source, out, registry=self.registry, actor="a", model="m", provider="p")
                self.assertFalse(out.exists())
        custom = self.root / "targets.json"
        custom.write_bytes((HERE / "data/targets.json").read_bytes())
        jobs = self.root / "custom"
        A.prepare(jobs, targets=custom, gp="arimasu-existence-inanimate")
        custom.write_bytes(custom.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "configuration hash mismatch"):
            A.ingest(jobs, source, self.root / "changed", registry=self.registry, actor="a", model="m", provider="p")

    def test_ingest_bad_rows_preserves_bytes_and_rejects_duplicate_ids(self):
        jobs = self.prepare(slots=2)
        job = self.rows(jobs, "jobs.jsonl")[0]
        raw = (b'{oops}\r\n' + A.lines([
            {"job_id": job["job_id"], "japanese": "あります。"},
            {"job_id": job["job_id"], "japanese": "あります。"},
            {"job_id": "unknown", "japanese": "あります。"},
            {"japanese": "あります。"},
            {"job_id": [], "japanese": "あります。"},
            {"job_id": "other", "japanese": 42},
            {"job_id": "other", "japanese": "", "reason": "not allowed"}])
            + b'{"job_id":"a","job_id":"b","japanese":"x"}\n' + b'\xff\n')
        source = self.root / "bad.jsonl"
        source.write_bytes(raw)
        out = self.root / "bad-out"
        A.ingest(jobs, source, out, registry=self.registry, actor="a", model="m", provider="p")
        self.assertEqual((out / "source.jsonl").read_bytes(), raw)
        self.assertEqual(self.rows(out, "accepted.jsonl"), [])
        errors = [e for r in self.rows(out, "rejected.jsonl") for e in r["errors"]]
        for expected in ("malformed_json", "duplicate_job_id", "unknown_job_id", "invalid_schema", "missing_job_id"):
            self.assertIn(expected, errors)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(raw).hexdigest())

    def test_custom_lexicon_transitivity_and_empty_pool_fail_closed(self):
        lexicon = self.root / "lexicon.json"
        lexicon.write_text(json.dumps({"n5_vocab": [{"text": "吸う", "pos": "verb", "pos_raw": ["vi"], "meaning": "inhale"}]}), encoding="utf-8")
        before = A.G._POS_RAW
        out = self.root / "empty-pool"
        with self.assertRaisesRegex(ValueError, "empty constrained"):
            A.prepare(out, lexicon=lexicon, gp="particle-o-object")
        self.assertIs(A.G._POS_RAW, before)
        self.assertFalse(out.exists())

    def test_cli_all_four_stages_and_no_implicit_provenance(self):
        def cli(*args, ok=True):
            result = subprocess.run([sys.executable, str(HERE / "agent_pipeline.py"), *map(str, args)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0 if ok else 2, result.stderr)
            return result
        jobs = self.root / "cli-jobs"
        cli("prepare", "--out", jobs, "--gp", "arimasu-existence-inanimate", "--seed", 17)
        self.assertTrue(jobs.exists())
        job = self.rows(jobs, "jobs.jsonl")[0]
        source = self.root / "cli-raw.jsonl"
        source.write_bytes(A.lines([{"job_id": job["job_id"], "japanese": job["required"][0]["text"] + "があります。"}]))
        ingested = self.root / "cli-ingested"
        cli("ingest", "--registry", self.registry, "--jobs", jobs, "--source", source, "--out", ingested, ok=False)
        labels = ["--actor", "session", "--model", "declared", "--provider", "subscription"]
        cli("ingest", "--registry", self.registry, "--jobs", jobs, "--source", source, "--out", ingested, *labels)
        blind = self.root / "cli-blind"
        cli("review-export", "--ingested", ingested, "--out", blind)
        sentence = self.rows(blind, "review.jsonl")[0]
        reviews = self.root / "cli-reviews.jsonl"
        reviews.write_bytes(A.lines([{"sentence_id": sentence["sentence_id"], "axes": {a: "pass" for a in A.AXES},
                                     "reason": "Fits target.", "target_span": "あります"}]))
        reviewed = self.root / "cli-reviewed"
        cli("review-import", "--exported", blind, "--source", reviews, "--out", reviewed, *labels)
        self.assertEqual(self.rows(reviewed, "loader.jsonl")[0]["status"], "needs_review")
        cli("review-export", "--ingested", ingested, "--out", blind, ok=False)

    def review_fixture(self):
        jobs = self.prepare()
        job = self.rows(jobs, "jobs.jsonl")[0]
        text = job["required"][0]["text"] + "があります。"
        ingested, _ = self.ingest(jobs, [{"job_id": job["job_id"], "japanese": text}])
        blind = self.root / "blind"
        A.review_export(ingested, blind)
        row = {"sentence_id": self.rows(blind, "review.jsonl")[0]["sentence_id"],
               "axes": {a: "pass" for a in A.AXES}, "reason": "Fits the target.", "target_span": "あります"}
        return blind, row

    def test_review_strict_coverage_axes_span_and_fabricated_overall(self):
        blind, good = self.review_fixture()
        variants = {
            "missing_id": [{k: v for k, v in good.items() if k != "sentence_id"}],
            "unknown_id": [{**good, "sentence_id": "unknown"}],
            "duplicate_id": [good, good],
            "missing_row": [],
            "missing_axis": [{**good, "axes": {k: v for k, v in good["axes"].items() if k != "natural"}}],
            "extra_axis": [{**good, "axes": {**good["axes"], "overall": "pass"}}],
            "boolean_axis": [{**good, "axes": {**good["axes"], "natural": True}}],
            "fabricated_overall": [{**good, "overall": "validated"}],
            "empty_reason": [{**good, "reason": ""}],
            "bad_span": [{**good, "target_span": "ない"}],
            "empty_passing_span": [{**good, "target_span": ""}],
            "nonobject": [None],
        }
        for name, rows in variants.items():
            with self.subTest(name=name):
                source = self.root / (name + ".jsonl")
                source.write_bytes(A.lines(rows))
                out = self.root / name
                with self.assertRaises(ValueError):
                    A.review_import(blind, source, out, actor="reviewer", model="other", provider="other")
                self.assertFalse(out.exists())
        source = self.root / "malformed-review.jsonl"
        source.write_bytes(b'{oops}\n')
        with self.assertRaises(ValueError):
            A.review_import(blind, source, self.root / "malformed", actor="a", model="m", provider="p")

    def test_review_uncertainty_failure_and_declared_independence(self):
        blind, good = self.review_fixture()
        for verdict in ("pass", "uncertain", "fail"):
            with self.subTest(verdict=verdict):
                source = self.root / (verdict + ".jsonl")
                source.write_bytes(A.lines([{**good, "axes": {**good["axes"], "natural": verdict}}]))
                out = self.root / verdict
                A.review_import(blind, source, out, actor="different-session", model="different-model", provider="subscription")
                result = self.rows(out, "reviews.jsonl")[0]
                self.assertEqual(result["overall"], verdict)
                self.assertEqual(result["review_relationship"], "independent_model_review")
                self.assertEqual(len(self.rows(out, "loader.jsonl")), 1 if verdict == "pass" else 0)
                with self.assertRaises(FileExistsError):
                    A.review_import(blind, source, out, actor="a", model="m", provider="p")
        # Same model is not independent merely because the actor/session changed.
        source.write_bytes(A.lines([good]))
        out = self.root / "same-model"
        A.review_import(blind, source, out, actor="different-session", model="declared-model", provider="subscription")
        self.assertEqual(self.rows(out, "reviews.jsonl")[0]["review_relationship"], "self_review")

    def test_import_and_full_runtime_deny_network_secrets_db_and_processes(self):
        script = r'''
import sys, pathlib, os
sys.dont_write_bytecode = True
repo, work = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(repo))
allowed = work.resolve()
def audit(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system"):
        raise AssertionError("forbidden runtime event: " + event)
    if event == "sqlite3.connect":
        value = os.fsdecode(args[0])
        if value.startswith("file:"):
            from urllib.parse import urlsplit, unquote
            value = unquote(urlsplit(value).path).lstrip("/")
        p = pathlib.Path(value)
        if not p.resolve().is_relative_to(allowed):
            raise AssertionError("SQLite outside scratch: " + str(p))
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        p = pathlib.Path(os.fsdecode(args[0])).resolve()
        if p.name in ("generate.py", "generate2.py", "loader.py", "push.py", ".env") or "secret" in p.name.lower():
            raise AssertionError("forbidden secret/legacy read: " + str(p))
        flags = args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if not p.is_relative_to(allowed):
                raise AssertionError("write outside scratch: " + str(p))
sys.addaudithook(audit)
import agent_pipeline as A
jobs, ingested, blind, reviewed = [work / n for n in ("jobs", "ingest", "blind", "reviewed")]
A.prepare(jobs, gp="arimasu-existence-inanimate")
job = A.read_rows((jobs / "jobs.jsonl").read_bytes())[0]
source = work / "source.jsonl"
source.write_bytes(A.lines([{"job_id": job["job_id"], "japanese": job["required"][0]["text"] + "があります。"}]))
A.ingest(jobs, source, ingested, registry=work / "registry.sqlite3", actor="a", model="m", provider="p")
A.review_export(ingested, blind)
sentence = A.read_rows((blind / "review.jsonl").read_bytes())[0]
source.write_bytes(A.lines([{"sentence_id": sentence["sentence_id"], "axes": {a:"pass" for a in A.AXES}, "reason":"Fits.", "target_span":"あります"}]))
A.review_import(blind, source, reviewed, actor="a", model="m", provider="p")
assert not {"generate", "generate2", "loader", "push", "psycopg", "supabase"}.intersection(sys.modules)
print("offline import and four stages OK")
'''
        result = subprocess.run([sys.executable, "-c", script, str(HERE), str(self.root)],
                                capture_output=True, text=True, encoding="utf-8", cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("offline import and four stages OK", result.stdout)

    def test_empty_and_non_japanese_text_never_pass(self):
        jobs = self.prepare(slots=2)
        records = self.rows(jobs, "jobs.jsonl")
        ingested, _ = self.ingest(jobs, [
            {"job_id": records[0]["job_id"], "japanese": "  "},
            {"job_id": records[1]["job_id"], "japanese": "not Japanese"}])
        self.assertEqual(self.rows(ingested, "accepted.jsonl"), [])
        rejected = self.rows(ingested, "rejected.jsonl")
        self.assertIn("empty_japanese", rejected[0]["errors"])
        self.assertIn("not_japanese", rejected[1]["errors"])
        for field in ("actor", "model", "provider"):
            labels = dict(actor="a", model="m", provider="p")
            labels[field] = " "
            with self.assertRaises(ValueError):
                A.ingest(jobs, self.root / "ingest-raw.jsonl", self.root / field, registry=self.registry, **labels)

    def test_prepare_real_target_is_reproducible_and_isolated(self):
        global_ledger = Path(A.G.RUNS) / "ledger.json"
        before = global_ledger.read_bytes() if global_ledger.exists() else None
        state = random.getstate()
        one, two = self.prepare(), self.prepare("again")
        jobs = self.rows(one, "jobs.jsonl")
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["target"]["grammar_point_id"], "arimasu-existence-inanimate")
        self.assertEqual(len(job["required"]), 1)
        self.assertIn(job["required"][0]["text"], job["prompt"]["user"])
        self.assertEqual(job["prompt"]["system"], A.G.system_for("arimasu-existence-inanimate"))
        self.assertEqual(len(job["content_sha256"]), 64)
        self.assertEqual((one / "jobs.jsonl").read_bytes(), (two / "jobs.jsonl").read_bytes())
        self.assertEqual(state, random.getstate())
        self.assertEqual(before, global_ledger.read_bytes() if global_ledger.exists() else None)
        self.assertTrue((one / "ledger.json").exists())
        with self.assertRaises((ValueError, FileExistsError)):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
