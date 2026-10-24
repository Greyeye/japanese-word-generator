"""Exact-string registry tests; all writes use isolated Hermes scratch."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import sqlite3
import unittest

import sentence_registry as R


class RegistryTest(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "AppData/Local/hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "registry.sqlite3"

    def source(self, name, texts):
        path = self.root / name
        path.write_text("".join(json.dumps({"japanese": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
        return path

    def test_atomic_claim_collision_idempotent_owner_and_exact_unicode(self):
        R.initialize(self.db)
        origin = {"owner": "run-a-slot-1", "source_path": "a", "source_sha256": "abc"}
        first = R.claim(self.db, "が。", origin)
        self.assertTrue(first["accepted"])
        self.assertTrue(R.claim(self.db, "が。", origin)["accepted"])
        collision = R.claim(self.db, "が。", {**origin, "owner": "run-b"})
        self.assertFalse(collision["accepted"])
        self.assertEqual(collision["origin"], origin)
        for text in ("か\u3099。", "が！", " が。", "が。 "):
            self.assertTrue(R.claim(self.db, text, origin)["accepted"])
        self.assertEqual(R.status(self.db)["count"], 5)
        with self.assertRaises(ValueError):
            R.claim(self.db, "", origin)
        with self.assertRaises(ValueError):
            R.claim(self.db, "本", {})

    def test_cli_init_seed_status_and_readonly_corpus_check(self):
        source = self.source("one.jsonl", ["本。", "本。"])
        other = self.source("two.jsonl", ["本。", "机。"])
        def cli(*args, code=0):
            result = subprocess.run([sys.executable, str(Path(R.__file__)), *map(str, args)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, code, result.stderr)
            return json.loads(result.stdout)
        cli("init", "--registry", self.db)
        self.assertEqual(cli("seed", "--registry", self.db, "--sources", source, other), {"inserted": 2, "existing": 2})
        before = self.db.read_bytes()
        self.assertEqual(cli("status", "--registry", self.db), {"count": 2})
        checked = cli("check", "--sources", source, other, code=1)
        self.assertEqual(checked["rows"], 4)
        self.assertEqual(checked["unique"], 2)
        self.assertEqual(len(checked["duplicates"]), 2)
        self.assertEqual(checked["duplicates"][1]["first"]["source_path"], str(source.resolve()))
        self.assertEqual(self.db.read_bytes(), before)

    def test_missing_uninitialized_registry_fails_without_creation(self):
        with self.assertRaises(sqlite3.Error):
            R.status(self.db)
        self.assertFalse(self.db.exists())
        self.db.write_bytes(b"")
        before = self.db.read_bytes()
        with self.assertRaises(ValueError):
            R.claim(self.db, "本。", {"owner": "x"})
        self.assertEqual(self.db.read_bytes(), before)

    def test_all_sources_validated_before_seed_mutation(self):
        R.initialize(self.db)
        good = self.source("good.jsonl", ["本。"])
        bad = self.root / "bad.jsonl"
        for raw in (b'{oops}\n', b'{"japanese":"a","japanese":"b"}\n',
                    b'{"japanese":null}\n', b'[]\n', b'{"japanese":" "}\n', b'\xff\n'):
            with self.subTest(raw=raw):
                bad.write_bytes(raw)
                before = self.db.read_bytes()
                with self.assertRaises(ValueError):
                    R.seed(self.db, [good, bad])
                self.assertEqual(R.status(self.db)["count"], 0)
                self.assertEqual(self.db.read_bytes(), before)
        with self.assertRaises(ValueError):
            R.seed(self.db, [])

    def test_multiprocess_competitors_have_precisely_one_owner(self):
        R.initialize(self.db)
        script = '''import json, pathlib, sys, time
import sentence_registry as R
db, gate, owner = sys.argv[1:]
while not pathlib.Path(gate).exists():
    time.sleep(.01)
print(json.dumps(R.claim(db, "same Japanese text", {"owner": owner})))
'''
        gate = self.root / "go"
        processes = [subprocess.Popen([sys.executable, "-W", "error", "-c", script,
                                      str(self.db), str(gate), str(i)],
                                     cwd=Path(R.__file__).parent, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8") for i in range(6)]
        try:
            gate.touch()
            results = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)
                results.append(json.loads(stdout))
            self.assertEqual(sum(r["accepted"] for r in results), 1)
            self.assertEqual(len({r["origin"]["owner"] for r in results}), 1)
            self.assertEqual(R.status(self.db)["count"], 1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_seed_preserves_unicode_line_separators_inside_json_strings(self):
        R.initialize(self.db)
        texts = ["本\u2028です。", "本\u2029です。", "本\u0085です。"]
        source = self.source("unicode.jsonl", texts)
        self.assertEqual(R.seed(self.db, [source])["inserted"], 3)
        for text in texts:
            self.assertIsNotNone(R.lookup(self.db, text))

    def test_explicit_init_seed_idempotent_and_first_provenance(self):
        source = self.source("corpus.jsonl", ["本があります。", "本があります。", "机があります。"])
        R.initialize(self.db)
        self.assertEqual(R.seed(self.db, [source]), {"inserted": 2, "existing": 1})
        self.assertEqual(R.seed(self.db, [source]), {"inserted": 0, "existing": 3})
        self.assertEqual(R.status(self.db), {"count": 2})
        origin = R.lookup(self.db, "本があります。")
        self.assertEqual(origin["source_path"], str(source.resolve()))
        self.assertEqual(origin["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(origin["source_line"], 1)
        with self.assertRaises(FileExistsError):
            R.initialize(self.db)


if __name__ == "__main__":
    unittest.main()
