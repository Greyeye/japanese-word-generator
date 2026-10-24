"""Local exact-string corpus registry. No normalization, network or remote DB."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

DEFAULT_REGISTRY = Path("H:/jlpt/runs/sentence_registry.sqlite3")
SCHEMA = "CREATE TABLE sentences (japanese TEXT COLLATE BINARY PRIMARY KEY NOT NULL, origin TEXT NOT NULL)"


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid constant: " + x)))


def _connect(path, *, readonly=False):
    path = Path(path).resolve()
    connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                                 uri=True, timeout=30)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("uninitialized or unsupported sentence registry")
        if connection.execute("SELECT sql FROM sqlite_master WHERE name='sentences'").fetchone() != (SCHEMA,):
            raise ValueError("invalid sentence registry schema")
    except BaseException:
        connection.close()
        raise
    return connection


def initialize(path):
    """Explicitly create an empty registry; never silently replace one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(SCHEMA)
        connection.execute("PRAGMA user_version=1")


def _corpus(sources):
    if not sources:
        raise ValueError("provide an explicit nonempty list of corpus sources")
    rows = []
    for source in sources:
        path = Path(source).resolve()
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        for number, line in enumerate(raw.splitlines(), 1):
            row = _json(line.decode("utf-8"))
            if not isinstance(row, dict) or not isinstance(row.get("japanese"), str) or not row["japanese"].strip():
                raise ValueError(f"{path}:{number}: requires nonempty japanese string")
            rows.append((row["japanese"], {"kind": "seed", "source_path": str(path),
                                          "source_sha256": sha, "source_line": number}))
    return rows


def seed(path, sources):
    rows = _corpus(sources)  # Validate ALL sources before opening a writable connection.
    inserted = 0
    with closing(_connect(path)) as connection, connection:
        for text, origin in rows:
            inserted += connection.execute("INSERT OR IGNORE INTO sentences VALUES (?, ?)",
                                           (text, json.dumps(origin, ensure_ascii=False))).rowcount
    return {"inserted": inserted, "existing": len(rows) - inserted}


def claim(path, text, origin):
    """Durably reserve text. Only the identical owner and origin may recover it."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("claim requires nonempty text")
    if not isinstance(origin, dict) or not isinstance(origin.get("owner"), str) or not origin["owner"]:
        raise ValueError("claim requires stable owner")
    payload = json.dumps(origin, ensure_ascii=False, sort_keys=True)
    with closing(_connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT OR IGNORE INTO sentences VALUES (?, ?)", (text, payload))
        prior = json.loads(connection.execute("SELECT origin FROM sentences WHERE japanese=?", (text,)).fetchone()[0])
        return {"accepted": prior == origin, "origin": prior}


def status(path):
    with closing(_connect(path, readonly=True)) as connection:
        return {"count": connection.execute("SELECT count(*) FROM sentences").fetchone()[0]}


def lookup(path, text):
    with closing(_connect(path, readonly=True)) as connection:
        row = connection.execute("SELECT origin FROM sentences WHERE japanese=?", (text,)).fetchone()
        return json.loads(row[0]) if row else None


def check(sources):
    """Read-only within/across-corpus audit, independent of registry contents."""
    rows = _corpus(sources)
    seen, duplicates = {}, []
    for text, origin in rows:
        if text in seen:
            duplicates.append({"japanese": text, "first": seen[text], "duplicate": origin})
        else:
            seen[text] = origin
    return {"rows": len(rows), "unique": len(seen), "duplicates": duplicates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "seed", "status", "check"):
        sub = commands.add_parser(name)
        if name != "check":
            sub.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
        if name in ("seed", "check"):
            sub.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize(args.registry)
            result = status(args.registry)
        elif args.command == "seed":
            result = seed(args.registry, args.sources)
        elif args.command == "status":
            result = status(args.registry)
        else:
            result = check(args.sources)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return int(args.command == "check" and bool(result["duplicates"]))


if __name__ == "__main__":
    raise SystemExit(main())
