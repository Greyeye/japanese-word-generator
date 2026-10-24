"""Offline, immutable file handoffs for assistant-authored Japanese sentences.

Only guardrails, Janome and local SQLite; no model client, credentials or remote DB.
Hashes detect accidental changes, not an adversary able to rewrite every seal.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

import guardrails as G
import sentence_registry as R

HERE = Path(__file__).resolve().parent
AXES = ("grammar_correct", "meaning_coherent", "target_used", "natural", "level", "register")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def lines(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid constant: " + x)))


def read_rows(raw):
    return [strict_json(row) for row in raw.decode("utf-8").splitlines()]


def publish(out, kind, config, files, **metadata):
    """Reserve exclusively; stage bytes first, publish seal last. Never overwrite."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "kind": kind, "config": config, **metadata,
                "files": {name: digest(raw) for name, raw in files.items()}}
    files = {**files, "manifest.json": encoded(manifest)}
    files["seal.json"] = encoded({"manifest_sha256": digest(files["manifest.json"])})
    with tempfile.TemporaryDirectory(prefix=".agent-stage-", dir=out.parent) as tmp:
        stage = Path(tmp)
        for name, raw in files.items():
            (stage / name).write_bytes(raw)
        out.mkdir()  # exclusive reservation, including across concurrent writers
        try:
            for name in files:  # seal is last: incomplete bundles fail closed
                os.replace(stage / name, out / name)
        except BaseException:
            shutil.rmtree(out)
            raise
    return out


def config_for(targets, lexicon):
    paths = [Path(targets).resolve(), Path(lexicon).resolve(), Path(G.__file__).resolve(),
             Path(__file__).resolve(), Path(R.__file__).resolve(), Path(G._NC_PATH), Path(G._NCO_PATH)]
    # Guardrails may otherwise silently continue on missing class input.
    return {"sources": [{"path": str(p), "sha256": digest(p.read_bytes())} for p in paths],
            "janome_version": importlib.metadata.version("Janome")}


def load_bundle(folder, kind):
    folder = Path(folder)
    raw = (folder / "manifest.json").read_bytes()
    seal = strict_json((folder / "seal.json").read_bytes())
    if seal != {"manifest_sha256": digest(raw)}:
        raise ValueError("manifest hash mismatch")
    manifest = strict_json(raw)
    if manifest["version"] != 1 or manifest["kind"] != kind:
        raise ValueError("unexpected bundle kind/version")
    for source in manifest["config"]["sources"]:
        if digest(Path(source["path"]).read_bytes()) != source["sha256"]:
            raise ValueError("configuration hash mismatch: " + source["path"])
    if manifest["config"]["janome_version"] != importlib.metadata.version("Janome"):
        raise ValueError("Janome version mismatch")
    files = {}
    for name, expected in manifest["files"].items():
        if Path(name).name != name or name in ("manifest.json", "seal.json"):
            raise ValueError("unsafe bundle filename")
        files[name] = (folder / name).read_bytes()
        if digest(files[name]) != expected:
            raise ValueError("file hash mismatch: " + name)
    return manifest, files


def prepare(out, *, targets=HERE / "data/targets.json", lexicon=HERE / "data/lexicon.json",
            gp=None, slots=1, seed=0):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if type(slots) is not int or slots < 1 or type(seed) is not int:
        raise ValueError("slots must be positive and seed an integer")
    config = config_for(targets, lexicon)
    all_targets = strict_json(Path(targets).read_bytes())["targets"]
    selected = [t for t in all_targets if gp is None or t["grammar_point_id"] == gp]
    if not selected:
        raise ValueError("no matching targets")
    if len({t["job_id"] for t in selected}) != len(selected):
        raise ValueError("duplicate target job IDs")
    lex = strict_json(Path(lexicon).read_bytes())
    pools = G.build_pools(lex)
    # Explicit class load avoids a stale guardrails class cache.
    classes = strict_json(Path(G._NC_PATH).read_bytes())
    overrides = strict_json(Path(G._NCO_PATH).read_bytes())
    for word, value in overrides.items():
        if not word.startswith("_"):
            classes.setdefault(word, {}).update(value)
    jobs = []
    state = random.getstate()
    old_pos_raw = G._POS_RAW
    try:
        G._POS_RAW = {v["text"]: set(v.get("pos_raw") or ()) for v in lex["n5_vocab"]}
        random.seed(seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agent-ledger-", dir=out.parent) as tmp:
            ledger = G.Ledger(path=str(Path(tmp) / "ledger.json"))
            for target in selected:
                point = target["grammar_point_id"]
                if not isinstance(target.get("check_regex"), str) or not target["check_regex"]:
                    raise ValueError("target requires check_regex")
                re.compile(target["check_regex"])
                checks = target.get("pos_check", [])
                if not isinstance(checks, list) or any(not isinstance(n, str) or n not in G.POS_CHECKS for n in checks):
                    raise ValueError("unknown POS check")
                need = G.pos_need(target["pattern"], target["meaning_en"], point)
                pool = G.filter_pool(point, pools.get(need, []), classes)
                # Existing filter_pool falls back to unfiltered input: fail closed here.
                pool = [v for v in pool if G.noun_ok(point, v["text"], classes)]
                if not pool:
                    raise ValueError("empty constrained vocabulary pool: " + point)
                for slot in range(slots):
                    required = ledger.draw(point, pool, 1)
                    ledger.bump(point, required[0]["text"])
                    job = {"job_id": f"{target['job_id']}-s{slot + 1:04d}",
                           "slot": slot + 1, "target": target, "required": required,
                           "pos_need": need,
                           "prompt": {"system": G.system_for(point),
                                      "user": G.build_user(target, required, 1)},
                           "response_schema": {"job_id": "copy job_id exactly", "japanese": "one sentence"}}
                    job["content_sha256"] = digest(encoded(job))
                    jobs.append(job)
            ledger_data = ledger.data
    finally:
        G._POS_RAW = old_pos_raw
        random.setstate(state)
    return publish(out, "jobs", config,
                   {"jobs.jsonl": lines(jobs), "ledger.json": encoded(ledger_data)},
                   seed=seed, slots_per_target=slots, job_count=len(jobs),
                   transport="Write JSONL {job_id,japanese}; transport overrides prompt JSON envelope only.")


def provenance(stage, actor, model, provider):
    labels = {"stage": stage, "actor": actor, "model": model, "provider": provider}
    if any(not isinstance(v, str) or not v.strip() for v in labels.values()):
        raise ValueError("actor/model/provider must be explicitly supplied nonempty labels")
    return labels


def source_ref(folder):
    return {"path": str(Path(folder).resolve()),
            "manifest_sha256": digest((Path(folder) / "manifest.json").read_bytes())}


def ingest(jobs, source, out, *, actor, model, provider, registry=R.DEFAULT_REGISTRY):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    registry = Path(registry).resolve()
    R.status(registry)  # Fail closed even when all candidate rows are invalid.
    manifest, files = load_bundle(jobs, "jobs")
    prov = provenance("generation", actor, model, provider)
    known = {r["job_id"]: r for r in read_rows(files["jobs.jsonl"])}
    raw = Path(source).read_bytes()
    parsed = []
    for number, row in enumerate(raw.splitlines(), 1):
        try:
            value = strict_json(row.decode("utf-8"))
            parsed.append((number, value, []))
        except (ValueError, UnicodeError):
            parsed.append((number, None, ["malformed_json"]))
    counts = Counter(v.get("job_id") for _, v, _ in parsed
                     if isinstance(v, dict) and isinstance(v.get("job_id"), str))
    audit, seen = [], set()
    for number, value, initial in parsed:
        errors = list(initial)
        record = {"source_line": number, "source_sha256": digest(raw), "errors": errors,
                  "provenance": prov, "source_path": str(Path(source).resolve()),
                  "registry": {"path": str(registry)}}
        if not errors:
            if (not isinstance(value, dict) or set(value) != {"job_id", "japanese"}
                    or not isinstance(value.get("job_id"), str)
                    or not isinstance(value.get("japanese"), str)):
                errors.append("invalid_schema")
            else:
                job_id, text = value["job_id"], value["japanese"]
                record.update(job_id=job_id, japanese=text)
                if job_id not in known:
                    errors.append("unknown_job_id")
                if counts[job_id] != 1:
                    errors.append("duplicate_job_id")
                if not text.strip():
                    errors.append("empty_japanese")
                elif not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text):
                    errors.append("not_japanese")
                if job_id in known:
                    job = known[job_id]
                    record.update(target=job["target"], required=job["required"])
                    if not re.search(job["target"]["check_regex"], text):
                        errors.append("regex_mismatch")
                    if not G.pos_ok(text, job["target"].get("pos_check", [])):
                        errors.append("pos_check_failed")
                    if not G.required_present(text, job["required"], G._get_tokenizer()):
                        errors.append("required_word_missing")
                if text in seen:
                    errors.append("duplicate_japanese")
                    record["registry"]["origin"] = R.lookup(registry, text)
                if not errors:
                    origin = {"kind": "ingest", "owner": digest(encoded({
                        "output": str(out.resolve()), "jobs": source_ref(jobs),
                        "source_path": str(Path(source).resolve()), "source_sha256": digest(raw),
                        "job_id": job_id})), "output_path": str(out.resolve()),
                        "jobs": source_ref(jobs), "job_id": job_id,
                        "source_path": str(Path(source).resolve()), "source_sha256": digest(raw),
                        "source_line": number, "provenance": prov}
                    claim = R.claim(registry, text, origin)
                    record["registry"].update(claim_id=origin["owner"], **claim)
                    if not claim["accepted"]:
                        errors.append("registry_collision")
                if not errors:
                    seen.add(text)
                    record["sentence_id"] = "s-" + digest(encoded({"job_id": job_id, "japanese": text}))
        record["mechanically_accepted"] = not errors
        audit.append(record)
    for job_id in sorted(known.keys() - counts.keys()):
        audit.append({"job_id": job_id, "source_line": None, "source_sha256": digest(raw),
                      "errors": ["missing_job_id"], "mechanically_accepted": False, "provenance": prov,
                      "source_path": str(Path(source).resolve()), "registry": {"path": str(registry)}})
    return publish(out, "ingest", manifest["config"],
                   {"source.jsonl": raw, "jobs.jsonl": files["jobs.jsonl"],
                    "audit.jsonl": lines(audit),
                    "accepted.jsonl": lines(r for r in audit if r["mechanically_accepted"]),
                    "rejected.jsonl": lines(r for r in audit if not r["mechanically_accepted"])},
                   source=source_ref(jobs), provenance=prov, source_sha256=digest(raw),
                   source_path=str(Path(source).resolve()), registry_path=str(registry),
                   registry_claim_ids=[r["registry"]["claim_id"] for r in audit
                                       if r["mechanically_accepted"]])


def review_export(ingested, out):
    manifest, files = load_bundle(ingested, "ingest")
    accepted = read_rows(files["accepted.jsonl"])
    if not accepted:
        raise ValueError("no mechanically accepted sentences to review")
    blind = [{key: row[key] for key in ("sentence_id", "japanese", "target", "required")}
             for row in accepted]
    return publish(out, "review_export", manifest["config"], {"review.jsonl": lines(blind)},
                   source=source_ref(ingested), generation_provenance=manifest["provenance"],
                   axes=list(AXES), instructions="Review only review.jsonl, without opening source bundles. "
                   "Return sentence_id, axes (each pass/fail/uncertain), reason, target_span. No overall.")


def review_import(exported, source, out, *, actor, model, provider):
    manifest, files = load_bundle(exported, "review_export")
    prov = provenance("review", actor, model, provider)
    expected = {r["sentence_id"]: r for r in read_rows(files["review.jsonl"])}
    raw = Path(source).read_bytes()
    reviews = read_rows(raw)
    ids = []
    for row in reviews:
        if not isinstance(row, dict) or set(row) != {"sentence_id", "axes", "reason", "target_span"}:
            raise ValueError("invalid review schema (overall is derived, never supplied)")
        if not isinstance(row["sentence_id"], str):
            raise ValueError("invalid sentence_id")
        ids.append(row["sentence_id"])
        if not isinstance(row["axes"], dict) or set(row["axes"]) != set(AXES):
            raise ValueError("review requires exactly six axes")
        if any(v not in ("pass", "fail", "uncertain") for v in row["axes"].values()):
            raise ValueError("invalid axis verdict")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("review requires nonempty reason")
        if not isinstance(row["target_span"], str):
            raise ValueError("target_span must be a literal substring or empty for absent target")
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("review IDs must cover every sentence exactly once")
    gen = manifest["generation_provenance"]
    relationship = ("self_review" if prov["actor"] == gen["actor"]
                    or (prov["model"], prov["provider"]) == (gen["model"], gen["provider"])
                    else "independent_model_review")
    results, loader = [], []
    for row in reviews:
        sentence = expected[row["sentence_id"]]
        span = row["target_span"]
        if span and span not in sentence["japanese"]:
            raise ValueError("target_span is not a literal substring")
        if row["axes"]["target_used"] == "pass" and not span:
            raise ValueError("passing target_used requires a nonempty target_span")
        verdicts = row["axes"].values()
        overall = "fail" if "fail" in verdicts else "uncertain" if "uncertain" in verdicts else "pass"
        result = {**row, "overall": overall, "provenance": prov,
                  "generation_provenance": gen, "review_relationship": relationship,
                  "source_sha256": digest(raw)}
        results.append(result)
        if overall == "pass":
            target = sentence["target"]
            loader.append({"japanese": sentence["japanese"],
                           "grammar_point_id": target["grammar_point_id"],
                           "pattern": target["pattern"], "meaning_en": target["meaning_en"],
                           "status": "needs_review", "note": [relationship + ": " + row["reason"]],
                           "sentence_id": row["sentence_id"], "evidence": result})
    return publish(out, "review_import", manifest["config"],
                   {"source.jsonl": raw, "review.jsonl": files["review.jsonl"],
                    "reviews.jsonl": lines(results), "loader.jsonl": lines(loader)},
                   source=source_ref(exported), provenance=prov, review_relationship=relationship,
                   source_sha256=digest(raw))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="prepare deterministic one-word generation slots")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--targets", type=Path, default=HERE / "data/targets.json")
    prep.add_argument("--lexicon", type=Path, default=HERE / "data/lexicon.json")
    prep.add_argument("--gp", help="one grammar_point_id; omit for all targets")
    prep.add_argument("--slots", type=int, default=1, help="slots per target, default 1")
    prep.add_argument("--seed", type=int, default=0)
    ingest_parser = commands.add_parser("ingest", help="audit assistant-written JSONL")
    ingest_parser.add_argument("--jobs", type=Path, required=True)
    ingest_parser.add_argument("--registry", type=Path, default=R.DEFAULT_REGISTRY,
                               help="initialized shared exact-string SQLite registry")
    export_parser = commands.add_parser("review-export", help="export blind mechanical accepts")
    export_parser.add_argument("--ingested", type=Path, required=True)
    import_parser = commands.add_parser("review-import", help="import six-axis review, derive overall")
    import_parser.add_argument("--exported", type=Path, required=True)
    for sub in (ingest_parser, import_parser):
        sub.add_argument("--source", type=Path, required=True)
        for label in ("actor", "model", "provider"):
            sub.add_argument("--" + label, required=True, help="explicit declared provenance, not authenticated")
    for sub in (ingest_parser, export_parser, import_parser):
        sub.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        out = {"prepare": prepare, "ingest": ingest, "review-export": review_export,
               "review-import": review_import}[command](**args)
    except (ValueError, OSError, KeyError, TypeError, R.sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps({"bundle": str(out), "stage": command}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
