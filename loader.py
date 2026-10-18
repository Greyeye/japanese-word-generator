"""
loader.py -- turn validated corpus rows into Supabase-ready SQL.

Derives every match key here rather than in SQL, so the stored keys and any
query keys come from the same functions. In particular norm()/content_set() are
the SAME normalisation validate.py used when it decided these rows were valid --
if the loader and the validator disagreed, the tier-2/3 lookups would silently
miss rows the validator had kept.

  python loader.py                 # write chunks of SQL ready to apply
  python loader.py --no-embed      # skip the embedding pass (tier 4 left null)

Outputs into ./load/: sentences_000.sql, tokens_000.sql, ...
"""

import json, os, re, sys, time, math, hashlib, urllib.request, collections
from janome.tokenizer import Tokenizer

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

LOAD = os.path.join(RUNS, "load")

_src  = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"', _src).group(1)
EMB_URL   = "http://192.168.0.12:8889/v1/embeddings"
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B-GGUF"

ROWS_PER_CHUNK = 150
EMB_BATCH      = 64

tok = Tokenizer()

CONTENT = ("名詞", "動詞", "形容詞", "副詞")
LIGHT   = {"為る", "有る", "居る", "成る", "する", "ある", "いる", "なる"}
# Particles kept for the particle_seq column and for form detection.
PARTICLES = set("はがをにでとものへやかねよなぞまでからより")

norm = lambda s: re.sub(r'[。．！？、…\s「」『』（）・]', '', s)
sha  = lambda s: hashlib.sha256(norm(s).encode("utf-8")).hexdigest()


def content_set(sentence):
    """Content-word lemmas -- identical logic to validate.py's content_set()."""
    out = set()
    for t in tok.tokenize(sentence):
        pos = t.part_of_speech.split(",")
        if pos[0] not in CONTENT:
            continue
        if pos[0] == "名詞" and pos[1] in ("数", "固有名詞"):
            continue
        lemma = t.base_form if t.base_form not in ("*", "") else t.surface
        if lemma in LIGHT:
            continue
        out.add(lemma)
    return out


def form_signature(s):
    """Coarse grammatical form. Deliberately ordered most-specific first."""
    if "ませんでした" in s:                     return "polite_past_negative"
    if "ません" in s:                          return "polite_negative"
    if "ましょう" in s:                        return "polite_volitional"
    if "ました" in s or "でした" in s:          return "polite_past"
    if "ください" in s:                        return "polite_request"
    if "です" in s or "ます" in s:             return "polite_nonpast"
    if "だった" in s:                          return "plain_past"
    return "plain"


def tokens_of(sentence):
    out = []
    for i, t in enumerate(tok.tokenize(sentence)):
        pos = t.part_of_speech.split(",")
        out.append({
            "position": i,
            "surface":  t.surface,
            "lemma":    t.base_form if t.base_form not in ("*", "") else t.surface,
            "reading":  t.reading if t.reading not in ("*", "") else None,
            "pos":      pos[0],
            "pos_detail": ",".join(pos[1:3]),
        })
    return out


def embed(texts):
    """Batch embeddings from the local Qwen3-Embedding-0.6B server."""
    vecs = []
    for i in range(0, len(texts), EMB_BATCH):
        chunk = texts[i:i + EMB_BATCH]
        req = urllib.request.Request(EMB_URL,
            data=json.dumps({"model": EMB_MODEL, "input": chunk}).encode(),
            headers={"Authorization": "Bearer " + TOKEN,
                     "Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        got = sorted(d["data"], key=lambda x: x["index"])
        vecs.extend(x["embedding"] for x in got)
        print(f"    embedded {min(i+EMB_BATCH, len(texts))}/{len(texts)}", flush=True)
    return vecs


def q(v):
    """SQL literal."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def arr(xs):
    return "null" if xs is None else "array[" + ",".join(q(x) for x in xs) + "]::text[]"


def vec(v):
    """pgvector literal, 6 decimals is well inside float32 precision."""
    return "null" if v is None else "'[" + ",".join(f"{x:.6f}" for x in v) + "]'::vector"


def arg(name, default=None):
    a = sys.argv
    if name not in a:
        return default
    i = a.index(name) + 1
    return a[i] if i < len(a) else True


def main():
    # --in makes the loader re-runnable against any corpus slice. The path used to
    # be hardcoded to v11_sentences.jsonl, so loading anything else meant editing
    # the file -- and editing the file while it also ran on import is how a
    # loader becomes untrustworthy.
    src_file = arg("--in") or os.path.join(RUNS, "v11_sentences.jsonl")
    print(f"input: {src_file}")

    # ---- read ------------------------------------------------------------------
    rows = []
    for line in open(src_file, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") in ("validated", "needs_review"):
            rows.append(r)

    # de-duplicate on norm_hash: keep the first, they are the same sentence
    seen, kept = set(), []
    for r in rows:
        h = sha(r["japanese"])
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    rows = kept
    print(f"loading {len(rows)} rows from {os.path.basename(src_file)}")

    os.makedirs(LOAD, exist_ok=True)
    for f in os.listdir(LOAD):
        if f.endswith(".sql"):
            os.remove(os.path.join(LOAD, f))

    # ---- derive ----------------------------------------------------------------
    do_embed = "--no-embed" not in sys.argv
    texts = [r["japanese"] for r in rows]
    vecs = None
    if do_embed:
        t0 = time.time()
        print(f"embedding {len(texts)} sentences with {EMB_MODEL.split('/')[-1]} ...")
        vecs = embed(texts)
        print(f"  {time.time()-t0:.1f}s total, dim={len(vecs[0])}")

    prepared = []
    for i, r in enumerate(rows, start=1):
        s = r["japanese"]
        ts = tokens_of(s)
        prepared.append({
            "id": i,
            "japanese": s,
            "norm_hash": sha(s),
            "content_lemmas": sorted(content_set(s)),
            "form_signature": form_signature(s),
            "particle_seq": [t["surface"] for t in ts if t["surface"] in PARTICLES],
            "token_count": len(ts),
            "char_count": len(s),
            "embedding": vecs[i - 1] if vecs else None,
            "grammar_point_id": r.get("grammar_point_id"),
            "register": r.get("register"),
            "model": r.get("model"),
            "job_id": r.get("job_id"),
            "status": r["status"],
            "notes": (r.get("note") or []),
            "tokens": ts,
        })

    # ---- emit ------------------------------------------------------------------
    n = 0
    for c in range(0, len(prepared), ROWS_PER_CHUNK):
        chunk = prepared[c:c + ROWS_PER_CHUNK]
        n += 1
        with open(os.path.join(LOAD, f"sentences_{n:03d}.sql"), "w", encoding="utf-8") as f:
            f.write("insert into sentences (id, japanese, norm_hash, content_lemmas, "
                    "form_signature, particle_seq, token_count, char_count, embedding, "
                    "grammar_point_id, register, source, model, job_id, status, notes) values\n")
            vals = []
            for p in chunk:
                vals.append(
                    f"({p['id']},{q(p['japanese'])},{q(p['norm_hash'])},"
                    f"{arr(p['content_lemmas'])},{q(p['form_signature'])},"
                    f"{arr(p['particle_seq'])},{p['token_count']},{p['char_count']},"
                    f"{vec(p['embedding'])},{q(p['grammar_point_id'])},{q(p['register'])},"
                    f"'generated',{q(p['model'])},{q(p['job_id'])},{q(p['status'])},"
                    f"{arr(p['notes'])})")
            f.write(",\n".join(vals) + "\non conflict (id) do nothing;\n")

        with open(os.path.join(LOAD, f"tokens_{n:03d}.sql"), "w", encoding="utf-8") as f:
            vals = []
            for p in chunk:
                for t in p["tokens"]:
                    vals.append(f"({p['id']},{t['position']},{q(t['surface'])},"
                                f"{q(t['lemma'])},{q(t['reading'])},{q(t['pos'])},"
                                f"{q(t['pos_detail'])})")
            for i in range(0, len(vals), 500):
                f.write("insert into sentence_tokens (sentence_id, position, surface, lemma, "
                        "reading, pos, pos_detail) values\n")
                f.write(",\n".join(vals[i:i + 500]) + "\non conflict do nothing;\n")

    with open(os.path.join(LOAD, "99_finalise.sql"), "w", encoding="utf-8") as f:
        f.write(f"select setval('sentences_id_seq', {len(prepared)}, true);\n")

    print(f"\nwrote {n} chunks to {LOAD}")
    tot_tok = sum(len(p["tokens"]) for p in prepared)
    print(f"  sentences: {len(prepared)}")
    print(f"  tokens   : {tot_tok}")
    print(f"  embedded : {'yes dim=' + str(len(vecs[0])) if vecs else 'no'}")
    print(f"  form signatures: {dict(collections.Counter(p['form_signature'] for p in prepared).most_common())}")
    sz = sum(os.path.getsize(os.path.join(LOAD, f)) for f in os.listdir(LOAD))
    print(f"  total SQL: {sz/1e6:.2f} MB")



if __name__ == "__main__":
    main()
