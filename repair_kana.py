"""
repair_kana.py -- rewrite kana spellings of N5 words to their kanji form, in place.

  python repair_kana.py --revert    # undo the previous (buggy) pass, restore --before
  python repair_kana.py --dry       # show what would change, touch nothing
  python repair_kana.py             # apply

WHY A TOKEN-BOUNDARY GUARD IS REQUIRED
The first attempt replaced every substring occurrence, and substring matching
crosses word boundaries:
    晩御飯はこれだけです。   ->  晩御飯箱れだけです。   (は particle + こ of これ)
    交番はここでした。       ->  交番箱こでした。      (は particle + こ of ここ)
So a match is only accepted when it starts AND ends exactly on a token boundary:
    はこ in はこに食べ物  -> tokens は|こ|に, はこ spans は+こ, boundaries at 0 and 2  ACCEPT
    はこ in はこれだけです -> tokens は|これ|..., はこ would end mid-token          REJECT
"""

import json, os, re, sys, time, hashlib, urllib.request
import psycopg
from janome.tokenizer import Tokenizer

OUT = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

ENV = r"C:\Users\greyeye\AppData\Local\hermes\.env"
_src = open(os.path.join(OUT, "generate.py"), encoding="utf-8").read()
TOKEN = re.search(r'TOKEN\s*=\s*"([^"]+)"', _src).group(1)
EMB_URL, EMB_MODEL = "http://192.168.0.12:8889/v1/embeddings", "Qwen/Qwen3-Embedding-0.6B-GGUF"

tok = Tokenizer()
CONTENT = ("名詞", "動詞", "形容詞", "副詞")
LIGHT = {"為る", "有る", "居る", "成る", "する", "ある", "いる", "なる"}
PARTICLES = set("はがをにでとものへやかねよなぞまでからより")
norm = lambda s: re.sub(r'[。．！？、…\s「」『』（）・]', '', s)
sha = lambda s: hashlib.sha256(norm(s).encode("utf-8")).hexdigest()


def content_set(s):
    out = set()
    for t in tok.tokenize(s):
        p = t.part_of_speech.split(",")
        if p[0] not in CONTENT:
            continue
        if p[0] == "名詞" and p[1] in ("数", "固有名詞"):
            continue
        l = t.base_form if t.base_form not in ("*", "") else t.surface
        if l in LIGHT:
            continue
        out.add(l)
    return sorted(out)


def form_signature(s):
    if "ませんでした" in s:             return "polite_past_negative"
    if "ません" in s:                  return "polite_negative"
    if "ましょう" in s:                return "polite_volitional"
    if "ました" in s or "でした" in s:  return "polite_past"
    if "ください" in s:                return "polite_request"
    if "です" in s or "ます" in s:     return "polite_nonpast"
    if "だった" in s:                  return "plain_past"
    return "plain"


def tokens_of(s):
    r = []
    for i, t in enumerate(tok.tokenize(s)):
        p = t.part_of_speech.split(",")
        r.append((i, t.surface, t.base_form if t.base_form not in ("*", "") else t.surface,
                  t.reading if t.reading not in ("*", "") else None, p[0], ",".join(p[1:3])))
    return r


def boundaries(s):
    """Character offsets where a token starts or ends."""
    b, pos = {0}, 0
    for t in tok.tokenize(s):
        pos += len(t.surface)
        b.add(pos)
    return b


def embed(texts):
    if not texts:
        return []
    req = urllib.request.Request(EMB_URL,
        data=json.dumps({"model": EMB_MODEL, "input": texts}).encode(),
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]


def rewrite(old, new):
    """Recompute every derived key for a changed sentence."""
    return (sha(new), content_set(new), form_signature(new),
            [t[1] for t in tokens_of(new) if t[1] in PARTICLES],
            len(tokens_of(new)), len(new), tokens_of(new))


url = re.search(r'^\s*SUPABASE_DB_URL\s*=\s*"?(.+?)"?\s*$',
                open(ENV, encoding="utf-8", errors="replace").read(), re.M).group(1)

fixes = json.load(open(os.path.join(RUNS, "kana_fix.json"), encoding="utf-8"))
fixes = [f for f in fixes if f["before"] != f["after"]]

# --- validate each proposed repair against token boundaries -------------------
# A repair is only safe when the matched kana starts AND ends on a token boundary
# in the original sentence. That is what separates the two cases:
#   はこ in はこに食べ物   -> tokens は|こ|に    -> span [0,2) is bounded   ACCEPT
#   はこ in はこれだけです -> tokens は|これ|... -> span [3,5) ends inside これ  REJECT
# The span must come from the audit, not from diffing the two strings: ほしい -> 欲しい
# differs in ONE character (ほ->欲) but the match span is the whole word [3,6), and
# both of those offsets are boundaries. Diffing gives (3,4), which is not.
def spans_ok(f):
    s, b = f["before"], boundaries(f["before"])
    spans = f.get("spans") or []
    for a, e, kana, kanji in spans:
        if a not in b or e not in b:
            return False
        if s[a:e] != kana:
            return False
    return bool(spans)


safe, unsafe = [], []
for f in fixes:
    (safe if spans_ok(f) else unsafe).append(f)

print(f"{len(fixes)} proposed repairs: {len(safe)} pass the token-boundary guard, "
      f"{len(unsafe)} rejected")
if unsafe:
    print("\n  REJECTED (match crossed a word boundary):")
    for f in unsafe[:10]:
        print(f"    {f['before']}  ->  {f['after']}")

if "--revert" in sys.argv:
    with psycopg.connect(url, connect_timeout=20) as conn, conn.cursor() as cur:
        n = 0
        for f in fixes:
            cur.execute("select id from sentences where norm_hash = %s", (sha(f["after"]),))
            row = cur.fetchone()
            if not row:
                continue
            sid = row[0]
            h, lem, form, part, tc, cc, toks = rewrite(f["after"], f["before"])
            notes = []
            cur.execute("""update sentences set japanese=%s, norm_hash=%s, content_lemmas=%s,
                form_signature=%s, particle_seq=%s, token_count=%s, char_count=%s, notes=%s
                where id=%s""", (f["before"], h, lem, form, part, tc, cc, notes, sid))
            cur.execute("delete from sentence_tokens where sentence_id=%s", (sid,))
            cur.executemany("""insert into sentence_tokens
                (sentence_id,position,surface,lemma,reading,pos,pos_detail)
                values (%s,%s,%s,%s,%s,%s,%s)""", [(sid,) + t for t in toks])
            n += 1
        conn.commit()
        print(f"\nreverted {n} rows to their original kana spelling")
        # re-embed the reverted rows so text and vector agree again
        cur.execute("select id, japanese from sentences where norm_hash = any(%s)",
                    ([sha(f["before"]) for f in fixes],))
        rows = cur.fetchall()
        if rows:
            vecs = embed([r[1] for r in rows])
            for (sid, ja), v in zip(rows, vecs):
                cur.execute("update sentences set embedding = %s::vector where id = %s",
                            ("[" + ",".join(f"{x:.6f}" for x in v) + "]", sid))
            conn.commit()
            print(f"re-embedded {len(rows)} reverted rows")
    sys.exit(0)

print()
if "--dry" in sys.argv:
    for f in safe:
        print(f"   {f['before']}  ->  {f['after']}")
    sys.exit(0)

print(f"embedding {len(safe)} repaired sentences ...")
vecs = embed([f["after"] for f in safe])

with psycopg.connect(url, connect_timeout=20) as conn, conn.cursor() as cur:
    done = 0
    for f, v in zip(safe, vecs):
        cur.execute("select id, notes from sentences where norm_hash = %s", (sha(f["before"]),))
        row = cur.fetchone()
        if not row:
            print(f"   not found: {f['before']}")
            continue
        sid, notes = row
        notes = [n for n in (notes or []) if n != "kana_repaired"] + ["kana_repaired"]
        h, lem, form, part, tc, cc, toks = rewrite(f["before"], f["after"])
        cur.execute("""update sentences set japanese=%s, norm_hash=%s, content_lemmas=%s,
            form_signature=%s, particle_seq=%s, token_count=%s, char_count=%s,
            embedding=%s::vector, notes=%s where id=%s""",
            (f["after"], h, lem, form, part, tc, cc,
             "[" + ",".join(f"{x:.6f}" for x in v) + "]", notes, sid))
        cur.execute("delete from sentence_tokens where sentence_id=%s", (sid,))
        cur.executemany("""insert into sentence_tokens
            (sentence_id,position,surface,lemma,reading,pos,pos_detail)
            values (%s,%s,%s,%s,%s,%s,%s)""", [(sid,) + t for t in toks])
        done += 1
    conn.commit()
    print(f"\nrepaired {done} sentences")

    cur.execute("select count(*) from sentences where notes @> array['kana_repaired']")
    print(f"rows marked kana_repaired: {cur.fetchone()[0]}")
    cur.execute("select japanese, content_lemmas from sentences where japanese like '箱%' limit 4")
    for ja, lem in cur.fetchall():
        print(f"   {ja}   lemmas={lem}")
    print("\n-- sanity: any sentence still containing a kana form we meant to fix? --")
    for probe in ("はこ", "ほしい", "つくえ"):
        cur.execute("select count(*) from sentences where japanese like %s and japanese not like %s",
                    (f"%{probe}%", "%箱%" if probe == "はこ" else "%" + {"ほしい": "欲しい", "つくえ": "机"}[probe] + "%"))
        print(f"   {probe}: {cur.fetchone()[0]} remaining")
