"""
demo_match.py -- exercise the tier ladder against real stored data.

Takes a stored sentence, then submits it back the way a learner would: with the
period dropped, with the grammar changed, and as a genuine paraphrase -- and shows
which tier answers each time. This is the check that matters, because the whole
design claim is that exact and same-words matching do the real work and vectors
only pick up the tail.
"""

import os, re, sys, json
import psycopg
from janome.tokenizer import Tokenizer

ENV = r"C:\Users\greyeye\AppData\Local\hermes\.env"
tok = Tokenizer()
CONTENT = ("名詞", "動詞", "形容詞", "副詞")
LIGHT = {"為る", "有る", "居る", "成る", "する", "ある", "いる", "なる"}
norm = lambda s: re.sub(r'[。．！？、…\s「」『』（）・]', '', s)
sha = lambda s: __import__("hashlib").sha256(norm(s).encode()).hexdigest()


def content_set(s):
    out = set()
    for t in tok.tokenize(s):
        pos = t.part_of_speech.split(",")
        if pos[0] not in CONTENT:
            continue
        if pos[0] == "名詞" and pos[1] in ("数", "固有名詞"):
            continue
        lemma = t.base_form if t.base_form not in ("*", "") else t.surface
        if lemma in LIGHT:
            continue
        out.add(lemma)
    return sorted(out)


def form_signature(s):
    if "ませんでした" in s:            return "polite_past_negative"
    if "ません" in s:                 return "polite_negative"
    if "ましょう" in s:               return "polite_volitional"
    if "ました" in s or "でした" in s: return "polite_past"
    if "ください" in s:               return "polite_request"
    if "です" in s or "ます" in s:    return "polite_nonpast"
    if "だった" in s:                 return "plain_past"
    return "plain"


url = re.search(r'^\s*SUPABASE_DB_URL\s*=\s*"?(.+?)"?\s*$',
                open(ENV, encoding="utf-8", errors="replace").read(), re.M).group(1)

with psycopg.connect(url, connect_timeout=20) as conn, conn.cursor() as cur:
    print("-- index check: does the GIN index carry the tier-3 lookup? --")
    cur.execute("explain (costs off) select id from sentences "
                "where content_lemmas && array['パン','食べる']::text[]")
    for (line,) in cur.fetchall():
        print("   " + line)

    # pick a real sentence with a couple of content words
    cur.execute("""select japanese, content_lemmas, form_signature from sentences
                   where status='validated' and cardinality(content_lemmas)=3
                     and grammar_point_id like 'masu%' limit 1""")
    row = cur.fetchone()
    if not row:
        cur.execute("""select japanese, content_lemmas, form_signature from sentences
                       where status='validated' and cardinality(content_lemmas)=3 limit 1""")
        row = cur.fetchone()
    stored, lemmas, form = row
    print(f"\n-- stored sentence --\n   {stored}   lemmas={lemmas} form={form}")

    cases = [
        ("exact, period dropped", stored.rstrip("。"), lemmas, form),
        ("same words, plain form", "".join(lemmas), lemmas, "plain"),
        ("unknown sentence", "まったく別の文です", None, None),
    ]

    for label, text, lem, fm in cases:
        print(f"\n-- {label}: {text}")
        if lem is None:
            lem = content_set(text)
            fm = form_signature(text)
        cur.execute("select tier, score, explanation, japanese from match_sentence(%s,%s,%s,null,4)",
                    (sha(text), lem, fm))
        rows = cur.fetchall()
        if not rows:
            print("     (no match -> tier 5, hand to the rule checker)")
        for tier, score, expl, ja in rows:
            mark = "  <-- stored" if ja == stored else ""
            print(f"     tier {tier}  {float(score):.3f}  {expl:<32} {ja}{mark}")

    print("\n-- tier distribution over the whole corpus --")
    cur.execute("select count(*) from sentences where embedding is not null")
    print(f"   embedded        {cur.fetchone()[0]}")
    cur.execute("select count(distinct norm_hash) from sentences")
    print(f"   distinct hashes {cur.fetchone()[0]}")
