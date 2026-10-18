"""
push.py -- execute the generated SQL chunks against Supabase over a direct
Postgres connection.

Why a direct connection and not the MCP/REST path: the payload is ~12 MB, almost
all of it 1024-dim vectors. Pushing that through an agent tool call would drag the
whole corpus through the conversation; a socket between disk and database keeps it
out. It is also re-runnable and scriptable as the corpus grows.

The connection string is read from the Hermes .env (SUPABASE_DB_URL) so the
password never appears in the conversation or in this file.

  python push.py --check     # connect and report, change nothing
  python push.py             # execute every chunk
  python push.py --verify    # run the end-to-end match test only
"""

import os, re, sys, glob, time
import psycopg

OUT  = r"H:\jlpt\gen"
# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

LOAD = os.path.join(RUNS, "load")
ENV  = r"C:\Users\greyeye\AppData\Local\hermes\.env"


def db_url():
    v = os.environ.get("SUPABASE_DB_URL")
    if v:
        return v.strip()
    if os.path.exists(ENV):
        txt = open(ENV, encoding="utf-8", errors="replace").read()
        m = re.search(r'^\s*SUPABASE_DB_URL\s*=\s*"?(.+?)"?\s*$', txt, re.M)
        if m:
            return m.group(1).strip()
    return None


def chunks():
    """Order matters: sentences before the tokens that reference them."""
    s = sorted(glob.glob(os.path.join(LOAD, "sentences_*.sql")))
    t = sorted(glob.glob(os.path.join(LOAD, "tokens_*.sql")))
    f = sorted(glob.glob(os.path.join(LOAD, "99_*.sql")))
    return s + t + f


def verify(cur):
    print("\n-- loaded state --")
    for tbl in ("sentences", "sentence_tokens"):
        cur.execute(f"select count(*) from {tbl}")
        print(f"   {tbl:<18} {cur.fetchone()[0]:>6} rows")
    cur.execute("select count(*) from sentences where embedding is not null")
    print(f"   with embeddings    {cur.fetchone()[0]:>6}")
    cur.execute("select status, count(*) from sentences group by status order by 2 desc")
    print("   status:", dict(cur.fetchall()))
    cur.execute("select form_signature, count(*) from sentences "
                "group by form_signature order by 2 desc limit 8")
    print("   forms:", dict(cur.fetchall()))

    print("\n-- end-to-end: match_sentence --")
    # take a real stored sentence, then perturb it the way a learner would
    cur.execute("select id, japanese, norm_hash, content_lemmas, form_signature "
                "from sentences where status='validated' "
                "and cardinality(content_lemmas) >= 2 limit 1")
    row = cur.fetchone()
    if row is None:
        # An empty corpus is a legitimate state (it is the state after a fresh
        # migration), and the end-to-end probe has nothing to probe. Crashing here
        # with "cannot unpack non-iterable NoneType" made a healthy empty database
        # look broken -- which is exactly the wrong signal for a --check command.
        print("   no validated sentences to probe (corpus empty)")
        print("   -> the connection and schema are fine; load something and re-run")
        return
    sid, ja, nh, lem, form = row
    print(f"   stored : {ja}   (lemmas={lem}, form={form})")

    cur.execute("""
        select tier, score, explanation, japanese
        from match_sentence(%s, %s, %s, null, 5)
    """, (nh, lem, form))
    for tier, score, expl, ja2 in cur.fetchall():
        print(f"     tier {tier}  {float(score):.3f}  {expl:<34} {ja2}")

    # now the same words, wrong grammar -> should land on tier 3
    cur.execute("""
        select tier, score, explanation, japanese
        from match_sentence(%s, %s, %s, null, 5)
    """, ("", lem, "plain"))
    print("   same words, different form:")
    for tier, score, expl, ja2 in cur.fetchall():
        print(f"     tier {tier}  {float(score):.3f}  {expl:<34} {ja2}")


def main():
    # push.py performed the database write at module level, so merely
    # importing it (to reuse db_url) executed the load. Third file with
    # this bug; the guard is now uniform across the pipeline.
    url = db_url()
    if not url:
        print("SUPABASE_DB_URL not found.")
        print(f"Add it to {ENV} as:")
        print("   SUPABASE_DB_URL=postgresql://postgres.wilmybwljzbozotopbhx:<password>@<host>:5432/postgres")
        print("(Dashboard -> Connect -> Session pooler. Secrets live in .env, never in config.yaml.)")
        sys.exit(1)

    redacted = re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", url)
    print(f"connecting to {redacted}")

    try:
        conn = psycopg.connect(url, connect_timeout=20)
    except Exception as e:
        print(f"connection failed: {str(e)[:300]}")
        sys.exit(1)

    with conn:
        with conn.cursor() as cur:
            cur.execute("select current_database(), version()")
            db, ver = cur.fetchone()
            print(f"   connected to {db} -- {ver.split(',')[0]}")

            if "--verify" in sys.argv:
                verify(cur)
                sys.exit(0)

            if "--check" in sys.argv:
                verify(cur)
                sys.exit(0)

            files = chunks()
            print(f"\napplying {len(files)} chunks")
            for f in files:
                sql = open(f, encoding="utf-8").read()
                t0 = time.time()
                cur.execute(sql)
                conn.commit()
                print(f"   {os.path.basename(f):<22} {len(sql)/1e6:5.2f} MB  {time.time()-t0:6.1f}s",
                      flush=True)

            verify(cur)



if __name__ == "__main__":
    main()
