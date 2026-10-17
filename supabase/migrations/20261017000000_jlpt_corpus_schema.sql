-- Migration: jlpt_corpus_schema
-- Version:   20261017000000
--
-- The first migration for the JLPT sentence corpus. Applied to project
--   wilmybwwljzbozotopbhx
-- and tracked by the Supabase GitHub integration, which reads
--   supabase/migrations/*.sql
--
-- The original standalone file was H:\jlpt\gen\schema.sql.
--
-- ============================================================================
-- JLPT sentence corpus -- Supabase/Postgres schema
--
-- Design note: this corpus is a MATCHING TARGET, not a text collection. The user
-- builds a phrase and we have to decide what it is. So the match keys are
-- materialised as indexed columns at load time -- never computed at query time.
--
-- Match tiers, cheapest first:
--   1. norm_hash        exact, after normalising punctuation/whitespace/width
--   2. content_lemmas + form_signature    same words, same grammar
--   3. content_lemmas   (GIN containment) same words, different grammar
--   4. embedding        (HNSW)            paraphrase, different words
--   5. no match         -> rule checker
--
-- Embeddings are tier 4, NOT tier 1: they are insensitive to particles and
-- grammatical form, so 「食べます」and「食べません」sit almost on top of each other
-- with opposite meanings. The grammatical distinction is the one thing a grammar
-- game cannot blur, so exact keys lead and vectors follow.
-- ============================================================================

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- corpus
-- ---------------------------------------------------------------------------
create table if not exists sentences (
  id                bigserial primary key,
  japanese          text not null,
  romaji            text,
  english           text,

  -- match keys (materialised)
  norm_hash         text not null,          -- sha256 of the normalised string
  content_lemmas    text[] not null default '{}',   -- sorted content-word lemmas
  form_signature    text,                   -- 'polite_nonpast' | 'polite_past' | 'plain' | ...
  particle_seq      text[] default '{}',
  token_count       int,
  char_count        int,

  -- retrieval
  embedding         vector(768),

  -- provenance
  grammar_point_id  text,
  jlpt_level        text,                   -- assigned post-hoc, may be null
  register          text,
  source            text not null default 'generated',
  model             text,
  job_id            text,
  status            text not null default 'validated',
  notes             text[],                 -- non-blocking observations from validate.py

  created_at        timestamptz not null default now()
);

create index if not exists sentences_norm_hash_idx   on sentences (norm_hash);
create index if not exists sentences_content_gin     on sentences using gin (content_lemmas);
create index if not exists sentences_form_idx        on sentences (form_signature);
create index if not exists sentences_grammar_idx     on sentences (grammar_point_id);
create index if not exists sentences_level_idx       on sentences (jlpt_level);
create index if not exists sentences_embedding_hnsw  on sentences
  using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- tokens -- mode 1 (word order) and mode 2 (cloze) both derive from these
-- ---------------------------------------------------------------------------
create table if not exists sentence_tokens (
  sentence_id  bigint not null references sentences(id) on delete cascade,
  position     int    not null,
  surface      text   not null,
  lemma        text,
  reading      text,
  pos          text,
  pos_detail   text,
  primary key (sentence_id, position)
);

create index if not exists tokens_lemma_idx on sentence_tokens (lemma);

-- ---------------------------------------------------------------------------
-- pools / ledger -- what the generator drew, so coverage is auditable
-- ---------------------------------------------------------------------------
create table if not exists generation_targets (
  id                bigserial primary key,
  grammar_point_id  text not null,
  job_id            text not null unique,
  pattern           text,
  meaning_en        text,
  formality         text,
  frame             text,                   -- structural requirement, when a word pool cannot express it
  check_regex       text,
  pos_need          text,
  target_count      int default 12
);

create table if not exists generation_attempts (
  id                bigserial primary key,
  job_id            text not null references generation_targets(job_id) on delete cascade,
  attempt           int  not null,
  model             text,
  temperature       numeric(3,2),
  required          text[],                 -- pool words this call had to use
  covered           text[],                 -- which of them it actually used
  pattern_ok        boolean,
  returned          int,
  sec               numeric(6,2),
  usage             jsonb,
  created_at        timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- validation -- one row per (sentence, check), so a rule change can be
-- re-measured without regenerating anything
-- ---------------------------------------------------------------------------
create table if not exists validation_results (
  sentence_id  bigint not null references sentences(id) on delete cascade,
  check_name   text   not null,
  severity     text   not null check (severity in ('hard','soft','note')),
  detail       text,
  ruleset      text,                        -- hash/version of validate.py's rules
  created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- tiered match function: given a user-built phrase, return what we know
--
-- Takes keys the CALLER already computed, rather than hashing in SQL. The
-- normalisation (strip 。．！？、… whitespace 「」『』（）・) lives in exactly one
-- place -- the loader, sharing content_set()/norm() with validate.py -- so the
-- stored keys and the query keys can never drift apart. Server-side hashing
-- would also need pgcrypto on the right search_path, which is a needless
-- coupling for a string comparison.
-- ---------------------------------------------------------------------------
create or replace function match_sentence(
  p_norm_hash     text,                     -- sha256 of the normalised phrase
  p_lemmas        text[],
  p_form          text default null,
  p_embedding     vector(768) default null,
  p_limit         int default 10
)
returns table (
  id          bigint,
  japanese    text,
  tier        int,
  score       numeric,
  explanation text
)
language sql stable as $$
  with exact as (
    select s.id, s.japanese, 1 as tier, 1.0::numeric as score,
           'identical sentence'::text as explanation
    from sentences s
    where s.norm_hash = p_norm_hash
  ),
  same_form as (
    select s.id, s.japanese, 2, 1.0,
           'same words, same grammar'::text
    from sentences s
    where s.content_lemmas = p_lemmas
      and (p_form is null or s.form_signature = p_form)
      and not exists (select 1 from exact e where e.id = s.id)
  ),
  same_words as (
    select s.id, s.japanese, 3,
           (cardinality(array(select unnest(s.content_lemmas)
                              intersect select unnest(p_lemmas)))::numeric
            / greatest(cardinality(array(select unnest(s.content_lemmas)
                              union select unnest(p_lemmas))), 1))::numeric,
           'same words, different grammar'::text
    from sentences s
    where s.content_lemmas && p_lemmas
  ),
  semantic as (
    select s.id, s.japanese, 4,
           (1 - (s.embedding <=> p_embedding))::numeric,
           'similar meaning, different words'::text
    from sentences s
    where p_embedding is not null and s.embedding is not null
    order by s.embedding <=> p_embedding
    limit p_limit
  )
  select * from (
    select * from exact
    union all select * from same_form   where not exists (select 1 from exact)
    union all select * from same_words  where not exists (select 1 from exact)
    union all select * from semantic    where not exists (select 1 from exact)
  ) r
  order by tier, score desc
  limit p_limit;
$$;
