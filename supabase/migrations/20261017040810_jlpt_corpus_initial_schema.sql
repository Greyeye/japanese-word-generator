-- Migration: jlpt_corpus_initial_schema
-- Version:   20261017040810
--
-- Mirrors the migration already recorded in the project's
-- supabase_migrations.schema_migrations table (idempotency_key
-- 'jlpt-corpus-schema-v1'), so the Supabase GitHub integration sees the repo
-- and the project agree and applies nothing.
--
-- Five tables (sentences, sentence_tokens, generation_targets,
-- generation_attempts, validation_results) and the five-tier match_sentence().
-- Embedding starts at 768 here and is widened by the next migration.
--
create extension if not exists vector;

create table if not exists sentences (
  id bigserial primary key,
  japanese text not null,
  romaji text,
  english text,
  norm_hash text not null,
  content_lemmas text[] not null default '{}',
  form_signature text,
  particle_seq text[] default '{}',
  token_count int,
  char_count int,
  embedding vector(768),
  grammar_point_id text,
  jlpt_level text,
  register text,
  source text not null default 'generated',
  model text,
  job_id text,
  status text not null default 'validated',
  notes text[],
  created_at timestamptz not null default now()
);

create index if not exists sentences_norm_hash_idx on sentences (norm_hash);
create index if not exists sentences_content_gin on sentences using gin (content_lemmas);
create index if not exists sentences_form_idx on sentences (form_signature);
create index if not exists sentences_grammar_idx on sentences (grammar_point_id);
create index if not exists sentences_level_idx on sentences (jlpt_level);
create index if not exists sentences_embedding_hnsw on sentences using hnsw (embedding vector_cosine_ops);

create table if not exists sentence_tokens (
  sentence_id bigint not null references sentences(id) on delete cascade,
  position int not null,
  surface text not null,
  lemma text,
  reading text,
  pos text,
  pos_detail text,
  primary key (sentence_id, position)
);
create index if not exists tokens_lemma_idx on sentence_tokens (lemma);

create table if not exists generation_targets (
  id bigserial primary key,
  grammar_point_id text not null,
  job_id text not null unique,
  pattern text,
  meaning_en text,
  formality text,
  frame text,
  check_regex text,
  pos_need text,
  target_count int default 12
);

create table if not exists generation_attempts (
  id bigserial primary key,
  job_id text not null references generation_targets(job_id) on delete cascade,
  attempt int not null,
  model text,
  temperature numeric(3,2),
  required text[],
  covered text[],
  pattern_ok boolean,
  returned int,
  sec numeric(6,2),
  usage jsonb,
  created_at timestamptz not null default now()
);

create table if not exists validation_results (
  sentence_id bigint not null references sentences(id) on delete cascade,
  check_name text not null,
  severity text not null check (severity in ('hard','soft','note')),
  detail text,
  ruleset text,
  created_at timestamptz not null default now()
);

create or replace function match_sentence(
  p_norm_hash text,
  p_lemmas text[],
  p_form text default null,
  p_embedding vector(768) default null,
  p_limit int default 10
)
returns table (id bigint, japanese text, tier int, score numeric, explanation text)
language sql stable as $$
  with exact as (
    select s.id, s.japanese, 1 as tier, 1.0::numeric as score, 'identical sentence'::text as explanation
    from sentences s where s.norm_hash = p_norm_hash
  ),
  same_form as (
    select s.id, s.japanese, 2, 1.0, 'same words, same grammar'::text
    from sentences s
    where s.content_lemmas = p_lemmas
      and (p_form is null or s.form_signature = p_form)
      and not exists (select 1 from exact e where e.id = s.id)
  ),
  same_words as (
    select s.id, s.japanese, 3,
           (cardinality(array(select unnest(s.content_lemmas) intersect select unnest(p_lemmas)))::numeric
            / greatest(cardinality(array(select unnest(s.content_lemmas) union select unnest(p_lemmas))), 1))::numeric,
           'same words, different grammar'::text
    from sentences s where s.content_lemmas && p_lemmas
  ),
  semantic as (
    select s.id, s.japanese, 4, (1 - (s.embedding <=> p_embedding))::numeric,
           'similar meaning, different words'::text
    from sentences s
    where p_embedding is not null and s.embedding is not null
    order by s.embedding <=> p_embedding limit p_limit
  )
  select * from (
    select * from exact
    union all select * from same_form where not exists (select 1 from exact)
    union all select * from same_words where not exists (select 1 from exact)
    union all select * from semantic where not exists (select 1 from exact)
  ) r
  order by tier, score desc limit p_limit;
$$;
