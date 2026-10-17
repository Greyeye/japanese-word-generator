-- Migration: jlpt_match_embedding_dim
-- Version:   20261018000000
--
-- match_sentence() was declared with p_embedding vector(768). Migration
-- 20261017042247 widened the COLUMN to vector(1024) but left the function
-- argument alone, so on a database rebuilt from these migrations the function
-- would reject the very embeddings the column stores. The live project's
-- argument is already unconstrained (verified: it accepts a 1024-dim vector),
-- so declaring it unconstrained converges the repo onto the live schema instead
-- of diverging from it.
--
-- The parameter is only ever compared against sentences.embedding, so leaving its
-- dimension open is correct: it lets the column's dimension change again without
-- another function migration.

create or replace function match_sentence(
  p_norm_hash text,
  p_lemmas text[],
  p_form text default null,
  p_embedding vector default null,
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
