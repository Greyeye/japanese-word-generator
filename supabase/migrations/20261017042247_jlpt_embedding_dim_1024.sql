-- Migration: jlpt_embedding_dim_1024
-- Version:   20261017042247
--
-- Widens sentences.embedding from vector(768) to vector(1024) so the corpus can
-- use Qwen3-Embedding-0.6B (1024 dim, Apache-2.0) rather than a 768-dim model.
-- The HNSW index survives the rewrite; verified present afterwards.
--
alter table sentences alter column embedding type vector(1024);
