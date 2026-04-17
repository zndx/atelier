\restrict dbmate

-- Dumped from database version 16.13
-- Dumped by pg_dump version 16.13

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: agents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agents (
    id text NOT NULL,
    name text,
    description text,
    role text,
    tool_ids text
);


--
-- Name: classification_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.classification_runs (
    id text NOT NULL,
    fsm_run_id text,
    table_name text NOT NULL,
    column_name text NOT NULL,
    predicted_code text,
    predicted_label text,
    confidence real,
    belief real,
    plausibility real,
    conflict real,
    evidence_sources text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: data_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.data_sources (
    id text NOT NULL,
    source_type text NOT NULL,
    source_uri text DEFAULT ''::text NOT NULL,
    display_name text NOT NULL,
    vocabulary_mode text DEFAULT 'universal'::text NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    metadata text,
    is_archived boolean DEFAULT false NOT NULL,
    vocab_uri text DEFAULT ''::text NOT NULL
);


--
-- Name: datasets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasets (
    id text NOT NULL,
    name text,
    parquet_path text,
    description text,
    row_count bigint,
    source_id text,
    version_number integer DEFAULT 1 NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    summary text,
    fsm_run_id text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_archived boolean DEFAULT false NOT NULL
);


--
-- Name: fsm_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fsm_runs (
    id text NOT NULL,
    state text DEFAULT 'IDLE'::text NOT NULL,
    started_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    config text,
    progress text,
    error text,
    result_path text,
    source_id text
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: vocabularies; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.vocabularies (
    id text NOT NULL,
    source text NOT NULL,
    loaded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    category_count integer,
    categories text
);


--
-- Name: agents agents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agents
    ADD CONSTRAINT agents_pkey PRIMARY KEY (id);


--
-- Name: classification_runs classification_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classification_runs
    ADD CONSTRAINT classification_runs_pkey PRIMARY KEY (id);


--
-- Name: data_sources data_sources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_sources
    ADD CONSTRAINT data_sources_pkey PRIMARY KEY (id);


--
-- Name: datasets datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);


--
-- Name: fsm_runs fsm_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fsm_runs
    ADD CONSTRAINT fsm_runs_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: vocabularies vocabularies_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vocabularies
    ADD CONSTRAINT vocabularies_pkey PRIMARY KEY (id);


--
-- Name: idx_data_sources_not_archived; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_data_sources_not_archived ON public.data_sources USING btree (is_archived) WHERE (is_archived = false);


--
-- Name: idx_datasets_not_archived; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasets_not_archived ON public.datasets USING btree (is_archived) WHERE (is_archived = false);


--
-- Name: idx_datasets_source_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasets_source_version ON public.datasets USING btree (source_id, version_number DESC);


--
-- Name: classification_runs classification_runs_fsm_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classification_runs
    ADD CONSTRAINT classification_runs_fsm_run_id_fkey FOREIGN KEY (fsm_run_id) REFERENCES public.fsm_runs(id);


--
-- Name: datasets datasets_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.data_sources(id);


--
-- Name: fsm_runs fsm_runs_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fsm_runs
    ADD CONSTRAINT fsm_runs_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.data_sources(id);


--
-- PostgreSQL database dump complete
--

\unrestrict dbmate


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20260406000000'),
    ('20260406000000_initial_schema'),
    ('20260412000000'),
    ('20260412000000_classification_schema'),
    ('20260413000000'),
    ('20260413000000_seed_keystone_agents'),
    ('20260415000000'),
    ('20260415000000_data_sources_and_versions'),
    ('20260416000000'),
    ('20260416000000_archive_flag'),
    ('20260416100000'),
    ('20260416100000_vocab_uri'),
    ('20260417000000');
