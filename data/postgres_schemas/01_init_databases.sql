-- ============================================================================
-- POSTGRESQL INITIALIZATION SCRIPT
-- ============================================================================
-- This script runs automatically when the PostgreSQL container starts
-- for the first time (via /docker-entrypoint-initdb.d/ mount)
--
-- Creates:
-- 1. equity_research database (separate from airflow metadata DB)
-- 2. Required PostgreSQL extensions (TimescaleDB, pg_trgm, uuid-ossp)
-- 3. Schemas for each agent (quantitative, insider, shared)
-- 4. Proper permissions
-- ============================================================================

-- Create equity_research database if it doesn't exist
-- Note: Cannot use IF NOT EXISTS in CREATE DATABASE, so we use a workaround
SELECT 'CREATE DATABASE equity_research'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'equity_research'
)\gexec

-- Connect to the new database
\c equity_research

-- ============================================================================
-- ENABLE EXTENSIONS
-- ============================================================================

-- TimescaleDB: For time-series hypertables (Insider & Sentiment Agent)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- pg_trgm: For fuzzy text search (Business Analyst Agent)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- uuid-ossp: For UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CREATE SCHEMAS
-- ============================================================================

-- Quantitative Fundamental Agent schema
CREATE SCHEMA IF NOT EXISTS quantitative;
COMMENT ON SCHEMA quantitative IS 'Financial statements, forensic scores, normalized taxonomy';

-- Insider & Sentiment Agent schema
CREATE SCHEMA IF NOT EXISTS insider;
COMMENT ON SCHEMA insider IS 'Insider transactions (TimescaleDB hypertables), institutional holdings';

-- Shared schema for cross-agent data
CREATE SCHEMA IF NOT EXISTS shared;
COMMENT ON SCHEMA shared IS 'Agent execution logs, metadata, audit trails';

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

-- Grant all privileges to the airflow user (used by all DAGs)
GRANT ALL PRIVILEGES ON DATABASE equity_research TO airflow;

-- Grant schema-level privileges
GRANT ALL PRIVILEGES ON SCHEMA quantitative TO airflow;
GRANT ALL PRIVILEGES ON SCHEMA insider TO airflow;
GRANT ALL PRIVILEGES ON SCHEMA shared TO airflow;

-- Grant default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA quantitative GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA insider GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA shared GRANT ALL ON TABLES TO airflow;

ALTER DEFAULT PRIVILEGES IN SCHEMA quantitative GRANT ALL ON SEQUENCES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA insider GRANT ALL ON SEQUENCES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA shared GRANT ALL ON SEQUENCES TO airflow;

-- ============================================================================
-- CREATE ROLES (Optional: for future multi-user access)
-- ============================================================================

-- Read-only role for reporting/dashboards
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_readonly') THEN
        CREATE ROLE agent_readonly;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE equity_research TO agent_readonly;
GRANT USAGE ON SCHEMA quantitative, insider, shared TO agent_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA quantitative, insider, shared TO agent_readonly;

-- Read-write role for DAG tasks
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_readwrite') THEN
        CREATE ROLE agent_readwrite;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE equity_research TO agent_readwrite;
GRANT USAGE ON SCHEMA quantitative, insider, shared TO agent_readwrite;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA quantitative, insider, shared TO agent_readwrite;

-- ============================================================================
-- LOGGING
-- ============================================================================

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE '===================================================================';
    RAISE NOTICE 'equity_research database initialized successfully';
    RAISE NOTICE 'Extensions enabled: timescaledb, pg_trgm, uuid-ossp';
    RAISE NOTICE 'Schemas created: quantitative, insider, shared';
    RAISE NOTICE 'Permissions granted to: airflow user';
    RAISE NOTICE '===================================================================';
END
$$;
