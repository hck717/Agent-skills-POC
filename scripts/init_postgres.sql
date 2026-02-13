-- ============================================================================
-- EQUITY RESEARCH DATABASE INITIALIZATION
-- ============================================================================
-- This script runs automatically on first Docker startup via
-- /docker-entrypoint-initdb.d mount in docker-compose.yml
--
-- Creates:
-- 1. equity_research database (separate from airflow metadata DB)
-- 2. TimescaleDB extension for time-series data
-- 3. Schemas for each agent
-- 4. Helper extensions (uuid, pg_trgm)
-- ============================================================================

-- Create equity_research database if it doesn't exist
SELECT 'CREATE DATABASE equity_research'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'equity_research')\gexec

-- Connect to equity_research database
\c equity_research

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- TimescaleDB (for Insider & Sentiment Agent time-series hypertables)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Fuzzy text search (for GAAP label matching in Quantitative Agent)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Full-text search
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ============================================================================
-- SCHEMAS
-- ============================================================================

-- Quantitative Fundamental Agent
CREATE SCHEMA IF NOT EXISTS quantitative;
COMMENT ON SCHEMA quantitative IS 'Normalized financial statements, forensic scores, factor calculations';

-- Insider & Sentiment Agent
CREATE SCHEMA IF NOT EXISTS insider;
COMMENT ON SCHEMA insider IS 'Form 4 insider transactions, sentiment scores (TimescaleDB hypertables)';

-- Shared metadata across all agents
CREATE SCHEMA IF NOT EXISTS shared;
COMMENT ON SCHEMA shared IS 'Ticker watchlist, ingestion logs, data quality metrics';

-- ============================================================================
-- SHARED SCHEMA TABLES
-- ============================================================================

-- Ticker watchlist (master list)
CREATE TABLE IF NOT EXISTS shared.tickers (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(20),
    cik VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    added_date TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickers_sector ON shared.tickers(sector);
CREATE INDEX IF NOT EXISTS idx_tickers_active ON shared.tickers(is_active);

-- Data ingestion log (track DAG runs)
CREATE TABLE IF NOT EXISTS shared.ingestion_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dag_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(100) NOT NULL,
    ticker VARCHAR(10),
    execution_date TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) NOT NULL,  -- success, failed, running
    records_processed INTEGER,
    error_message TEXT,
    duration_seconds NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_dag ON shared.ingestion_log(dag_id, execution_date);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_ticker ON shared.ingestion_log(ticker);

-- ============================================================================
-- QUANTITATIVE SCHEMA TABLES
-- ============================================================================

-- Financial statements (normalized)
CREATE TABLE IF NOT EXISTS quantitative.financial_statements (
    statement_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    fiscal_date DATE NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER,  -- NULL for annual
    statement_type VARCHAR(20) NOT NULL,  -- income, balance, cashflow
    frequency VARCHAR(10) NOT NULL,  -- quarterly, annual
    line_item VARCHAR(255) NOT NULL,
    gaap_label VARCHAR(500),  -- Standardized US GAAP label
    value NUMERIC,
    source VARCHAR(20) NOT NULL,  -- eodhd, fmp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, fiscal_date, statement_type, frequency, line_item)
);

CREATE INDEX IF NOT EXISTS idx_fin_stmt_ticker_date ON quantitative.financial_statements(ticker, fiscal_date DESC);
CREATE INDEX IF NOT EXISTS idx_fin_stmt_gaap ON quantitative.financial_statements USING gin(gaap_label gin_trgm_ops);

-- Forensic scores (M-Score, Z-Score, F-Score)
CREATE TABLE IF NOT EXISTS quantitative.forensic_scores (
    score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    fiscal_date DATE NOT NULL,
    m_score NUMERIC,  -- Beneish M-Score
    z_score NUMERIC,  -- Altman Z-Score
    f_score INTEGER,  -- Piotroski F-Score (0-9)
    dso NUMERIC,  -- Days Sales Outstanding
    dso_growth_pct NUMERIC,
    channel_stuffing_flag VARCHAR(20),
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, fiscal_date)
);

CREATE INDEX IF NOT EXISTS idx_forensic_ticker_date ON quantitative.forensic_scores(ticker, fiscal_date DESC);

-- Dual-path verification audit trail
CREATE TABLE IF NOT EXISTS quantitative.audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    pandas_value NUMERIC,
    sql_value NUMERIC,
    divergence_pct NUMERIC,
    passed BOOLEAN,
    audited_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_failed ON quantitative.audit_trail(passed, audited_at DESC);

-- ============================================================================
-- INSIDER SCHEMA TABLES (TimescaleDB Hypertables)
-- ============================================================================

-- Insider transactions (Form 4)
CREATE TABLE IF NOT EXISTS insider.transactions (
    transaction_id UUID DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    transaction_date TIMESTAMPTZ NOT NULL,
    filed_date TIMESTAMPTZ NOT NULL,
    insider_name VARCHAR(255) NOT NULL,
    insider_title VARCHAR(255),
    transaction_type VARCHAR(20) NOT NULL,  -- Buy, Sell, Option Exercise, Gift, Award
    shares_traded BIGINT NOT NULL,
    price_per_share NUMERIC,
    value_usd NUMERIC,
    shares_owned_after BIGINT,
    source VARCHAR(20) NOT NULL,  -- eodhd, fmp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ticker, transaction_date, insider_name, transaction_id)
);

-- Convert to TimescaleDB hypertable (partitioned by transaction_date)
SELECT create_hypertable(
    'insider.transactions', 
    'transaction_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider.transactions(ticker, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_insider_type ON insider.transactions(transaction_type, transaction_date DESC);

-- Aggregated insider conviction scores (quarterly)
CREATE TABLE IF NOT EXISTS insider.conviction_scores (
    score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    quarter_end DATE NOT NULL,
    net_buy_value NUMERIC,  -- Total buys - sells in USD
    net_buy_shares BIGINT,
    num_buyers INTEGER,
    num_sellers INTEGER,
    conviction_score NUMERIC,  -- Custom score based on size + role
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, quarter_end)
);

CREATE INDEX IF NOT EXISTS idx_conviction_ticker ON insider.conviction_scores(ticker, quarter_end DESC);

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT ALL PRIVILEGES ON DATABASE equity_research TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantitative TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA insider TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA shared TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA quantitative TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA insider TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA shared TO airflow;

-- Grant usage on schemas
GRANT USAGE ON SCHEMA quantitative TO airflow;
GRANT USAGE ON SCHEMA insider TO airflow;
GRANT USAGE ON SCHEMA shared TO airflow;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA quantitative GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA insider GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA shared GRANT ALL ON TABLES TO airflow;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- List all schemas
SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('quantitative', 'insider', 'shared');

-- List all tables
SELECT schemaname, tablename FROM pg_tables WHERE schemaname IN ('quantitative', 'insider', 'shared') ORDER BY schemaname, tablename;

-- Verify TimescaleDB hypertable
SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'transactions';

-- ============================================================================
-- NOTES
-- ============================================================================
-- 1. This script is idempotent (safe to run multiple times)
-- 2. TimescaleDB hypertable automatically partitions insider.transactions by month
-- 3. GAAP label fuzzy matching uses pg_trgm extension (similarity search)
-- 4. All timestamps are TIMESTAMPTZ (timezone-aware) for global market support
-- 5. airflow user has full privileges on equity_research database
-- ============================================================================
