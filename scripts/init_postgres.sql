-- Initialize PostgreSQL databases for equity research system

-- Create equity_research database
CREATE DATABASE equity_research;

-- Connect to equity_research
\c equity_research;

-- Enable TimescaleDB extension (if available)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS news;
CREATE SCHEMA IF NOT EXISTS financials;
CREATE SCHEMA IF NOT EXISTS insider;
CREATE SCHEMA IF NOT EXISTS sentiment;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE equity_research TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA news TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA financials TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA insider TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA sentiment TO airflow;
GRANT USAGE ON SCHEMA public TO airflow; 

COMMIT;
