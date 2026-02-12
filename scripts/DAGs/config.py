"""
Shared configuration for all Airflow DAGs - LOCAL DATABASES VERSION
All databases run in Docker containers on localhost
"""
from datetime import timedelta
import os
from pathlib import Path

# ============================================================================
# PATHS (Inside Docker containers)
# ============================================================================
STORAGE_ROOT = Path(os.getenv('STORAGE_ROOT', '/opt/airflow/storage'))
DATA_ROOT = Path(os.getenv('DATA_ROOT', '/opt/airflow/data'))

# DuckDB storage path
DUCKDB_PATH = os.getenv('DUCKDB_PATH', '/opt/airflow/data/duckdb')

# Neo4j import path (for CSV imports)
NEO4J_IMPORT_PATH = DATA_ROOT / "neo4j_imports"

# ============================================================================
# API KEYS
# ============================================================================
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# API Endpoints
EODHD_BASE_URL = "https://eodhistoricaldata.com/api"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# ============================================================================
# LOCAL DATABASE CONNECTIONS (Docker)
# ============================================================================

# PostgreSQL (Local Docker container)
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'equity_research')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')

# Neo4j (Local Docker container)
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j_password')

# Qdrant (Local Docker container - no API key needed)
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_API_KEY = None  # Local Qdrant doesn't need API key
QDRANT_URL = os.getenv('QDRANT_URL', f'http://{QDRANT_HOST}:{QDRANT_PORT}')

# DuckDB (Local files - no Motherduck)
DUCKDB_PRICES_DB = f"{DUCKDB_PATH}/prices.db"
DUCKDB_MACRO_DB = f"{DUCKDB_PATH}/macro.db"
DUCKDB_FUNDAMENTALS_DB = f"{DUCKDB_PATH}/fundamentals.db"

# Redis - NOT USED (removed from architecture)
REDIS_HOST = None
REDIS_PORT = None
REDIS_PASSWORD = None
REDIS_URL = None

# ============================================================================
# QDRANT COLLECTIONS
# ============================================================================
QDRANT_COLLECTIONS = {
    "business_analyst_10k": "business_analyst_10k_filings",
    "business_analyst_10q": "business_analyst_10q_filings",
    "business_analyst_transcripts": "business_analyst_transcripts",
    "insider_sentiment_transcripts": "insider_sentiment_transcripts",
    "macro_central_bank_comms": "macro_central_bank_communications",
}

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# ============================================================================
# TICKERS
# ============================================================================
# Default watchlist (can be overridden by loading from file)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "JPM", "JNJ", "V", "PG", "MA", "HD", "UNH", "DIS", "BAC", "XOM",
    "ADBE", "NFLX", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM"
]

# ============================================================================
# DAG DEFAULT ARGUMENTS
# ============================================================================
DEFAULT_ARGS = {
    "owner": "equity_research_system",
    "depends_on_past": False,
    "email": [os.getenv("ALERT_EMAIL", "admin@example.com")],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}

# ============================================================================
# TASK EXECUTION SETTINGS
# ============================================================================
MAX_ACTIVE_RUNS = 3
API_TIMEOUT_SECONDS = 30
MAX_PRICE_GAP_PCT = 0.20  # 20% circuit breaker threshold
MIN_TICKER_COVERAGE_PCT = 0.90  # Require 90% ticker coverage

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
