"""
Shared configuration for all Airflow DAGs - CLOUD + LOCAL HYBRID VERSION
Supports both local Docker databases and cloud services (Neon, Neo4j Aura, Qdrant Cloud)
"""
from datetime import timedelta
import os
from pathlib import Path
from urllib.parse import urlparse

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
# DATABASE CONNECTIONS (Cloud-ready with local fallback)
# ============================================================================

# PostgreSQL - Supports both Neon/Supabase (cloud) and local Docker
POSTGRES_URL = os.getenv('POSTGRES_URL')

if POSTGRES_URL:
    # Parse cloud connection URL
    parsed = urlparse(POSTGRES_URL)
    POSTGRES_HOST = parsed.hostname
    POSTGRES_PORT = parsed.port or 5432
    POSTGRES_DB = parsed.path.lstrip('/')
    POSTGRES_USER = parsed.username
    POSTGRES_PASSWORD = parsed.password
else:
    # Local Docker fallback
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'equity_research')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')
    POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Neo4j - Supports both Neo4j Aura (cloud) and local Docker
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j_password')

# Detect if using cloud Neo4j Aura (neo4j+s:// protocol)
IS_NEO4J_CLOUD = NEO4J_URI.startswith('neo4j+s://')

# Qdrant - Supports both Qdrant Cloud and local Docker
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not QDRANT_URL:
    # Local Docker fallback
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
    QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
    QDRANT_URL = f'http://{QDRANT_HOST}:{QDRANT_PORT}'
    QDRANT_API_KEY = None  # Local Qdrant doesn't need API key
else:
    # Parse cloud URL
    parsed = urlparse(QDRANT_URL)
    QDRANT_HOST = parsed.hostname
    QDRANT_PORT = parsed.port or 6333

# Detect if using cloud Qdrant
IS_QDRANT_CLOUD = QDRANT_URL and 'cloud.qdrant.io' in QDRANT_URL

# Ollama - Support both local and remote instances
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')

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

# ============================================================================
# DEPLOYMENT INFO (for debugging)
# ============================================================================
DEPLOYMENT_MODE = "cloud" if (IS_NEO4J_CLOUD or IS_QDRANT_CLOUD) else "local"

# Print configuration on import (useful for debugging)
if __name__ != "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded: {DEPLOYMENT_MODE.upper()} mode")
    logger.info(f"  Postgres: {POSTGRES_HOST}:{POSTGRES_PORT}")
    logger.info(f"  Neo4j: {NEO4J_URI} (cloud={IS_NEO4J_CLOUD})")
    logger.info(f"  Qdrant: {QDRANT_URL} (cloud={IS_QDRANT_CLOUD})")
    logger.info(f"  Ollama: {OLLAMA_URL}")
