"""
Airflow DAGs Configuration
Centralized config for all 13 data pipelines
"""
from datetime import timedelta
import os

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

TEST_MODE = os.getenv('AIRFLOW_TEST_MODE', 'True').lower() == 'true'
MAX_ACTIVE_RUNS = 1

# ============================================================================
# DEFAULT AIRFLOW ARGS
# ============================================================================

DEFAULT_ARGS = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# UNIVERSE DEFINITION
# ============================================================================

DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    'META', 'TSLA', 'BRK.B', 'JNJ', 'V',
    'JPM', 'WMT', 'PG', 'MA', 'UNH',
    'DIS', 'NFLX', 'PYPL', 'ADBE', 'CRM'
]

# ============================================================================
# DATABASE CONNECTIONS
# ============================================================================

# Postgres
POSTGRES_URL = os.getenv(
    'POSTGRES_URL',
    'postgresql://postgres:postgres@localhost:5432/financial_data'
)

# Neo4j
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Qdrant
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')

# DuckDB
DUCKDB_PRICES_DB = os.getenv('DUCKDB_PRICES_DB', '/tmp/prices.duckdb')
DUCKDB_MACRO_DB = os.getenv('DUCKDB_MACRO_DB', '/tmp/macro.duckdb')

# ============================================================================
# QDRANT COLLECTIONS
# ============================================================================

QDRANT_COLLECTIONS = {
    'business_analyst_10k': 'ba_10k_propositions',
    'insider_sentiment_transcripts': 'insider_earnings_transcripts',
    'macro_central_bank_comms': 'macro_cb_statements',
}

# ============================================================================
# API KEYS AND ENDPOINTS
# ============================================================================

# EODHD
EODHD_API_KEY = os.getenv('EODHD_API_KEY', 'demo')
EODHD_BASE_URL = 'https://eodhd.com/api'

# Financial Modeling Prep
FMP_API_KEY = os.getenv('FMP_API_KEY', 'demo')
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

# OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# API Timeouts
API_TIMEOUT_SECONDS = 30

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================

MIN_TICKER_COVERAGE_PCT = 0.80  # 80% of tickers must have data
MAX_PRICE_GAP_PCT = 0.20  # 20% price gap = anomaly
LOOKBACK_DAYS = 7  # Default lookback for incremental loads
MAX_NEWS_ARTICLES = 50  # Max articles per ticker per fetch

# ============================================================================
# SCHEDULING
# ============================================================================

# Timezone: Hong Kong Time (HKT = UTC+8)
# Note: Airflow schedules use UTC internally
# Example: 6 PM HKT = 10 AM UTC

SCHEDULE_INTERVALS = {
    'daily_market_data': '0 23 * * 1-5',  # 11 PM UTC = 7 AM HKT (after US close)
    'daily_news': '0 */6 * * *',  # Every 6 hours
    'weekly_fundamentals': '0 18 * * 0',  # Sunday 6 PM UTC = Monday 2 AM HKT
    'monthly_insider': '0 19 1 * *',  # 1st of month, 7 PM UTC = 3 AM HKT
    'monthly_institutional': '0 19 15 * *',  # 15th of month
    'weekly_macro': '0 17 * * 1',  # Monday 5 PM UTC = 1 AM HKT
    'daily_quality': '0 15 * * *',  # 3 PM UTC = 11 PM HKT
    'monthly_retraining': '0 20 1 * *',  # 1st of month, 8 PM UTC = 4 AM HKT
    'neo4j_sync': '0 */12 * * *',  # Every 12 hours
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_LLM_CHUNKING = os.getenv('ENABLE_LLM_CHUNKING', 'False').lower() == 'true'
ENABLE_FORENSIC_SCORES = os.getenv('ENABLE_FORENSIC_SCORES', 'True').lower() == 'true'
ENABLE_GRAPH_ANALYTICS = os.getenv('ENABLE_GRAPH_ANALYTICS', 'True').lower() == 'true'

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
