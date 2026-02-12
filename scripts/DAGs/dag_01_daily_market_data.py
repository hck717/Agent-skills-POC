"""
DAG 1: Daily Market Data Pipeline
Schedule: Every trading day at 6:00 PM ET (after market close)
Purpose: Ingest EOD prices, volumes, technical indicators, and daily factors
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
import config

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def extract_eodhd_eod_prices(**context):
    """
    Fetch EOD prices for all tracked tickers using EODHD bulk API
    """
    import requests
    import pandas as pd
    from datetime import datetime
    
    execution_date = context['ds']
    print(f"Extracting EODHD EOD prices for {execution_date}")
    
    # EODHD Bulk EOD API endpoint
    url = f"{config.EODHD_BASE_URL}/eod-bulk-last-day/US"
    params = {
        "api_token": config.EODHD_API_KEY,
        "fmt": "json",
        "filter": "extended"
    }
    
    try:
        response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        
        # Filter for tracked tickers
        tickers = config.DEFAULT_TICKERS
        filtered_data = [d for d in data if d.get('code') in tickers]
        
        # Filter out zero volume (non-trading days)
        filtered_data = [d for d in filtered_data if d.get('volume', 0) > 0]
        
        df = pd.DataFrame(filtered_data)
        
        # Store in XCom
        context['ti'].xcom_push(key='eodhd_prices', value=df.to_dict('records'))
        print(f"Extracted {len(df)} ticker prices from EODHD")
        return True
        
    except Exception as e:
        print(f"Error extracting EODHD prices: {e}")
        raise


def extract_fmp_eod_prices(**context):
    """
    Fetch FMP historical prices as backup/validation source
    """
    import requests
    import pandas as pd
    
    execution_date = context['ds']
    print(f"Extracting FMP EOD prices for {execution_date}")
    
    all_data = []
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.FMP_BASE_URL}/historical-price-full/{ticker}"
        params = {
            "apikey": config.FMP_API_KEY,
            "from": execution_date,
            "to": execution_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if 'historical' in data and len(data['historical']) > 0:
                price_data = data['historical'][0]
                price_data['ticker'] = ticker
                all_data.append(price_data)
                
        except Exception as e:
            print(f"Error fetching {ticker} from FMP: {e}")
            continue
    
    df = pd.DataFrame(all_data)
    context['ti'].xcom_push(key='fmp_prices', value=df.to_dict('records'))
    print(f"Extracted {len(df)} ticker prices from FMP")
    return True


def extract_splits_dividends(**context):
    """
    Fetch corporate actions (splits, dividends) for the day
    """
    import requests
    from datetime import datetime
    
    execution_date = context['ds']
    print(f"Extracting splits/dividends for {execution_date}")
    
    actions = []
    
    # EODHD Dividends API
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.EODHD_BASE_URL}/div/{ticker}.US"
        params = {
            "api_token": config.EODHD_API_KEY,
            "fmt": "json",
            "from": execution_date,
            "to": execution_date
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            for item in 
                item['ticker'] = ticker
                item['action_type'] = 'dividend'
                actions.append(item)
                
        except Exception as e:
            print(f"Error fetching dividends for {ticker}: {e}")
    
    context['ti'].xcom_push(key='corporate_actions', value=actions)
    print(f"Extracted {len(actions)} corporate actions")
    return True


def transform_price_data(**context):
    """
    Transform and validate price data
    Calculate daily returns, detect anomalies
    """
    import pandas as pd
    import numpy as np
    
    ti = context['ti']
    eodhd_prices = ti.xcom_pull(key='eodhd_prices', task_ids='extract_eodhd_eod_prices')
    fmp_prices = ti.xcom_pull(key='fmp_prices', task_ids='extract_fmp_eod_prices')
    
    df_eodhd = pd.DataFrame(eodhd_prices)
    df_fmp = pd.DataFrame(fmp_prices)
    
    # Cross-validate prices (ensure consistency between sources)
    if not df_eodhd.empty and not df_fmp.empty:
        # Merge and check for discrepancies > 1%
        merged = pd.merge(
            df_eodhd[['code', 'close']],
            df_fmp[['ticker', 'close']],
            left_on='code',
            right_on='ticker',
            suffixes=('_eodhd', '_fmp')
        )
        merged['price_diff_pct'] = abs(merged['close_eodhd'] - merged['close_fmp']) / merged['close_eodhd']
        discrepancies = merged[merged['price_diff_pct'] > 0.01]
        
        if len(discrepancies) > 0:
            print(f"WARNING: Price discrepancies found for {len(discrepancies)} tickers")
    
    # Calculate daily returns
    # Note: Requires previous day's close - would query from DuckDB
    df_eodhd['daily_return'] = 0  # Placeholder
    
    # Detect anomalies (circuit breakers, gaps > 20%)
    if 'daily_return' in df_eodhd.columns:
        df_eodhd['anomaly'] = abs(df_eodhd['daily_return']) > config.MAX_PRICE_GAP_PCT
    
    context['ti'].xcom_push(key='transformed_prices', value=df_eodhd.to_dict('records'))
    print(f"Transformed {len(df_eodhd)} price records")
    return True


def load_duckdb_prices(**context):
    """
    Load prices into DuckDB partitioned table
    """
    import duckdb
    import pandas as pd
    from pathlib import Path
    
    ti = context['ti']
    transformed_prices = ti.xcom_pull(key='transformed_prices', task_ids='transform_price_data')
    df = pd.DataFrame(transformed_prices)
    
    if df.empty:
        print("No prices to load")
        return False
    
    # Ensure DuckDB directory exists
    Path(config.DUCKDB_PATH).mkdir(parents=True, exist_ok=True)
    
    # Connect to DuckDB
    conn = duckdb.connect(config.DUCKDB_PRICES_DB)
    
    # Create table if not exists (partitioned by year-month)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adjusted_close DOUBLE,
            volume BIGINT,
            daily_return DOUBLE,
            anomaly BOOLEAN,
            source VARCHAR,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert data (idempotent - delete existing records for this date first)
    execution_date = context['ds']
    conn.execute(f"DELETE FROM prices WHERE date = '{execution_date}'")
    conn.execute("INSERT INTO prices SELECT * FROM df")
    
    row_count = conn.execute("SELECT COUNT(*) FROM prices WHERE date = ?", [execution_date]).fetchone()[0]
    conn.close()
    
    print(f"Loaded {row_count} price records into DuckDB")
    return True


def calculate_technical_indicators(**context):
    """
    Calculate technical indicators (MA, RSI, MACD, Bollinger Bands)
    """
    import duckdb
    import pandas as pd
    
    conn = duckdb.connect(config.DUCKDB_PRICES_DB)
    
    # Create technical_indicators table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            ticker VARCHAR,
            date DATE,
            ma_20 DOUBLE,
            ma_50 DOUBLE,
            ma_200 DOUBLE,
            rsi_14 DOUBLE,
            macd DOUBLE,
            macd_signal DOUBLE,
            bb_upper DOUBLE,
            bb_lower DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    
    # Calculate indicators using window functions
    # Note: This is simplified - real implementation would use pandas-ta or ta-lib
    for ticker in config.DEFAULT_TICKERS:
        query = f"""
            INSERT OR REPLACE INTO technical_indicators
            SELECT 
                ticker,
                date,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma_20,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as ma_50,
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma_200,
                NULL as rsi_14,  -- Placeholder
                NULL as macd,
                NULL as macd_signal,
                NULL as bb_upper,
                NULL as bb_lower
            FROM prices
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT 1
        """
        conn.execute(query)
    
    conn.close()
    print("Technical indicators calculated")
    return True


def calculate_daily_factors(**context):
    """
    Calculate momentum and volatility factors
    """
    import duckdb
    
    conn = duckdb.connect(config.DUCKDB_PRICES_DB)
    
    # Create derived_factors table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS derived_factors (
            ticker VARCHAR,
            date DATE,
            factor_type VARCHAR,  -- 'momentum', 'volatility'
            factor_name VARCHAR,  -- '1M_return', 'beta_60d'
            value DOUBLE,
            percentile DOUBLE,  -- Cross-sectional ranking
            PRIMARY KEY (ticker, date, factor_type, factor_name)
        )
    """)
    
    execution_date = context['ds']
    
    # Calculate momentum factors (1M, 3M, 6M, 12M returns)
    momentum_periods = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}
    
    for period_name, days in momentum_periods.items():
        conn.execute(f"""
            INSERT OR REPLACE INTO derived_factors
            SELECT 
                ticker,
                '{execution_date}' as date,
                'momentum' as factor_type,
                '{period_name}_return' as factor_name,
                (close / LAG(close, {days}) OVER (PARTITION BY ticker ORDER BY date) - 1) as value,
                NULL as percentile
            FROM prices
            WHERE date = '{execution_date}'
        """)
    
    # Calculate volatility (60-day std dev)
    conn.execute(f"""
        INSERT OR REPLACE INTO derived_factors
        SELECT 
            ticker,
            '{execution_date}' as date,
            'volatility' as factor_type,
            'std_60d' as factor_name,
            STDDEV(daily_return) OVER (
                PARTITION BY ticker 
                ORDER BY date 
                ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) as value,
            NULL as percentile
        FROM prices
        WHERE date <= '{execution_date}'
    """)
    
    conn.close()
    print("Daily factors calculated")
    return True


def data_quality_check(**context):
    """
    Validate data quality
    """
    import duckdb
    
    execution_date = context['ds']
    conn = duckdb.connect(config.DUCKDB_PRICES_DB)
    
    # Check 1: No NULL prices
    null_count = conn.execute("""
        SELECT COUNT(*) FROM prices 
        WHERE date = ? AND (close IS NULL OR volume IS NULL)
    """, [execution_date]).fetchone()[0]
    
    if null_count > 0:
        print(f"WARNING: {null_count} records with NULL values")
    
    # Check 2: Coverage percentage
    expected_count = len(config.DEFAULT_TICKERS)
    actual_count = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM prices WHERE date = ?",
        [execution_date]
    ).fetchone()[0]
    
    coverage_pct = actual_count / expected_count
    
    if coverage_pct < config.MIN_TICKER_COVERAGE_PCT:
        raise ValueError(f"Coverage too low: {coverage_pct:.2%} < {config.MIN_TICKER_COVERAGE_PCT:.2%}")
    
    print(f"Data quality check passed. Coverage: {coverage_pct:.2%}")
    conn.close()
    return True


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = config.DEFAULT_ARGS.copy()
default_args.update({
    "start_date": datetime(2026, 1, 1),
})

with DAG(
    dag_id="dag_01_daily_market_data_pipeline",
    default_args=default_args,
    description="Ingest EOD prices, technical indicators, and daily factors",
    schedule_interval="0 22 * * 1-5",  # 6 PM ET = 10 PM HKT on weekdays
    catchup=False,
    max_active_runs=config.MAX_ACTIVE_RUNS,
    tags=["daily", "market_data", "prices"],
) as dag:

    # Extract tasks
    task_extract_eodhd = PythonOperator(
        task_id="extract_eodhd_eod_prices",
        python_callable=extract_eodhd_eod_prices,
    )

    task_extract_fmp = PythonOperator(
        task_id="extract_fmp_eod_prices",
        python_callable=extract_fmp_eod_prices,
    )

    task_extract_actions = PythonOperator(
        task_id="extract_splits_dividends",
        python_callable=extract_splits_dividends,
    )

    # Transform task
    task_transform = PythonOperator(
        task_id="transform_price_data",
        python_callable=transform_price_data,
    )

    # Load task
    task_load_prices = PythonOperator(
        task_id="load_duckdb_prices",
        python_callable=load_duckdb_prices,
    )

    # Calculate tasks
    task_calc_indicators = PythonOperator(
        task_id="calculate_technical_indicators",
        python_callable=calculate_technical_indicators,
    )

    task_calc_factors = PythonOperator(
        task_id="calculate_daily_factors",
        python_callable=calculate_daily_factors,
    )

    # Quality check
    task_quality_check = PythonOperator(
        task_id="data_quality_check",
        python_callable=data_quality_check,
    )

    # Define dependencies
    [task_extract_eodhd, task_extract_fmp] >> task_transform
    task_transform >> task_load_prices
    task_load_prices >> [task_calc_indicators, task_calc_factors]
    [task_calc_indicators, task_calc_factors] >> task_quality_check
