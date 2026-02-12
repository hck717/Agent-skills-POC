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
import sys
sys.path.insert(0, '/opt/airflow/dags')
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
            
            # Fixed: complete the for loop
            for item in data:
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
    fmp_prices = ti.xcom_pull(key='fmp_eod_prices', task_ids='extract_fmp_eod_prices')
    
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
    df_eodhd['daily_return'] = 0  # Placeholder
    
    # Detect anomalies (circuit breakers, gaps > 20%)
    if 'daily_return' in df_eodhd.columns:
        df_eodhd['anomaly'] = abs(df_eodhd['daily_return']) > config.MAX_PRICE_GAP_PCT
    
    context['ti'].xcom_push(key='transformed_prices', value=df_eodhd.to_dict('records'))
    print(f"Transformed {len(df_eodhd)} price records")
    return True


def load_postgres_prices(**context):
    """
    Load prices into Postgres instead of DuckDB
    """
    import psycopg2
    from psycopg2.extras import execute_values
    import pandas as pd
    
    ti = context['ti']
    transformed_prices = ti.xcom_pull(key='transformed_prices', task_ids='transform_price_data')
    df = pd.DataFrame(transformed_prices)
    
    if df.empty:
        print("No prices to load")
        return False
    
    # Connect to Postgres
    conn = psycopg2.connect(config.POSTGRES_URL)
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_prices (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            date DATE,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            adjusted_close DOUBLE PRECISION,
            volume BIGINT,
            daily_return DOUBLE PRECISION,
            anomaly BOOLEAN,
            source VARCHAR(20),
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    
    execution_date = context['ds']
    
    # Prepare data
    insert_data = [
        (
            row.get('code'),
            execution_date,
            row.get('open'),
            row.get('high'),
            row.get('low'),
            row.get('close'),
            row.get('adjusted_close'),
            row.get('volume'),
            row.get('daily_return'),
            row.get('anomaly'),
            'eodhd'
        )
        for row in transformed_prices
    ]
    
    # Upsert data
    execute_values(
        cur,
        """
        INSERT INTO market_prices (
            ticker, date, open, high, low, close, adjusted_close,
            volume, daily_return, anomaly, source
        ) VALUES %s
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """,
        insert_data
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Loaded {len(insert_data)} price records into Postgres")
    return True


def data_quality_check(**context):
    """
    Validate data quality
    """
    import psycopg2
    
    execution_date = context['ds']
    conn = psycopg2.connect(config.POSTGRES_URL)
    cur = conn.cursor()
    
    # Check 1: No NULL prices
    cur.execute("""
        SELECT COUNT(*) FROM market_prices 
        WHERE date = %s AND (close IS NULL OR volume IS NULL)
    """, [execution_date])
    null_count = cur.fetchone()[0]
    
    if null_count > 0:
        print(f"WARNING: {null_count} records with NULL values")
    
    # Check 2: Coverage percentage
    expected_count = len(config.DEFAULT_TICKERS)
    cur.execute(
        "SELECT COUNT(DISTINCT ticker) FROM market_prices WHERE date = %s",
        [execution_date]
    )
    actual_count = cur.fetchone()[0]
    
    coverage_pct = actual_count / expected_count
    
    if coverage_pct < config.MIN_TICKER_COVERAGE_PCT:
        raise ValueError(f"Coverage too low: {coverage_pct:.2%} < {config.MIN_TICKER_COVERAGE_PCT:.2%}")
    
    print(f"Data quality check passed. Coverage: {coverage_pct:.2%}")
    cur.close()
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
    dag_id="01_daily_market_data",
    default_args=default_args,
    description="Ingest EOD prices and technical indicators",
    schedule_interval="0 22 * * 1-5",
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
        task_id="load_postgres_prices",
        python_callable=load_postgres_prices,
    )

    # Quality check
    task_quality_check = PythonOperator(
        task_id="data_quality_check",
        python_callable=data_quality_check,
    )

    # Define dependencies
    [task_extract_eodhd, task_extract_fmp] >> task_transform
    task_transform >> task_load_prices >> task_quality_check
