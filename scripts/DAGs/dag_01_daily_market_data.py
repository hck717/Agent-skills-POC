"""
DAG 01: Daily Market Data Pipeline
Schedule: Every trading day at 6:00 PM ET (after market close)
Purpose: Ingest EOD prices, volumes, and corporate actions
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import psycopg2
from neo4j import GraphDatabase
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE,
    POSTGRES_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EODHD_API_KEY, FMP_API_KEY, EODHD_BASE_URL, FMP_BASE_URL,
    MAX_ACTIVE_RUNS, LOOKBACK_DAYS, MIN_TICKER_COVERAGE_PCT
)

logger = logging.getLogger(__name__)

# EXTRACTION
def extract_eodhd_eod_prices(**context):
    logger.info(f"Extracting EOD prices for {len(DEFAULT_TICKERS)} tickers from EODHD")
    prices_data = []
    failed_tickers = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{EODHD_BASE_URL}/eod/{ticker}.US"
            params = {"api_token": EODHD_API_KEY, "fmt": "json", "from": (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for row in data:
                if row.get('volume', 0) > 0:
                    prices_data.append({
                        'ticker': ticker, 'date': row['date'], 'open': float(row['open']),
                        'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close']),
                        'adjusted_close': float(row.get('adjusted_close', row['close'])),
                        'volume': int(row['volume']), 'source': 'eodhd'
                    })
            logger.info(f"✅ {ticker}: {len([p for p in prices_data if p['ticker']==ticker])} records")
        except Exception as e:
            logger.warning(f"⚠️ EODHD failed for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    context['ti'].xcom_push(key='prices_data', value=prices_data)
    context['ti'].xcom_push(key='failed_tickers', value=failed_tickers)
    return {"extracted": len(prices_data), "failed": len(failed_tickers)}

def extract_fmp_eod_prices(**context):
    failed_tickers = context['ti'].xcom_pull(task_ids='extract_eodhd_eod_prices', key='failed_tickers') or []
    if not failed_tickers:
        return {"extracted": 0}
    
    logger.info(f"Falling back to FMP for {len(failed_tickers)} tickers")
    prices_data = context['ti'].xcom_pull(task_ids='extract_eodhd_eod_prices', key='prices_data')
    
    for ticker in failed_tickers:
        try:
            url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
            response = requests.get(url, params={"apikey": FMP_API_KEY}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'historical' in data:
                for row in data['historical'][:LOOKBACK_DAYS]:
                    if row.get('volume', 0) > 0:
                        prices_data.append({
                            'ticker': ticker, 'date': row['date'], 'open': float(row['open']),
                            'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close']),
                            'adjusted_close': float(row.get('adjClose', row['close'])),
                            'volume': int(row['volume']), 'source': 'fmp'
                        })
                logger.info(f"✅ FMP rescued {ticker}")
        except Exception as e:
            logger.error(f"❌ FMP also failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='prices_data', value=prices_data)
    return {"extracted": len([p for p in prices_data if p['source']=='fmp'])}

def extract_splits_dividends(**context):
    logger.info("Extracting splits and dividends")
    splits_data, dividends_data = [], []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{EODHD_BASE_URL}/splits/{ticker}.US"
            response = requests.get(url, params={"api_token": EODHD_API_KEY, "fmt": "json"}, timeout=15)
            if response.status_code == 200: splits_data.extend(response.json())
            
            url = f"{EODHD_BASE_URL}/div/{ticker}.US"
            response = requests.get(url, params={"api_token": EODHD_API_KEY, "fmt": "json"}, timeout=15)
            if response.status_code == 200: dividends_data.extend(response.json())
        except Exception as e:
            logger.warning(f"Failed splits/dividends for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='splits_data', value=splits_data)
    context['ti'].xcom_push(key='dividends_data', value=dividends_data)
    return {"splits": len(splits_data), "dividends": len(dividends_data)}

# TRANSFORMATION
def transform_price_data(**context):
    # FIX: Pull from FMP task which contains merged data from both EODHD and FMP
    prices_data = context['ti'].xcom_pull(task_ids='extract_fmp_eod_prices', key='prices_data')
    
    # Fallback to EODHD data if FMP didn't push anything
    if not prices_data:
        prices_data = context['ti'].xcom_pull(task_ids='extract_eodhd_eod_prices', key='prices_data') or []
    
    logger.info(f"Transforming {len(prices_data)} price records")
    
    if not prices_data:
        logger.error("No price data available for transformation")
        raise ValueError("No price data to transform")
    
    df = pd.DataFrame(prices_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    df['price_gap'] = df['daily_return'].abs() > 0.20
    
    anomalies = df[df['price_gap']]
    if not anomalies.empty:
        logger.warning(f"⚠️ Price anomalies: {len(anomalies)} records")
    
    context['ti'].xcom_push(key='transformed_data', value=df.to_dict('records'))
    return {"transformed": len(df), "anomalies": len(anomalies)}

# LOADING
def load_postgres_prices(**context):
    transformed_data = context['ti'].xcom_pull(task_ids='transform_price_data', key='transformed_data')
    logger.info(f"Loading {len(transformed_data)} records to Postgres")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker VARCHAR(10) PRIMARY KEY, name VARCHAR(255), exchange VARCHAR(50),
            sector VARCHAR(100), industry VARCHAR(100), updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_prices (
            ticker VARCHAR(10), date DATE, open NUMERIC, high NUMERIC, low NUMERIC,
            close NUMERIC, adjusted_close NUMERIC, volume BIGINT, daily_return NUMERIC,
            source VARCHAR(20), PRIMARY KEY (ticker, date)
        )
    """)
    
    for ticker in set(r['ticker'] for r in transformed_data):
        cursor.execute("INSERT INTO companies (ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING", (ticker,))
    
    for record in transformed_data:
        cursor.execute("""
            INSERT INTO market_prices (ticker, date, open, high, low, close, adjusted_close, volume, daily_return, source)
            VALUES (%(ticker)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adjusted_close)s, %(volume)s, %(daily_return)s, %(source)s)
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
                adjusted_close = EXCLUDED.adjusted_close, volume = EXCLUDED.volume,
                daily_return = EXCLUDED.daily_return, source = EXCLUDED.source
        """, record)
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"✅ Loaded {len(transformed_data)} records to Postgres")
    return {"loaded": len(transformed_data)}

def load_neo4j_prices(**context):
    transformed_data = context['ti'].xcom_pull(task_ids='transform_price_data', key='transformed_data')
    logger.info(f"Loading {len(transformed_data)} records to Neo4j")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        df = pd.DataFrame(transformed_data)
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            session.run("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.created_at = datetime()
                ON MATCH SET c.updated_at = datetime()
            """, ticker=ticker)
            
            for _, row in ticker_data.iterrows():
                session.run("""
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (p:Price {ticker: $ticker, date: date($date)})
                    ON CREATE SET p.open = $open, p.high = $high, p.low = $low, p.close = $close,
                                  p.volume = $volume, p.daily_return = $daily_return, p.created_at = datetime()
                    ON MATCH SET p.open = $open, p.high = $high, p.low = $low, p.close = $close,
                                 p.volume = $volume, p.daily_return = $daily_return, p.updated_at = datetime()
                    MERGE (c)-[:HAS_PRICE]->(p)
                """, ticker=ticker, date=row['date'].strftime('%Y-%m-%d'),
                     open=float(row['open']), high=float(row['high']), low=float(row['low']),
                     close=float(row['close']), volume=int(row['volume']),
                     daily_return=float(row.get('daily_return', 0) or 0))
            
            logger.info(f"✅ Loaded {ticker}: {len(ticker_data)} prices to Neo4j")
    
    driver.close()
    logger.info(f"✅ Neo4j load complete")
    return {"loaded": len(transformed_data)}

# QUALITY CHECK
def data_quality_check(**context):
    logger.info("Running data quality checks...")
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM market_prices WHERE date >= CURRENT_DATE - INTERVAL '7 days'")
    recent_tickers = cursor.fetchone()[0]
    coverage = recent_tickers / len(DEFAULT_TICKERS)
    logger.info(f"Coverage: {recent_tickers}/{len(DEFAULT_TICKERS)} ({coverage:.1%})")
    
    if coverage < MIN_TICKER_COVERAGE_PCT:
        raise ValueError(f"Coverage {coverage:.1%} < {MIN_TICKER_COVERAGE_PCT:.1%}")
    
    cursor.execute("SELECT COUNT(*) FROM market_prices WHERE close IS NULL AND date >= CURRENT_DATE - INTERVAL '7 days'")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        raise ValueError(f"Found {null_count} NULL prices")
    
    cursor.close()
    conn.close()
    logger.info("✅ Quality checks passed")
    return {"coverage": coverage, "null_count": null_count}

# DAG DEFINITION
with DAG(
    dag_id='01_daily_market_data_pipeline',
    default_args=DEFAULT_ARGS,
    description='Daily EOD prices with Neo4j loading',
    schedule_interval='0 23 * * 1-5' if not TEST_MODE else None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['daily', 'market_data', 'critical'],
) as dag:
    
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_eod_prices', python_callable=extract_eodhd_eod_prices)
    task_extract_fmp = PythonOperator(task_id='extract_fmp_eod_prices', python_callable=extract_fmp_eod_prices)
    task_extract_splits = PythonOperator(task_id='extract_splits_dividends', python_callable=extract_splits_dividends)
    task_transform = PythonOperator(task_id='transform_price_data', python_callable=transform_price_data)
    task_load_postgres = PythonOperator(task_id='load_postgres_prices', python_callable=load_postgres_prices)
    task_load_neo4j = PythonOperator(task_id='load_neo4j_prices', python_callable=load_neo4j_prices)
    task_quality_check = PythonOperator(task_id='data_quality_check', python_callable=data_quality_check)
    
    [task_extract_eodhd, task_extract_splits] >> task_extract_fmp >> task_transform >> [task_load_postgres, task_load_neo4j] >> task_quality_check
