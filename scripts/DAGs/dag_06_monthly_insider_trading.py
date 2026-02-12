"""
DAG 06: Monthly Insider Trading Pipeline
Schedule: 1st of every month at 3:00 AM HKT
Purpose: Refresh insider transaction data and calculate sentiment scores
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import psycopg2
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE, POSTGRES_URL,
    EODHD_API_KEY, FMP_API_KEY, EODHD_BASE_URL, FMP_BASE_URL, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_eodhd_insider_transactions(**context):
    """Extract Form 4 data from EODHD (last 90 days)"""
    logger.info(f"Extracting insider transactions for {len(DEFAULT_TICKERS)} tickers")
    transactions_data = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{EODHD_BASE_URL}/insider-transactions"
            params = {
                "api_token": EODHD_API_KEY,
                "code": f"{ticker}.US",
                "limit": 100
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for txn in data:
                transactions_data.append({
                    'ticker': ticker,
                    'transaction_date': txn.get('date'),
                    'owner_name': txn.get('ownerName'),
                    'owner_title': txn.get('ownerTitle'),
                    'transaction_type': txn.get('transactionType'),
                    'shares': txn.get('transactionShares'),
                    'price': txn.get('transactionPrice'),
                    'value': txn.get('transactionValue'),
                    'source': 'eodhd'
                })
            
            logger.info(f"✅ {ticker}: {len(data)} insider transactions")
        except Exception as e:
            logger.warning(f"EODHD insider data failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='transactions_data', value=transactions_data)
    return {"extracted": len(transactions_data)}

def extract_fmp_insider_trading(**context):
    """Extract FMP insider trading as backup"""
    transactions_data = context['ti'].xcom_pull(task_ids='extract_eodhd_insider_transactions', key='transactions_data') or []
    logger.info("Extracting FMP insider trading data")
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{FMP_BASE_URL}/insider-trading"
            params = {"symbol": ticker, "apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for txn in data[:50]:
                transactions_data.append({
                    'ticker': ticker,
                    'transaction_date': txn.get('filingDate'),
                    'owner_name': txn.get('reportingName'),
                    'owner_title': txn.get('typeOfOwner'),
                    'transaction_type': txn.get('acquistionOrDisposition'),
                    'shares': txn.get('securitiesTransacted'),
                    'price': txn.get('price'),
                    'value': txn.get('securitiesTransacted', 0) * txn.get('price', 0),
                    'source': 'fmp'
                })
        except Exception as e:
            logger.warning(f"FMP insider data failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='transactions_data', value=transactions_data)
    logger.info(f"✅ Total transactions: {len(transactions_data)}")
    return {"extracted": len(transactions_data)}

def transform_insider_data(**context):
    """Classify transaction types and detect clusters"""
    transactions_data = context['ti'].xcom_pull(task_ids='extract_fmp_insider_trading', key='transactions_data') or []
    logger.info(f"Transforming {len(transactions_data)} insider transactions")
    
    df = pd.DataFrame(transactions_data)
    if df.empty:
        context['ti'].xcom_push(key='transformed_data', value=[])
        return {"transformed": 0}
    
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df = df.dropna(subset=['transaction_date'])
    
    def classify_txn(row):
        txn_type = str(row['transaction_type']).lower()
        if 'buy' in txn_type or 'purchase' in txn_type:
            return 'buy'
        elif 'sell' in txn_type or 'sale' in txn_type:
            return 'sell'
        elif 'option' in txn_type or 'exercise' in txn_type:
            return 'option_exercise'
        elif 'gift' in txn_type:
            return 'gift'
        else:
            return 'other'
    
    df['txn_class'] = df.apply(classify_txn, axis=1)
    
    df = df.sort_values(['ticker', 'transaction_date'])
    df['cluster'] = (df.groupby('ticker')['transaction_date'].diff().dt.days > 7).cumsum()
    
    # FIX: Convert pandas Timestamp to string for XCom serialization
    df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')
    
    transformed_data = df.to_dict('records')
    context['ti'].xcom_push(key='transformed_data', value=transformed_data)
    logger.info(f"✅ Transformed {len(transformed_data)} transactions")
    return {"transformed": len(transformed_data)}

def load_postgres_insider_transactions(**context):
    """Load insider transactions to Postgres"""
    transformed_data = context['ti'].xcom_pull(task_ids='transform_insider_data', key='transformed_data') or []
    
    if not transformed_data:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(transformed_data)} insider transactions to Postgres")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            transaction_date DATE,
            owner_name VARCHAR(255),
            owner_title VARCHAR(100),
            transaction_type VARCHAR(50),
            txn_class VARCHAR(20),
            shares BIGINT,
            price NUMERIC,
            value NUMERIC,
            cluster INTEGER,
            source VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, transaction_date, owner_name, shares)
        )
    """)
    
    for record in transformed_data:
        cursor.execute("""
            INSERT INTO insider_transactions 
            (ticker, transaction_date, owner_name, owner_title, transaction_type, txn_class, shares, price, value, cluster, source)
            VALUES (%(ticker)s, %(transaction_date)s, %(owner_name)s, %(owner_title)s, %(transaction_type)s, 
                    %(txn_class)s, %(shares)s, %(price)s, %(value)s, %(cluster)s, %(source)s)
            ON CONFLICT (ticker, transaction_date, owner_name, shares) DO UPDATE SET
                value = EXCLUDED.value,
                txn_class = EXCLUDED.txn_class
        """, record)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✅ Loaded {len(transformed_data)} insider transactions to Postgres")
    return {"loaded": len(transformed_data)}

def calculate_insider_sentiment_score(**context):
    """Calculate quarterly insider sentiment score"""
    logger.info("Calculating insider sentiment scores")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ticker,
            DATE_TRUNC('quarter', transaction_date) as quarter,
            SUM(CASE WHEN txn_class = 'buy' THEN value ELSE 0 END) as buy_value,
            SUM(CASE WHEN txn_class = 'sell' THEN value ELSE 0 END) as sell_value,
            COUNT(DISTINCT owner_name) as unique_insiders
        FROM insider_transactions
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '2 years'
        GROUP BY ticker, quarter
    """)
    
    sentiment_data = cursor.fetchall()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insider_sentiment (
            ticker VARCHAR(10),
            quarter DATE,
            buy_value NUMERIC,
            sell_value NUMERIC,
            net_sentiment NUMERIC,
            unique_insiders INTEGER,
            PRIMARY KEY (ticker, quarter)
        )
    """)
    
    for ticker, quarter, buy_value, sell_value, unique_insiders in sentiment_data:
        total_value = (buy_value or 0) + (sell_value or 0)
        net_sentiment = ((buy_value or 0) - (sell_value or 0)) / total_value if total_value > 0 else 0
        
        cursor.execute("""
            INSERT INTO insider_sentiment (ticker, quarter, buy_value, sell_value, net_sentiment, unique_insiders)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, quarter) DO UPDATE SET
                buy_value = EXCLUDED.buy_value,
                sell_value = EXCLUDED.sell_value,
                net_sentiment = EXCLUDED.net_sentiment,
                unique_insiders = EXCLUDED.unique_insiders
        """, (ticker, quarter, buy_value or 0, sell_value or 0, net_sentiment, unique_insiders))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✅ Calculated sentiment for {len(sentiment_data)} ticker-quarters")
    return {"calculated": len(sentiment_data)}

with DAG(
    dag_id='06_monthly_insider_trading_pipeline',
    default_args=DEFAULT_ARGS,
    description='Monthly insider transaction data refresh',
    schedule_interval='0 19 1 * *' if not TEST_MODE else None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['monthly', 'insider_trading'],
) as dag:
    
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_insider_transactions', python_callable=extract_eodhd_insider_transactions)
    task_extract_fmp = PythonOperator(task_id='extract_fmp_insider_trading', python_callable=extract_fmp_insider_trading)
    task_transform = PythonOperator(task_id='transform_insider_data', python_callable=transform_insider_data)
    task_load_postgres = PythonOperator(task_id='load_postgres_insider_transactions', python_callable=load_postgres_insider_transactions)
    task_calculate_sentiment = PythonOperator(task_id='calculate_insider_sentiment_score', python_callable=calculate_insider_sentiment_score)
    
    task_extract_eodhd >> task_extract_fmp >> task_transform >> task_load_postgres >> task_calculate_sentiment
