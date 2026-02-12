"""
DAG 6: Monthly Insider Trading Pipeline
Schedule: 1st of every month at 3:00 AM HKT
Purpose: Refresh insider transaction data
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import config

def extract_eodhd_insider_transactions(**context):
    """Fetch Form 4 data for all tracked tickers (last 90 days)"""
    import requests
    from datetime import datetime, timedelta
    
    end_date = datetime.fromisoformat(context['ds'])
    start_date = end_date - timedelta(days=90)
    
    all_transactions = []
    
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.EODHD_BASE_URL}/insider-transactions"
        params = {
            "api_token": config.EODHD_API_KEY,
            "code": f"{ticker}.US",
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "fmt": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            for transaction in 
                transaction['ticker'] = ticker
                all_transactions.append(transaction)
                
        except Exception as e:
            print(f"Error fetching insider transactions for {ticker}: {e}")
    
    context['ti'].xcom_push(key='eodhd_transactions', value=all_transactions)
    print(f"Extracted {len(all_transactions)} insider transactions from EODHD")
    return True

def extract_fmp_insider_trading(**context):
    """Fetch FMP insider trading API"""
    import requests
    
    all_transactions = []
    
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.FMP_BASE_URL}/insider-trading"
        params = {
            "symbol": ticker,
            "apikey": config.FMP_API_KEY,
            "limit": 100
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            for transaction in 
                transaction['ticker'] = ticker
                all_transactions.append(transaction)
                
        except Exception as e:
            print(f"Error fetching FMP insider trades for {ticker}: {e}")
    
    context['ti'].xcom_push(key='fmp_transactions', value=all_transactions)
    print(f"Extracted {len(all_transactions)} insider transactions from FMP")
    return True

def transform_insider_data(**context):
    """Classify transaction types and detect clusters"""
    import pandas as pd
    from datetime import timedelta
    
    ti = context['ti']
    eodhd_trans = ti.xcom_pull(key='eodhd_transactions', task_ids='extract_eodhd_insider_transactions') or []
    fmp_trans = ti.xcom_pull(key='fmp_transactions', task_ids='extract_fmp_insider_trading') or []
    
    # Combine and deduplicate
    all_trans = eodhd_trans + fmp_trans
    df = pd.DataFrame(all_trans)
    
    if df.empty:
        print("No insider transactions to process")
        context['ti'].xcom_push(key='transformed_transactions', value=[])
        return True
    
    # Classify transaction types
    def classify_transaction(row):
        trans_type = str(row.get('transactionType', '')).lower()
        
        if 'sale' in trans_type or 'sell' in trans_type:
            return 'Sale'
        elif 'purchase' in trans_type or 'buy' in trans_type:
            return 'Purchase'
        elif 'option' in trans_type:
            return 'Option Exercise'
        elif 'gift' in trans_type:
            return 'Gift'
        else:
            return 'Other'
    
    df['classified_type'] = df.apply(classify_transaction, axis=1)
    
    # Calculate transaction value
    df['transaction_value'] = df.get('shares', 0) * df.get('price', 0)
    
    # Detect clusters (multiple insiders trading within 7 days)
    df['transaction_date'] = pd.to_datetime(df['transactionDate'])
    df = df.sort_values(['ticker', 'transaction_date'])
    
    # Flag clusters
    clusters = []
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        
        for i, row in ticker_df.iterrows():
            window_start = row['transaction_date'] - timedelta(days=7)
            window_end = row['transaction_date'] + timedelta(days=7)
            
            window_trans = ticker_df[
                (ticker_df['transaction_date'] >= window_start) &
                (ticker_df['transaction_date'] <= window_end)
            ]
            
            if len(window_trans) >= 3:  # 3+ insiders in 7-day window
                clusters.append({
                    'ticker': ticker,
                    'date': row['transaction_date'],
                    'transaction_count': len(window_trans),
                    'cluster_type': row['classified_type']
                })
    
    context['ti'].xcom_push(key='transformed_transactions', value=df.to_dict('records'))
    context['ti'].xcom_push(key='transaction_clusters', value=clusters)
    print(f"Transformed {len(df)} transactions, detected {len(clusters)} clusters")
    return True

def load_postgres_insider_transactions(**context):
    """Upsert into Postgres insider_transactions table"""
    import psycopg2
    from psycopg2.extras import execute_values
    
    ti = context['ti']
    transactions = ti.xcom_pull(key='transformed_transactions', task_ids='transform_insider_data')
    
    if not transactions:
        print("No transactions to load")
        return True
    
    # Connect to Postgres
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            transaction_date DATE NOT NULL,
            owner_name VARCHAR(200),
            owner_title VARCHAR(200),
            transaction_type VARCHAR(50),
            classified_type VARCHAR(50),
            shares BIGINT,
            price NUMERIC,
            transaction_value NUMERIC,
            shares_owned_after BIGINT,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, transaction_date, owner_name, transaction_type)
        )
    """)
    
    # Create TimescaleDB hypertable (if TimescaleDB extension is available)
    try:
        cur.execute("""
            SELECT create_hypertable('insider_transactions', 'transaction_date', 
                                   if_not_exists => TRUE,
                                   migrate_data => TRUE)
        """)
    except Exception as e:
        print(f"Note: Could not create hypertable (normal if TimescaleDB not installed): {e}")
    
    # Prepare data for insertion
    insert_data = [
        (
            t.get('ticker'),
            t.get('transactionDate'),
            t.get('reportingName'),
            t.get('typeOfOwner'),
            t.get('transactionType'),
            t.get('classified_type'),
            t.get('shares'),
            t.get('price'),
            t.get('transaction_value'),
            t.get('securitiesOwned')
        )
        for t in transactions
    ]
    
    # Upsert data
    execute_values(
        cur,
        """
        INSERT INTO insider_transactions (
            ticker, transaction_date, owner_name, owner_title, transaction_type,
            classified_type, shares, price, transaction_value, shares_owned_after
        ) VALUES %s
        ON CONFLICT (ticker, transaction_date, owner_name, transaction_type)
        DO UPDATE SET
            shares = EXCLUDED.shares,
            price = EXCLUDED.price,
            transaction_value = EXCLUDED.transaction_value
        """,
        insert_data
    )
    
    conn.commit()
    
    # Create aggregates table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insider_aggregates (
            ticker VARCHAR(10),
            quarter VARCHAR(7),  -- Format: 2026-Q1
            total_buys BIGINT,
            total_sells BIGINT,
            buy_value NUMERIC,
            sell_value NUMERIC,
            unique_insiders INT,
            net_sentiment NUMERIC,
            PRIMARY KEY (ticker, quarter)
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Loaded {len(transactions)} insider transactions into Postgres")
    return True

def calculate_insider_sentiment_score(**context):
    """Calculate aggregated insider sentiment by quarter"""
    import psycopg2
    
    # Connect to Postgres
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Calculate quarterly aggregates
    cur.execute("""
        INSERT INTO insider_aggregates (ticker, quarter, total_buys, total_sells, 
                                       buy_value, sell_value, unique_insiders, net_sentiment)
        SELECT 
            ticker,
            TO_CHAR(transaction_date, 'YYYY') || '-Q' || EXTRACT(QUARTER FROM transaction_date)::TEXT as quarter,
            SUM(CASE WHEN classified_type = 'Purchase' THEN shares ELSE 0 END) as total_buys,
            SUM(CASE WHEN classified_type = 'Sale' THEN shares ELSE 0 END) as total_sells,
            SUM(CASE WHEN classified_type = 'Purchase' THEN transaction_value ELSE 0 END) as buy_value,
            SUM(CASE WHEN classified_type = 'Sale' THEN transaction_value ELSE 0 END) as sell_value,
            COUNT(DISTINCT owner_name) as unique_insiders,
            (SUM(CASE WHEN classified_type = 'Purchase' THEN transaction_value ELSE 0 END) - 
             SUM(CASE WHEN classified_type = 'Sale' THEN transaction_value ELSE 0 END)) / 
            NULLIF(SUM(ABS(transaction_value)), 0) as net_sentiment
        FROM insider_transactions
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY ticker, quarter
        ON CONFLICT (ticker, quarter) DO UPDATE SET
            total_buys = EXCLUDED.total_buys,
            total_sells = EXCLUDED.total_sells,
            buy_value = EXCLUDED.buy_value,
            sell_value = EXCLUDED.sell_value,
            unique_insiders = EXCLUDED.unique_insiders,
            net_sentiment = EXCLUDED.net_sentiment
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("Insider sentiment scores calculated")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_06_monthly_insider_trading_pipeline",
    default_args=default_args,
    description="Refresh insider transaction data",
    schedule_interval="0 3 1 * *",  # 1st of month at 3 AM HKT
    catchup=False,
    max_active_runs=1,
    tags=["monthly", "insider_trading", "form_4"],
) as dag:

    extract_eodhd = PythonOperator(
        task_id="extract_eodhd_insider_transactions",
        python_callable=extract_eodhd_insider_transactions,
    )

    extract_fmp = PythonOperator(
        task_id="extract_fmp_insider_trading",
        python_callable=extract_fmp_insider_trading,
    )

    transform = PythonOperator(
        task_id="transform_insider_data",
        python_callable=transform_insider_data,
    )

    load_postgres = PythonOperator(
        task_id="load_postgres_insider_transactions",
        python_callable=load_postgres_insider_transactions,
    )

    calc_sentiment = PythonOperator(
        task_id="calculate_insider_sentiment_score",
        python_callable=calculate_insider_sentiment_score,
    )

    [extract_eodhd, extract_fmp] >> transform >> load_postgres >> calc_sentiment
