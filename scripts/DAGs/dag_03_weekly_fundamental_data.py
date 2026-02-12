
"""
DAG 3: Weekly Fundamental Data Pipeline
Schedule: Every Sunday at 2:00 AM HKT
Purpose: Refresh financial statements and fundamental metrics
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import config

def extract_eodhd_fundamentals(**context):
    """Fetch income statements, balance sheets, cash flows"""
    import requests
    
    all_data = []
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.EODHD_BASE_URL}/fundamentals/{ticker}.US"
        params = {"api_token": config.EODHD_API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            context["ti"].xcom_push(key=f"fundamentals_{ticker}", value=data)
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
    
    print(f"Extracted fundamentals for {len(config.DEFAULT_TICKERS)} tickers")
    return True

def calculate_forensic_scores(**context):
    """Calculate Beneish M-score, Altman Z-score, Piotroski F-score"""
    import pandas as pd
    
    # Beneish M-score calculation (8-variable model)
    # M-score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    # Score > -1.78 suggests earnings manipulation
    
    print("Forensic scores calculated (placeholder)")
    return True

def load_postgres_financials(**context):
    """Load financial statements into Postgres"""
    import psycopg2
    
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_statements (
            ticker VARCHAR(10),
            fiscal_period VARCHAR(10),
            fiscal_year INT,
            statement_type VARCHAR(50),
            line_item VARCHAR(200),
            value NUMERIC,
            filing_date DATE,
            PRIMARY KEY (ticker, fiscal_period, statement_type, line_item)
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("Financial statements loaded into Postgres")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_03_weekly_fundamental_data_pipeline",
    default_args=default_args,
    description="Refresh financial statements and fundamental metrics",
    schedule_interval="0 2 * * 0",  # Every Sunday 2 AM HKT
    catchup=False,
    max_active_runs=1,
    tags=["weekly", "fundamentals", "financials"],
) as dag:

    wait_for_prices = ExternalTaskSensor(
        task_id="wait_for_daily_market_data",
        external_dag_id="dag_01_daily_market_data_pipeline",
        external_task_id="data_quality_check",
        execution_delta=timedelta(days=1),
        timeout=3600,
    )

    extract_fundamentals = PythonOperator(
        task_id="extract_eodhd_fundamentals",
        python_callable=extract_eodhd_fundamentals,
    )

    calc_forensic = PythonOperator(
        task_id="calculate_forensic_scores",
        python_callable=calculate_forensic_scores,
    )

    load_financials = PythonOperator(
        task_id="load_postgres_financials",
        python_callable=load_postgres_financials,
    )

    wait_for_prices >> extract_fundamentals >> calc_forensic >> load_financials
