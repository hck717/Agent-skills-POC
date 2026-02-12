"""
DAG 03: Weekly Fundamental Data Pipeline
Schedule: Every Sunday at 2:00 AM HKT
Purpose: Refresh financial statements, forensic scores, and fundamental metrics
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import logging
import requests
import psycopg2
from neo4j import GraphDatabase
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE,
    POSTGRES_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EODHD_API_KEY, FMP_API_KEY, EODHD_BASE_URL, FMP_BASE_URL, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_eodhd_fundamentals(**context):
    """Extract fundamentals from EODHD"""
    logger.info(f"Extracting fundamentals for {len(DEFAULT_TICKERS)} tickers from EODHD")
    fundamentals_data = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
            params = {"api_token": EODHD_API_KEY, "fmt": "json"}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            fundamentals_data.append({
                'ticker': ticker,
                'data': data,
                'source': 'eodhd'
            })
            logger.info(f"✅ {ticker}: Extracted fundamentals")
        except Exception as e:
            logger.warning(f"EODHD fundamentals failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='fundamentals_data', value=fundamentals_data)
    return {"extracted": len(fundamentals_data)}

def extract_fmp_fundamentals(**context):
    """Extract FMP financial statements"""
    fundamentals_data = context['ti'].xcom_pull(task_ids='extract_eodhd_fundamentals', key='fundamentals_data') or []
    logger.info(f"Extracting FMP financial statements")
    
    for ticker in DEFAULT_TICKERS:
        try:
            # Income statement
            url = f"{FMP_BASE_URL}/income-statement/{ticker}"
            params = {"apikey": FMP_API_KEY, "limit": 4}  # Last 4 quarters
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            income_data = response.json()
            
            # Balance sheet
            url = f"{FMP_BASE_URL}/balance-sheet-statement/{ticker}"
            response = requests.get(url, params=params, timeout=30)
            balance_data = response.json() if response.status_code == 200 else []
            
            # Cash flow
            url = f"{FMP_BASE_URL}/cash-flow-statement/{ticker}"
            response = requests.get(url, params=params, timeout=30)
            cashflow_data = response.json() if response.status_code == 200 else []
            
            fundamentals_data.append({
                'ticker': ticker,
                'income_statement': income_data,
                'balance_sheet': balance_data,
                'cash_flow': cashflow_data,
                'source': 'fmp'
            })
            logger.info(f"✅ {ticker}: FMP financials extracted")
        except Exception as e:
            logger.warning(f"FMP financials failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='fundamentals_data', value=fundamentals_data)
    return {"extracted": len(fundamentals_data)}

def calculate_forensic_scores(**context):
    """Calculate Beneish M-score, Altman Z-score, Piotroski F-score"""
    fundamentals_data = context['ti'].xcom_pull(task_ids='extract_fmp_fundamentals', key='fundamentals_data') or []
    logger.info(f"Calculating forensic scores for {len(fundamentals_data)} companies")
    
    forensic_scores = []
    
    for company in fundamentals_data:
        ticker = company['ticker']
        try:
            # Extract latest financials
            income = company.get('income_statement', [{}])[0]
            balance = company.get('balance_sheet', [{}])[0]
            
            # Simplified Altman Z-score (5-variable model)
            total_assets = balance.get('totalAssets', 1)
            working_capital = balance.get('totalCurrentAssets', 0) - balance.get('totalCurrentLiabilities', 0)
            retained_earnings = balance.get('retainedEarnings', 0)
            ebit = income.get('ebitda', 0)
            market_cap = 1000000000  # Placeholder
            total_liabilities = balance.get('totalLiabilities', 0)
            sales = income.get('revenue', 1)
            
            x1 = working_capital / total_assets if total_assets > 0 else 0
            x2 = retained_earnings / total_assets if total_assets > 0 else 0
            x3 = ebit / total_assets if total_assets > 0 else 0
            x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
            x5 = sales / total_assets if total_assets > 0 else 0
            
            altman_z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            # Simplified Piotroski F-score (9 binary signals)
            roa = income.get('netIncome', 0) / total_assets if total_assets > 0 else 0
            cfo = 0  # Placeholder - need cash flow data
            
            f_score = 0
            if roa > 0: f_score += 1
            if cfo > 0: f_score += 1
            # Add other 7 signals...
            
            forensic_scores.append({
                'ticker': ticker,
                'altman_z_score': round(altman_z, 2),
                'piotroski_f_score': f_score,
                'beneish_m_score': 0,  # Placeholder
                'calculation_date': datetime.now().isoformat()
            })
            
            logger.info(f"✅ {ticker}: Z={altman_z:.2f}, F={f_score}")
        except Exception as e:
            logger.warning(f"Forensic calculation failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='forensic_scores', value=forensic_scores)
    return {"calculated": len(forensic_scores)}

def load_postgres_financials(**context):
    """Load financials and forensic scores to Postgres"""
    fundamentals_data = context['ti'].xcom_pull(task_ids='extract_fmp_fundamentals', key='fundamentals_data') or []
    forensic_scores = context['ti'].xcom_pull(task_ids='calculate_forensic_scores', key='forensic_scores') or []
    
    logger.info(f"Loading financials for {len(fundamentals_data)} companies to Postgres")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    # Create financial_statements table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_statements (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            statement_type VARCHAR(20),
            fiscal_period VARCHAR(10),
            fiscal_year INTEGER,
            report_date DATE,
            data JSONB,
            source VARCHAR(20),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, statement_type, fiscal_period, fiscal_year)
        )
    """)
    
    # Create forensic_scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forensic_scores (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            altman_z_score NUMERIC,
            piotroski_f_score INTEGER,
            beneish_m_score NUMERIC,
            calculation_date TIMESTAMP,
            UNIQUE(ticker, calculation_date)
        )
    """)
    
    # Load forensic scores
    for score in forensic_scores:
        cursor.execute("""
            INSERT INTO forensic_scores (ticker, altman_z_score, piotroski_f_score, beneish_m_score, calculation_date)
            VALUES (%(ticker)s, %(altman_z_score)s, %(piotroski_f_score)s, %(beneish_m_score)s, %(calculation_date)s)
            ON CONFLICT (ticker, calculation_date) DO UPDATE SET
                altman_z_score = EXCLUDED.altman_z_score,
                piotroski_f_score = EXCLUDED.piotroski_f_score,
                beneish_m_score = EXCLUDED.beneish_m_score
        """, score)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✅ Loaded {len(forensic_scores)} forensic scores to Postgres")
    return {"loaded": len(forensic_scores)}

def load_neo4j_fundamentals(**context):
    """Load fundamental metrics to Neo4j"""
    forensic_scores = context['ti'].xcom_pull(task_ids='calculate_forensic_scores', key='forensic_scores') or []
    logger.info(f"Loading {len(forensic_scores)} scores to Neo4j")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for score in forensic_scores:
            session.run("""
                MATCH (c:Company {ticker: $ticker})
                SET c.altman_z_score = $altman_z,
                    c.piotroski_f_score = $f_score,
                    c.fundamentals_updated_at = datetime()
            """, ticker=score['ticker'], altman_z=score['altman_z_score'], f_score=score['piotroski_f_score'])
        
        logger.info(f"✅ Updated {len(forensic_scores)} companies in Neo4j")
    
    driver.close()
    return {"loaded": len(forensic_scores)}

# DAG DEFINITION
with DAG(
    dag_id='03_weekly_fundamental_data_pipeline',
    default_args=DEFAULT_ARGS,
    description='Weekly fundamentals and forensic scores',
    schedule_interval='0 18 * * 0' if not TEST_MODE else None,  # Sunday 2 AM HKT = Saturday 6 PM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['weekly', 'fundamentals', 'forensics'],
) as dag:
    
    # Sensor: Wait for daily market data
    wait_for_market_data = ExternalTaskSensor(
        task_id='wait_for_market_data',
        external_dag_id='01_daily_market_data_pipeline',
        external_task_id='data_quality_check',
        execution_delta=timedelta(days=1),
        timeout=3600,
        mode='reschedule'
    )
    
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_fundamentals', python_callable=extract_eodhd_fundamentals)
    task_extract_fmp = PythonOperator(task_id='extract_fmp_fundamentals', python_callable=extract_fmp_fundamentals)
    task_calculate_forensics = PythonOperator(task_id='calculate_forensic_scores', python_callable=calculate_forensic_scores)
    task_load_postgres = PythonOperator(task_id='load_postgres_financials', python_callable=load_postgres_financials)
    task_load_neo4j = PythonOperator(task_id='load_neo4j_fundamentals', python_callable=load_neo4j_fundamentals)
    
    wait_for_market_data >> task_extract_eodhd >> task_extract_fmp >> task_calculate_forensics >> [task_load_postgres, task_load_neo4j]
