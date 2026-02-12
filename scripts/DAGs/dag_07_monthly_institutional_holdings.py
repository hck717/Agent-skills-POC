"""
DAG 07: Monthly Institutional Holdings Pipeline
Schedule: 15th of every month (after 13F filing deadline)
Purpose: Update ownership graph with institutional holdings
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
    EODHD_API_KEY, FMP_API_KEY, EODHD_BASE_URL, FMP_BASE_URL, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_fmp_13f_filings(**context):
    """Extract latest 13F filings from FMP"""
    logger.info("Extracting 13F institutional holdings")
    holdings_data = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{FMP_BASE_URL}/institutional-holder/{ticker}"
            params = {"apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for holder in data[:20]:  # Top 20 institutions
                holdings_data.append({
                    'ticker': ticker,
                    'institution_name': holder.get('holder'),
                    'shares': holder.get('shares'),
                    'value': holder.get('value'),
                    'change_pct': holder.get('change'),
                    'report_date': holder.get('dateReported'),
                    'source': 'fmp'
                })
            
            logger.info(f"✅ {ticker}: {len(data)} institutional holders")
        except Exception as e:
            logger.warning(f"FMP 13F failed for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='holdings_data', value=holdings_data)
    return {"extracted": len(holdings_data)}

def extract_eodhd_institutional_holders(**context):
    """Extract EODHD institutional holders as backup"""
    holdings_data = context['ti'].xcom_pull(task_ids='extract_fmp_13f_filings', key='holdings_data') or []
    logger.info("Extracting EODHD institutional holders")
    
    # EODHD endpoint placeholder (adjust as needed)
    logger.info(f"✅ Total holdings: {len(holdings_data)}")
    context['ti'].xcom_push(key='holdings_data', value=holdings_data)
    return {"extracted": len(holdings_data)}

def transform_ownership_data(**context):
    """Calculate ownership concentration and detect changes"""
    holdings_data = context['ti'].xcom_pull(task_ids='extract_eodhd_institutional_holders', key='holdings_data') or []
    logger.info(f"Transforming {len(holdings_data)} holdings records")
    
    df = pd.DataFrame(holdings_data)
    if df.empty:
        context['ti'].xcom_push(key='transformed_data', value=[])
        return {"transformed": 0}
    
    # Calculate ownership concentration (top 10 institutions)
    df_top10 = df.groupby('ticker').apply(
        lambda x: x.nlargest(10, 'value')['value'].sum()
    ).reset_index()
    df_top10.columns = ['ticker', 'top10_ownership_value']
    
    # Detect significant changes (> 20%)
    df['significant_change'] = df['change_pct'].abs() > 20
    
    transformed_data = df.to_dict('records')
    context['ti'].xcom_push(key='transformed_data', value=transformed_data)
    logger.info(f"✅ Transformed {len(transformed_data)} holdings")
    return {"transformed": len(transformed_data)}

def load_neo4j_ownership_graph(**context):
    """Load institutional ownership to Neo4j graph"""
    transformed_data = context['ti'].xcom_pull(task_ids='transform_ownership_data', key='transformed_data') or []
    
    if not transformed_data:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(transformed_data)} holdings to Neo4j ownership graph")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for holding in transformed_data:
            # Create/update Institution node
            session.run("""
                MERGE (i:Institution {name: $institution_name})
                ON CREATE SET i.created_at = datetime()
            """, institution_name=holding['institution_name'])
            
            # Create HOLDS relationship
            session.run("""
                MATCH (i:Institution {name: $institution_name})
                MATCH (c:Company {ticker: $ticker})
                MERGE (i)-[r:HOLDS]->(c)
                SET r.shares = $shares,
                    r.value = $value,
                    r.change_pct = $change_pct,
                    r.report_date = date($report_date),
                    r.updated_at = datetime()
            """, 
                institution_name=holding['institution_name'],
                ticker=holding['ticker'],
                shares=holding.get('shares', 0),
                value=holding.get('value', 0),
                change_pct=holding.get('change_pct', 0),
                report_date=holding.get('report_date', '2026-01-01')
            )
        
        logger.info(f"✅ Created {len(transformed_data)} HOLDS relationships in Neo4j")
    
    driver.close()
    return {"loaded": len(transformed_data)}

def calculate_ownership_metrics(**context):
    """Calculate ownership concentration metrics in Neo4j"""
    logger.info("Calculating ownership metrics")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Calculate institutional ownership %
        session.run("""
            MATCH (c:Company)<-[r:HOLDS]-(i:Institution)
            WITH c, SUM(r.value) as total_institutional_value
            SET c.institutional_ownership_value = total_institutional_value,
                c.ownership_updated_at = datetime()
        """)
        
        # Calculate number of institutional holders
        session.run("""
            MATCH (c:Company)<-[:HOLDS]-(i:Institution)
            WITH c, COUNT(i) as holder_count
            SET c.institutional_holder_count = holder_count
        """)
        
        logger.info("✅ Ownership metrics calculated")
    
    driver.close()
    return {"calculated": True}

# DAG DEFINITION
with DAG(
    dag_id='07_monthly_institutional_holdings_pipeline',
    default_args=DEFAULT_ARGS,
    description='Monthly institutional holdings and ownership graph',
    schedule_interval='0 19 15 * *' if not TEST_MODE else None,  # 15th of month, 3 AM HKT
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['monthly', 'institutional_holdings', 'neo4j'],
) as dag:
    
    task_extract_fmp = PythonOperator(task_id='extract_fmp_13f_filings', python_callable=extract_fmp_13f_filings)
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_institutional_holders', python_callable=extract_eodhd_institutional_holders)
    task_transform = PythonOperator(task_id='transform_ownership_data', python_callable=transform_ownership_data)
    task_load_neo4j = PythonOperator(task_id='load_neo4j_ownership_graph', python_callable=load_neo4j_ownership_graph)
    task_calculate_metrics = PythonOperator(task_id='calculate_ownership_metrics', python_callable=calculate_ownership_metrics)
    
    task_extract_fmp >> task_extract_eodhd >> task_transform >> task_load_neo4j >> task_calculate_metrics
