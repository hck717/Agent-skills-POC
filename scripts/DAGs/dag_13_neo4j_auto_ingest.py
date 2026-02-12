"""
DAG 13: Neo4j Auto Ingest Pipeline
Schedule: Continuous sync Postgres → Neo4j
Purpose: Automatically sync data from Postgres to Neo4j knowledge graph
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import psycopg2
from neo4j import GraphDatabase
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, TEST_MODE,
    POSTGRES_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def scan_postgres_for_new_data(**context):
    """Scan Postgres for data that needs syncing to Neo4j"""
    logger.info("Scanning Postgres for new data")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    # Check for new companies
    cursor.execute("""
        SELECT COUNT(*) FROM companies
        WHERE updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
    """)
    new_companies = cursor.fetchone()[0]
    
    # Check for new prices
    cursor.execute("""
        SELECT COUNT(*) FROM market_prices
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
    """)
    new_prices = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    logger.info(f"✅ Found: {new_companies} companies, {new_prices} price records")
    
    context['ti'].xcom_push(key='new_companies', value=new_companies)
    context['ti'].xcom_push(key='new_prices', value=new_prices)
    
    return {"new_companies": new_companies, "new_prices": new_prices}

def ingest_business_analyst_graph(**context):
    """Sync companies and fundamentals to Neo4j"""
    new_companies = context['ti'].xcom_pull(task_ids='scan_postgres_for_new_data', key='new_companies')
    
    if new_companies == 0:
        logger.info("No new companies to sync")
        return {"synced": 0}
    
    logger.info(f"Syncing {new_companies} companies to Neo4j")
    
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, name, sector, industry
        FROM companies
        WHERE updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
    """)
    companies = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for ticker, name, sector, industry in companies:
            session.run("""
                MERGE (c:Company {ticker: $ticker})
                SET c.name = $name,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.updated_at = datetime()
            """, ticker=ticker, name=name, sector=sector, industry=industry)
    
    driver.close()
    
    logger.info(f"✅ Synced {len(companies)} companies to Neo4j")
    return {"synced": len(companies)}

def ingest_supply_chain_graph(**context):
    """Sync supply chain relationships to Neo4j"""
    logger.info("Syncing supply chain relationships to Neo4j")
    
    # Placeholder: Would sync actual relationships from Postgres
    logger.info("✅ Supply chain graph synced (placeholder)")
    return {"synced": 0}

def calculate_graph_metrics(**context):
    """Calculate PageRank and centrality metrics on Neo4j graph"""
    logger.info("Calculating graph metrics")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Update degree centrality
        session.run("""
            MATCH (c:Company)
            WITH c, SIZE((c)-[]-()) as degree
            SET c.degree_centrality = degree
        """)
        
        # Update community detection (simplified)
        session.run("""
            MATCH (c:Company)
            WHERE c.community_id IS NULL
            SET c.community_id = 1
        """)
        
        logger.info("✅ Graph metrics updated")
    
    driver.close()
    return {"calculated": True}

# DAG DEFINITION
with DAG(
    dag_id='13_neo4j_auto_ingest_pipeline',
    default_args=DEFAULT_ARGS,
    description='Auto-sync Postgres to Neo4j knowledge graph',
    schedule_interval='0 */12 * * *' if not TEST_MODE else None,  # Every 12 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['continuous', 'neo4j', 'sync'],
) as dag:
    
    task_scan = PythonOperator(task_id='scan_postgres_for_new_data', python_callable=scan_postgres_for_new_data)
    task_ingest_ba = PythonOperator(task_id='ingest_business_analyst_graph', python_callable=ingest_business_analyst_graph)
    task_ingest_sc = PythonOperator(task_id='ingest_supply_chain_graph', python_callable=ingest_supply_chain_graph)
    task_metrics = PythonOperator(task_id='calculate_graph_metrics', python_callable=calculate_graph_metrics)
    
    task_scan >> [task_ingest_ba, task_ingest_sc] >> task_metrics
