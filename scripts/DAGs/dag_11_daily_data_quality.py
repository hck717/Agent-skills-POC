"""
DAG 11: Daily Data Quality Monitoring
Schedule: Every day at 11:00 PM HKT
Purpose: Monitor data freshness and integrity across all databases
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import psycopg2
import duckdb
from neo4j import GraphDatabase

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE,
    POSTGRES_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTIONS,
    DUCKDB_PRICES_DB, DUCKDB_MACRO_DB, MAX_ACTIVE_RUNS, MIN_TICKER_COVERAGE_PCT
)

logger = logging.getLogger(__name__)

def check_postgres_data_integrity(**context):
    """Verify Postgres data quality"""
    logger.info("Checking Postgres data integrity")
    
    issues = []
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    
    # Check 1: NULL values in critical columns
    cursor.execute("""
        SELECT COUNT(*) FROM market_prices
        WHERE close IS NULL OR volume IS NULL
        AND date >= CURRENT_DATE - INTERVAL '7 days'
    """)
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        issues.append(f"Found {null_count} NULL prices in last 7 days")
    
    # Check 2: Ticker coverage
    cursor.execute("""
        SELECT COUNT(DISTINCT ticker) FROM market_prices
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
    """)
    ticker_count = cursor.fetchone()[0]
    expected = len(DEFAULT_TICKERS)
    coverage = ticker_count / expected
    
    if coverage < MIN_TICKER_COVERAGE_PCT:
        issues.append(f"Ticker coverage {coverage:.1%} < {MIN_TICKER_COVERAGE_PCT:.1%}")
    
    cursor.close()
    conn.close()
    
    if issues:
        logger.error(f"❌ Postgres issues: {issues}")
    else:
        logger.info("✅ Postgres data integrity OK")
    
    context['ti'].xcom_push(key='postgres_issues', value=issues)
    return {"issues": len(issues)}

def check_neo4j_graph_consistency(**context):
    """Verify Neo4j graph consistency"""
    logger.info("Checking Neo4j graph consistency")
    
    issues = []
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Check 1: Orphaned nodes
        result = session.run("""
            MATCH (c:Company)
            WHERE NOT (c)-[]->()
            RETURN COUNT(c) as orphan_count
        """)
        orphan_count = result.single()['orphan_count']
        
        if orphan_count > 0:
            issues.append(f"Found {orphan_count} orphaned Company nodes")
        
        # Check 2: Missing centrality scores
        result = session.run("""
            MATCH (c:Company)
            WHERE c.pagerank_score IS NULL
            RETURN COUNT(c) as missing_scores
        """)
        missing = result.single()['missing_scores']
        
        if missing > 0:
            issues.append(f"{missing} companies missing centrality scores")
    
    driver.close()
    
    if issues:
        logger.error(f"❌ Neo4j issues: {issues}")
    else:
        logger.info("✅ Neo4j graph consistency OK")
    
    context['ti'].xcom_push(key='neo4j_issues', value=issues)
    return {"issues": len(issues)}

def check_qdrant_index_health(**context):
    """Verify Qdrant collection health"""
    logger.info("Checking Qdrant index health")
    
    issues = []
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        for name, collection_name in QDRANT_COLLECTIONS.items():
            try:
                info = client.get_collection(collection_name)
                point_count = info.points_count if hasattr(info, 'points_count') else 0
                
                logger.info(f"✅ {collection_name}: {point_count} vectors")
                
                if point_count == 0:
                    issues.append(f"{collection_name} is empty")
            except Exception as e:
                issues.append(f"{collection_name} not accessible: {str(e)}")
    
    except Exception as e:
        issues.append(f"Qdrant connection failed: {str(e)}")
    
    if issues:
        logger.error(f"❌ Qdrant issues: {issues}")
    else:
        logger.info("✅ Qdrant index health OK")
    
    context['ti'].xcom_push(key='qdrant_issues', value=issues)
    return {"issues": len(issues)}

def generate_data_quality_report(**context):
    """Generate summary report and send alerts"""
    postgres_issues = context['ti'].xcom_pull(task_ids='check_postgres_data_integrity', key='postgres_issues') or []
    neo4j_issues = context['ti'].xcom_pull(task_ids='check_neo4j_graph_consistency', key='neo4j_issues') or []
    qdrant_issues = context['ti'].xcom_pull(task_ids='check_qdrant_index_health', key='qdrant_issues') or []
    
    total_issues = len(postgres_issues) + len(neo4j_issues) + len(qdrant_issues)
    
    logger.info("="*70)
    logger.info("DATA QUALITY REPORT")
    logger.info("="*70)
    logger.info(f"Postgres: {len(postgres_issues)} issues")
    for issue in postgres_issues:
        logger.warning(f"  - {issue}")
    
    logger.info(f"Neo4j: {len(neo4j_issues)} issues")
    for issue in neo4j_issues:
        logger.warning(f"  - {issue}")
    
    logger.info(f"Qdrant: {len(qdrant_issues)} issues")
    for issue in qdrant_issues:
        logger.warning(f"  - {issue}")
    
    logger.info("="*70)
    
    if total_issues == 0:
        logger.info("✅ ALL QUALITY CHECKS PASSED!")
    else:
        logger.error(f"❌ {total_issues} QUALITY ISSUES DETECTED")
    
    logger.info("="*70)
    
    return {"total_issues": total_issues}

# DAG DEFINITION
with DAG(
    dag_id='11_daily_data_quality_monitoring',
    default_args=DEFAULT_ARGS,
    description='Daily data quality monitoring across all databases',
    schedule_interval='0 15 * * *' if not TEST_MODE else None,  # 11 PM HKT
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['daily', 'quality', 'monitoring'],
) as dag:
    
    task_check_postgres = PythonOperator(task_id='check_postgres_data_integrity', python_callable=check_postgres_data_integrity)
    task_check_neo4j = PythonOperator(task_id='check_neo4j_graph_consistency', python_callable=check_neo4j_graph_consistency)
    task_check_qdrant = PythonOperator(task_id='check_qdrant_index_health', python_callable=check_qdrant_index_health)
    task_report = PythonOperator(task_id='generate_data_quality_report', python_callable=generate_data_quality_report)
    
    [task_check_postgres, task_check_neo4j, task_check_qdrant] >> task_report
