"""
DAG 11: Daily Data Quality Monitoring
Schedule: Every day at 11:00 PM HKT
Purpose: Monitor data freshness and quality across all databases
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import config

def check_duckdb_data_freshness(**context):
    """Verify latest prices updated today and check ticker coverage"""
    import duckdb
    from datetime import datetime
    
    execution_date = context['ds']
    
    issues = []
    
    # Check prices database
    conn = duckdb.connect(config.DUCKDB_PRICES_DB)
    
    # Check if data exists for today
    result = conn.execute(
        "SELECT COUNT(DISTINCT ticker) as ticker_count FROM prices WHERE date = ?",
        [execution_date]
    ).fetchone()
    
    if result:
        ticker_count = result[0]
        expected_count = len(config.DEFAULT_TICKERS)
        coverage_pct = ticker_count / expected_count if expected_count > 0 else 0
        
        if coverage_pct < config.MIN_TICKER_COVERAGE_PCT:
            issues.append({
                'database': 'duckdb_prices',
                'issue': 'low_coverage',
                'message': f"Only {ticker_count}/{expected_count} tickers ({coverage_pct:.1%}) have data",
                'severity': 'high'
            })
    else:
        issues.append({
            'database': 'duckdb_prices',
            'issue': 'no_data',
            'message': f"No price data found for {execution_date}",
            'severity': 'critical'
        })
    
    # Check for missing tickers
    missing_tickers = conn.execute("""
        SELECT ticker FROM (
            SELECT UNNEST($tickers) as ticker
        ) expected
        WHERE ticker NOT IN (
            SELECT DISTINCT ticker FROM prices WHERE date = $date
        )
    """, {"tickers": config.DEFAULT_TICKERS, "date": execution_date}).fetchall()
    
    if missing_tickers:
        issues.append({
            'database': 'duckdb_prices',
            'issue': 'missing_tickers',
            'message': f"Missing data for: {', '.join([t[0] for t in missing_tickers])}",
            'severity': 'medium'
        })
    
    conn.close()
    
    context['ti'].xcom_push(key='duckdb_issues', value=issues)
    print(f"DuckDB check: Found {len(issues)} issues")
    return len(issues) == 0

def check_postgres_data_integrity(**context):
    """Verify no NULL values and financial statements balance"""
    import psycopg2
    
    issues = []
    
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Check for NULL values in critical columns
    cur.execute("""
        SELECT COUNT(*) FROM news_articles
        WHERE ticker IS NULL OR publish_date IS NULL OR headline IS NULL
    """)
    null_count = cur.fetchone()[0]
    
    if null_count > 0:
        issues.append({
            'database': 'postgres',
            'table': 'news_articles',
            'issue': 'null_values',
            'message': f"{null_count} records with NULL values",
            'severity': 'medium'
        })
    
    # Check financial statements balance (Assets = Liabilities + Equity)
    cur.execute("""
        SELECT ticker, fiscal_year, fiscal_period
        FROM (
            SELECT ticker, fiscal_year, fiscal_period,
                   SUM(CASE WHEN line_item LIKE '%Asset%' THEN value ELSE 0 END) as assets,
                   SUM(CASE WHEN line_item LIKE '%Liability%' OR line_item LIKE '%Equity%' THEN value ELSE 0 END) as liabilities_equity
            FROM financial_statements
            WHERE statement_type = 'Balance Sheet'
            GROUP BY ticker, fiscal_year, fiscal_period
        ) t
        WHERE ABS(assets - liabilities_equity) > 1000  -- Allow $1000 rounding difference
    """)
    unbalanced = cur.fetchall()
    
    if unbalanced:
        issues.append({
            'database': 'postgres',
            'table': 'financial_statements',
            'issue': 'unbalanced_sheets',
            'message': f"{len(unbalanced)} balance sheets don't balance",
            'severity': 'high'
        })
    
    cur.close()
    conn.close()
    
    context['ti'].xcom_push(key='postgres_issues', value=issues)
    print(f"Postgres check: Found {len(issues)} issues")
    return len(issues) == 0

def check_qdrant_index_health(**context):
    """Verify collection sizes and check for stale documents"""
    from qdrant_client import QdrantClient
    from datetime import datetime, timedelta
    
    issues = []
    
    # Connect to Qdrant
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    # Check each collection
    for collection_key, collection_name in config.QDRANT_COLLECTIONS.items():
        try:
            collection_info = client.get_collection(collection_name)
            vector_count = collection_info.points_count
            
            # Check if collection is empty (might be expected for some)
            if vector_count == 0:
                issues.append({
                    'database': 'qdrant',
                    'collection': collection_name,
                    'issue': 'empty_collection',
                    'message': f"Collection has 0 vectors",
                    'severity': 'low'
                })
            
            # Check for stale documents (would need to query by date field)
            # Placeholder - real implementation would check ingestion timestamps
            
        except Exception as e:
            issues.append({
                'database': 'qdrant',
                'collection': collection_name,
                'issue': 'collection_error',
                'message': str(e),
                'severity': 'high'
            })
    
    context['ti'].xcom_push(key='qdrant_issues', value=issues)
    print(f"Qdrant check: Found {len(issues)} issues")
    return len(issues) == 0

def check_neo4j_graph_consistency(**context):
    """Verify no orphaned nodes and centrality scores computed"""
    from neo4j import GraphDatabase
    
    issues = []
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Check for orphaned Company nodes (no relationships)
        result = session.run("""
            MATCH (c:Company)
            WHERE NOT (c)--() 
            RETURN COUNT(c) as orphan_count
        """)
        orphan_count = result.single()['orphan_count']
        
        if orphan_count > 0:
            issues.append({
                'database': 'neo4j',
                'graph': 'supply_chain',
                'issue': 'orphaned_nodes',
                'message': f"{orphan_count} Company nodes have no relationships",
                'severity': 'medium'
            })
        
        # Check if centrality scores are computed
        result = session.run("""
            MATCH (c:Company)
            WHERE c.pagerank IS NULL
            RETURN COUNT(c) as missing_scores
        """)
        missing_scores = result.single()['missing_scores']
        
        if missing_scores > 0:
            issues.append({
                'database': 'neo4j',
                'graph': 'supply_chain',
                'issue': 'missing_centrality',
                'message': f"{missing_scores} nodes missing PageRank scores",
                'severity': 'low'
            })
    
    driver.close()
    
    context['ti'].xcom_push(key='neo4j_issues', value=issues)
    print(f"Neo4j check: Found {len(issues)} issues")
    return len(issues) == 0

def generate_data_quality_report(**context):
    """Summary metrics and alert generation"""
    ti = context['ti']
    
    duckdb_issues = ti.xcom_pull(key='duckdb_issues', task_ids='check_duckdb_data_freshness') or []
    postgres_issues = ti.xcom_pull(key='postgres_issues', task_ids='check_postgres_data_integrity') or []
    qdrant_issues = ti.xcom_pull(key='qdrant_issues', task_ids='check_qdrant_index_health') or []
    neo4j_issues = ti.xcom_pull(key='neo4j_issues', task_ids='check_neo4j_graph_consistency') or []
    
    all_issues = duckdb_issues + postgres_issues + qdrant_issues + neo4j_issues
    
    # Count by severity
    critical = [i for i in all_issues if i.get('severity') == 'critical']
    high = [i for i in all_issues if i.get('severity') == 'high']
    medium = [i for i in all_issues if i.get('severity') == 'medium']
    low = [i for i in all_issues if i.get('severity') == 'low']
    
    report = {
        'date': context['ds'],
        'total_issues': len(all_issues),
        'critical_count': len(critical),
        'high_count': len(high),
        'medium_count': len(medium),
        'low_count': len(low),
        'issues': all_issues,
        'status': 'FAIL' if len(critical) + len(high) > 0 else ('WARNING' if len(medium) > 0 else 'PASS')
    }
    
    # Print summary
    print("\n" + "="*70)
    print(f"Data Quality Report - {report['date']}")
    print("="*70)
    print(f"Status: {report['status']}")
    print(f"Total Issues: {report['total_issues']}")
    print(f"  - Critical: {report['critical_count']}")
    print(f"  - High:     {report['high_count']}")
    print(f"  - Medium:   {report['medium_count']}")
    print(f"  - Low:      {report['low_count']}")
    
    if all_issues:
        print("\nIssues:")
        for issue in all_issues:
            print(f"  [{issue['severity'].upper()}] {issue['database']}: {issue['message']}")
    
    print("="*70 + "\n")
    
    context['ti'].xcom_push(key='quality_report', value=report)
    
    # Send alert if critical or high severity issues
    if len(critical) + len(high) > 0:
        print("⚠️  ALERT: Critical or high severity issues detected!")
        # Would send email/Slack notification here
    
    return True

def trigger_backfill_dags(**context):
    """Trigger backfill DAGs if missing data detected"""
    from airflow.api.common.trigger_dag import trigger_dag
    
    ti = context['ti']
    report = ti.xcom_pull(key='quality_report', task_ids='generate_data_quality_report')
    
    if not report:
        return True
    
    # Check for specific issues that require backfill
    issues = report.get('issues', [])
    
    for issue in issues:
        if issue.get('issue') == 'no_data' and issue.get('database') == 'duckdb_prices':
            print("Triggering backfill for daily market data...")
            # trigger_dag('dag_01_daily_market_data_pipeline', run_id=f"backfill_{context['ds']}")
    
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({
    "start_date": datetime(2026, 1, 1),
    "retries": 1,  # Monitoring should not retry too much
})

with DAG(
    dag_id="dag_11_daily_data_quality_monitoring",
    default_args=default_args,
    description="Monitor data freshness and quality across all databases",
    schedule_interval="0 23 * * *",  # 11 PM HKT daily
    catchup=False,
    max_active_runs=1,
    tags=["daily", "monitoring", "data_quality"],
) as dag:

    check_duckdb = PythonOperator(
        task_id="check_duckdb_data_freshness",
        python_callable=check_duckdb_data_freshness,
    )

    check_postgres = PythonOperator(
        task_id="check_postgres_data_integrity",
        python_callable=check_postgres_data_integrity,
    )

    check_qdrant = PythonOperator(
        task_id="check_qdrant_index_health",
        python_callable=check_qdrant_index_health,
    )

    check_neo4j = PythonOperator(
        task_id="check_neo4j_graph_consistency",
        python_callable=check_neo4j_graph_consistency,
    )

    generate_report = PythonOperator(
        task_id="generate_data_quality_report",
        python_callable=generate_data_quality_report,
    )

    trigger_backfill = PythonOperator(
        task_id="trigger_backfill_dags",
        python_callable=trigger_backfill_dags,
    )

    [check_duckdb, check_postgres, check_qdrant, check_neo4j] >> generate_report >> trigger_backfill
