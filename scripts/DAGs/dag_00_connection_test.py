"""
DAG 00: Connection Test - Verify all cloud services are accessible
Run this first to ensure Postgres, Neo4j, Qdrant connections work
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

# Import shared config
import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS,
    POSTGRES_URL, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY,
    EODHD_API_KEY, FMP_API_KEY, TAVILY_API_KEY,
    DEPLOYMENT_MODE
)

logger = logging.getLogger(__name__)

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_postgres_connection(**context):
    """Test PostgreSQL connection (Neon/Supabase)"""
    import psycopg2
    from psycopg2 import sql
    
    logger.info(f"Testing Postgres connection to {POSTGRES_HOST}:{POSTGRES_PORT}")
    
    try:
        # Connect to database (no timeout)
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✅ Postgres connected successfully!")
        logger.info(f"   Version: {version}")
        
        # Test create table (will be used by other DAGs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connection_test (
                test_id SERIAL PRIMARY KEY,
                test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT
            );
        """)
        
        # Insert test record
        cursor.execute("""
            INSERT INTO connection_test (status) 
            VALUES ('Airflow connection test successful')
            RETURNING test_id, test_time;
        """)
        test_id, test_time = cursor.fetchone()
        logger.info(f"   Test record inserted: ID={test_id}, Time={test_time}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"status": "success", "version": version, "test_id": test_id}
        
    except Exception as e:
        logger.error(f"❌ Postgres connection failed: {str(e)}")
        raise


def test_neo4j_connection(**context):
    """Test Neo4j Aura connection"""
    from neo4j import GraphDatabase
    
    logger.info(f"Testing Neo4j connection to {NEO4J_URI}")
    
    try:
        # Connect to Neo4j (no timeout)
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Verify connection
        driver.verify_connectivity()
        logger.info("✅ Neo4j connected successfully!")
        
        # Test query
        with driver.session() as session:
            result = session.run("""
                CALL dbms.components() 
                YIELD name, versions, edition 
                RETURN name, versions[0] as version, edition
            """)
            record = result.single()
            logger.info(f"   Version: {record['version']}")
            logger.info(f"   Edition: {record['edition']}")
            
            # Create test node
            result = session.run("""
                MERGE (t:ConnectionTest {name: 'Airflow Test', timestamp: datetime()})
                RETURN t.name as name, t.timestamp as timestamp
            """)
            record = result.single()
            logger.info(f"   Test node created: {record['name']} at {record['timestamp']}")
        
        driver.close()
        return {"status": "success", "uri": NEO4J_URI}
        
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {str(e)}")
        raise


def test_qdrant_connection(**context):
    """Test Qdrant Cloud connection with NO timeout"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    
    logger.info(f"Testing Qdrant connection to {QDRANT_URL}")
    
    try:
        # Connect to Qdrant with NO timeout (timeout=None means wait forever)
        if QDRANT_API_KEY:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=None)
        else:
            client = QdrantClient(url=QDRANT_URL, timeout=None)
        
        logger.info("✅ Qdrant connected successfully!")
        
        # List existing collections
        collections = client.get_collections()
        logger.info(f"   Existing collections: {len(collections.collections)}")
        for col in collections.collections:
            logger.info(f"     - {col.name}")
        
        # Create test collection (if doesn't exist)
        test_collection = "airflow_connection_test"
        try:
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            logger.info(f"   Test collection '{test_collection}' created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"   Test collection '{test_collection}' already exists")
            else:
                raise
        
        # Get collection info
        info = client.get_collection(test_collection)
        logger.info(f"   Collection vectors: {info.vectors_count}")
        
        return {"status": "success", "collections": len(collections.collections)}
        
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {str(e)}")
        raise


def test_api_keys(**context):
    """Test API keys are loaded"""
    import requests
    
    logger.info("Testing API keys...")
    
    results = {}
    
    # Test EODHD API (no timeout)
    if EODHD_API_KEY:
        try:
            url = f"https://eodhistoricaldata.com/api/eod/AAPL.US?api_token={EODHD_API_KEY}&fmt=json&limit=1"
            response = requests.get(url, timeout=None)
            if response.status_code == 200:
                logger.info("✅ EODHD API key valid")
                results['eodhd'] = 'valid'
            else:
                logger.warning(f"⚠️ EODHD API returned status {response.status_code}")
                results['eodhd'] = 'invalid'
        except Exception as e:
            logger.error(f"❌ EODHD API test failed: {str(e)}")
            results['eodhd'] = 'error'
    else:
        logger.warning("⚠️ EODHD_API_KEY not set")
        results['eodhd'] = 'missing'
    
    # Test FMP API (optional - gracefully handle 403, no timeout)
    if FMP_API_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=None)
            if response.status_code == 200 and response.json():
                logger.info("✅ FMP API key valid")
                results['fmp'] = 'valid'
            elif response.status_code == 403:
                logger.warning("⚠️ FMP API key exists but access denied (403) - may need paid tier")
                results['fmp'] = 'limited_access'
            else:
                logger.warning(f"⚠️ FMP API returned status {response.status_code}")
                results['fmp'] = 'invalid'
        except Exception as e:
            logger.error(f"❌ FMP API test failed: {str(e)}")
            results['fmp'] = 'error'
    else:
        logger.warning("⚠️ FMP_API_KEY not set (optional)")
        results['fmp'] = 'missing'
    
    # Test Tavily API
    if TAVILY_API_KEY:
        logger.info("✅ TAVILY_API_KEY loaded (not testing actual API)")
        results['tavily'] = 'loaded'
    else:
        logger.warning("⚠️ TAVILY_API_KEY not set")
        results['tavily'] = 'missing'
    
    logger.info(f"API Key Test Results: {results}")
    return results


def print_summary(**context):
    """Print test summary"""
    ti = context['ti']
    
    postgres_result = ti.xcom_pull(task_ids='test_postgres')
    neo4j_result = ti.xcom_pull(task_ids='test_neo4j')
    qdrant_result = ti.xcom_pull(task_ids='test_qdrant')
    api_result = ti.xcom_pull(task_ids='test_api_keys')
    
    logger.info("\n" + "="*70)
    logger.info("CONNECTION TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Deployment Mode: {DEPLOYMENT_MODE.upper()}")
    logger.info("")
    logger.info(f"✅ PostgreSQL: {postgres_result['status']}")
    logger.info(f"✅ Neo4j: {neo4j_result['status']}")
    logger.info(f"✅ Qdrant: {qdrant_result['status']} ({qdrant_result['collections']} collections)")
    logger.info("")
    logger.info("API Keys:")
    for api, status in api_result.items():
        emoji = "✅" if status in ['valid', 'loaded', 'limited_access'] else "⚠️"
        logger.info(f"  {emoji} {api.upper()}: {status}")
    logger.info("="*70)
    logger.info("🎉 All connection tests completed!")
    logger.info("="*70 + "\n")

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    dag_id='00_connection_test',
    default_args=DEFAULT_ARGS,
    description='Test all cloud database connections and API keys',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['test', 'setup', 'diagnostics'],
) as dag:
    
    test_postgres = PythonOperator(
        task_id='test_postgres',
        python_callable=test_postgres_connection,
        provide_context=True,
    )
    
    test_neo4j = PythonOperator(
        task_id='test_neo4j',
        python_callable=test_neo4j_connection,
        provide_context=True,
    )
    
    test_qdrant = PythonOperator(
        task_id='test_qdrant',
        python_callable=test_qdrant_connection,
        provide_context=True,
    )
    
    test_apis = PythonOperator(
        task_id='test_api_keys',
        python_callable=test_api_keys,
        provide_context=True,
    )
    
    summary = PythonOperator(
        task_id='print_summary',
        python_callable=print_summary,
        provide_context=True,
    )
    
    # Run all tests in parallel, then print summary
    [test_postgres, test_neo4j, test_qdrant, test_apis] >> summary
