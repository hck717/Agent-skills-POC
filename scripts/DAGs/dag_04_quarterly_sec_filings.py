"""
DAG 04: Quarterly SEC Filings Pipeline
Schedule: Triggered by daily news pipeline when new 10-K/10-Q detected
Purpose: Process SEC filings with proposition chunking and knowledge graph extraction
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import logging
import requests
import psycopg2
from neo4j import GraphDatabase

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE, POSTGRES_URL,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTIONS,
    FMP_API_KEY, FMP_BASE_URL, OPENAI_API_KEY, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_fmp_10k_10q_full_text(**context):
    """Extract 10-K/10-Q full text from FMP"""
    logger.info("Extracting 10-K/10-Q filings")
    
    # Check for new filings from news pipeline
    conn = psycopg2.connect(POSTGRES_URL)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT ticker, form_type FROM sec_filings
        WHERE form_type IN ('10-K', '10-Q')
        AND filing_date >= CURRENT_DATE - INTERVAL '7 days'
    """)
    new_filings = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not new_filings:
        logger.info("No new 10-K/10-Q filings to process")
        context['ti'].xcom_push(key='filings_text', value=[])
        return {"extracted": 0}
    
    logger.info(f"Processing {len(new_filings)} new filings")
    filings_text = []
    
    for ticker, form_type in new_filings:
        try:
            url = f"{FMP_BASE_URL}/sec_filings/{ticker}"
            params = {"type": form_type.replace('-', ''), "apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data:
                filing = data[0]  # Most recent
                filings_text.append({
                    'ticker': ticker,
                    'form_type': form_type,
                    'filing_date': filing.get('fillingDate'),
                    'text': filing.get('finalLink', ''),  # URL to full filing
                })
                logger.info(f"✅ {ticker} {form_type} extracted")
        except Exception as e:
            logger.warning(f"Failed to extract {ticker} {form_type}: {str(e)}")
    
    context['ti'].xcom_push(key='filings_text', value=filings_text)
    return {"extracted": len(filings_text)}

def transform_proposition_chunking(**context):
    """LLM-based proposition chunking (simplified)"""
    filings_text = context['ti'].xcom_pull(task_ids='extract_fmp_10k_10q_full_text', key='filings_text') or []
    
    if not filings_text:
        context['ti'].xcom_push(key='propositions', value=[])
        return {"chunks": 0}
    
    logger.info(f"Chunking {len(filings_text)} filings into propositions")
    propositions = []
    
    for filing in filings_text:
        # Simplified: Just chunk by sections (in production, use GPT-4 for atomic propositions)
        sections = ['Business Description', 'Risk Factors', 'MD&A']
        
        for section in sections:
            propositions.append({
                'ticker': filing['ticker'],
                'form_type': filing['form_type'],
                'filing_date': filing['filing_date'],
                'section': section,
                'text': f"Placeholder text for {section}",  # In production: extract actual section
                'proposition_id': f"{filing['ticker']}_{section.replace(' ', '_')}"
            })
    
    context['ti'].xcom_push(key='propositions', value=propositions)
    logger.info(f"✅ Generated {len(propositions)} propositions")
    return {"chunks": len(propositions)}

def load_qdrant_propositions(**context):
    """Load propositions to Qdrant with embeddings"""
    propositions = context['ti'].xcom_pull(task_ids='transform_proposition_chunking', key='propositions') or []
    
    if not propositions:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(propositions)} propositions to Qdrant")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import openai
        
        openai.api_key = OPENAI_API_KEY
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        collection_name = QDRANT_COLLECTIONS['business_analyst_10k']
        
        # Create collection if not exists
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except:
            pass  # Collection already exists
        
        # Generate embeddings and upsert
        for i, prop in enumerate(propositions):
            # Generate embedding
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=prop['text']
            )
            embedding = response.data[0].embedding
            
            # Upsert to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=hash(prop['proposition_id']) % (10 ** 8),
                        vector=embedding,
                        payload={
                            'ticker': prop['ticker'],
                            'form_type': prop['form_type'],
                            'filing_date': prop['filing_date'],
                            'section': prop['section'],
                            'text': prop['text']
                        }
                    )
                ]
            )
        
        logger.info(f"✅ Loaded {len(propositions)} propositions to Qdrant")
        return {"loaded": len(propositions)}
    
    except Exception as e:
        logger.error(f"Qdrant load failed: {str(e)}")
        return {"loaded": 0}

def load_neo4j_knowledge_graph(**context):
    """Extract entities and load to Neo4j knowledge graph"""
    propositions = context['ti'].xcom_pull(task_ids='transform_proposition_chunking', key='propositions') or []
    
    if not propositions:
        return {"loaded": 0}
    
    logger.info(f"Extracting knowledge graph from {len(propositions)} propositions")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for prop in propositions:
            # Simplified: Create generic nodes (in production: use NER + relation extraction)
            session.run("""
                MATCH (c:Company {ticker: $ticker})
                MERGE (s:Strategy {name: $section})
                MERGE (c)-[:HAS_STRATEGY {filing_date: date($filing_date)}]->(s)
            """, ticker=prop['ticker'], section=prop['section'], filing_date=prop['filing_date'])
        
        logger.info(f"✅ Created knowledge graph nodes for {len(propositions)} propositions")
    
    driver.close()
    return {"loaded": len(propositions)}

# DAG DEFINITION
with DAG(
    dag_id='04_quarterly_sec_filings_pipeline',
    default_args=DEFAULT_ARGS,
    description='Process SEC 10-K/10-Q filings with embeddings and knowledge graph',
    schedule_interval=None,  # Triggered by DAG 02
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['quarterly', 'sec_filings', 'qdrant', 'neo4j'],
) as dag:
    
    task_extract = PythonOperator(task_id='extract_fmp_10k_10q_full_text', python_callable=extract_fmp_10k_10q_full_text)
    task_chunk = PythonOperator(task_id='transform_proposition_chunking', python_callable=transform_proposition_chunking)
    task_load_qdrant = PythonOperator(task_id='load_qdrant_propositions', python_callable=load_qdrant_propositions)
    task_load_neo4j = PythonOperator(task_id='load_neo4j_knowledge_graph', python_callable=load_neo4j_knowledge_graph)
    
    task_extract >> task_chunk >> [task_load_qdrant, task_load_neo4j]
