"""
DAG 10: Quarterly Central Bank Communications Pipeline
Schedule: Triggered after FOMC/ECB/BOJ policy meetings
Purpose: Process central bank statements with tone analysis and embeddings
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, TEST_MODE,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTIONS,
    OPENAI_API_KEY, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_fomc_minutes(**context):
    """Extract FOMC minutes from Federal Reserve website"""
    logger.info("Extracting FOMC minutes")
    
    statements = []
    
    try:
        # Simplified: Would scrape https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
        statements.append({
            'country': 'USA',
            'central_bank': 'Federal Reserve',
            'meeting_date': '2026-01-29',
            'policy_decision': 'hold',
            'rate_change': 0,
            'statement_text': 'The Federal Reserve will maintain policy rate at 4.25-4.50% range.',
            'tone': 'neutral'
        })
        
        logger.info(f"✅ Extracted {len(statements)} FOMC statements")
    except Exception as e:
        logger.warning(f"FOMC extraction failed: {str(e)}")
    
    context['ti'].xcom_push(key='statements', value=statements)
    return {"extracted": len(statements)}

def transform_policy_tone_analysis(**context):
    """Classify policy statements as hawkish/dovish"""
    statements = context['ti'].xcom_pull(task_ids='extract_fomc_minutes', key='statements') or []
    logger.info(f"Analyzing tone for {len(statements)} statements")
    
    # Simplified sentiment scoring
    hawkish_keywords = ['restrictive', 'tighten', 'inflation', 'raise']
    dovish_keywords = ['accommodative', 'support', 'lower', 'ease']
    
    for stmt in statements:
        text = stmt['statement_text'].lower()
        hawkish_count = sum(1 for kw in hawkish_keywords if kw in text)
        dovish_count = sum(1 for kw in dovish_keywords if kw in text)
        
        if hawkish_count > dovish_count:
            stmt['tone'] = 'hawkish'
            stmt['tone_score'] = 0.7
        elif dovish_count > hawkish_count:
            stmt['tone'] = 'dovish'
            stmt['tone_score'] = -0.7
        else:
            stmt['tone'] = 'neutral'
            stmt['tone_score'] = 0.0
    
    context['ti'].xcom_push(key='statements', value=statements)
    logger.info(f"✅ Tone analysis complete")
    return {"analyzed": len(statements)}

def load_qdrant_central_bank_comms(**context):
    """Load policy statements to Qdrant with embeddings"""
    statements = context['ti'].xcom_pull(task_ids='transform_policy_tone_analysis', key='statements') or []
    
    if not statements:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(statements)} statements to Qdrant")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import openai
        
        openai.api_key = OPENAI_API_KEY
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        collection_name = QDRANT_COLLECTIONS['macro_central_bank_comms']
        
        # Create collection
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except:
            pass
        
        # Load with embeddings
        for i, stmt in enumerate(statements):
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=stmt['statement_text']
            )
            embedding = response.data[0].embedding
            
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=i,
                        vector=embedding,
                        payload=stmt
                    )
                ]
            )
        
        logger.info(f"✅ Loaded {len(statements)} statements to Qdrant")
        return {"loaded": len(statements)}
    
    except Exception as e:
        logger.error(f"Qdrant load failed: {str(e)}")
        return {"loaded": 0}

# DAG DEFINITION
with DAG(
    dag_id='10_quarterly_central_bank_comms_pipeline',
    default_args=DEFAULT_ARGS,
    description='Process central bank policy statements',
    schedule_interval=None,  # Triggered by policy meetings
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['quarterly', 'central_bank', 'policy', 'macro'],
) as dag:
    
    task_extract = PythonOperator(task_id='extract_fomc_minutes', python_callable=extract_fomc_minutes)
    task_analyze = PythonOperator(task_id='transform_policy_tone_analysis', python_callable=transform_policy_tone_analysis)
    task_load_qdrant = PythonOperator(task_id='load_qdrant_central_bank_comms', python_callable=load_qdrant_central_bank_comms)
    
    task_extract >> task_analyze >> task_load_qdrant
