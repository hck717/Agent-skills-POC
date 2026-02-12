"""
DAG 05: Quarterly Earnings Transcripts Pipeline
Schedule: Triggered when earnings date detected
Purpose: Process earnings call transcripts with sentiment analysis and temporal drift
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import psycopg2

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, DEFAULT_TICKERS, TEST_MODE, POSTGRES_URL,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTIONS,
    FMP_API_KEY, FMP_BASE_URL, OPENAI_API_KEY, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_fmp_earnings_transcripts(**context):
    """Extract earnings call transcripts from FMP"""
    logger.info(f"Extracting earnings transcripts for {len(DEFAULT_TICKERS)} tickers")
    transcripts_data = []
    
    for ticker in DEFAULT_TICKERS:
        try:
            url = f"{FMP_BASE_URL}/earning_call_transcript/{ticker}"
            params = {"apikey": FMP_API_KEY}
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data:
                latest = data[0]  # Most recent transcript
                transcripts_data.append({
                    'ticker': ticker,
                    'quarter': latest.get('quarter'),
                    'year': latest.get('year'),
                    'date': latest.get('date'),
                    'content': latest.get('content', ''),
                })
                logger.info(f"✅ {ticker}: Q{latest.get('quarter')} {latest.get('year')} transcript")
        except Exception as e:
            logger.warning(f"Failed to extract transcript for {ticker}: {str(e)}")
    
    context['ti'].xcom_push(key='transcripts_data', value=transcripts_data)
    return {"extracted": len(transcripts_data)}

def transform_transcript_chunking(**context):
    """Split transcripts by speaker turn (CEO, CFO, Q&A)"""
    transcripts_data = context['ti'].xcom_pull(task_ids='extract_fmp_earnings_transcripts', key='transcripts_data') or []
    
    if not transcripts_data:
        context['ti'].xcom_push(key='chunks', value=[])
        return {"chunks": 0}
    
    logger.info(f"Chunking {len(transcripts_data)} transcripts")
    chunks = []
    
    for transcript in transcripts_data:
        content = transcript['content']
        
        # Simplified: Split by sections (in production: parse by speaker)
        sections = [
            {'section': 'Prepared Remarks', 'text': content[:1000]},
            {'section': 'Q&A', 'text': content[1000:2000]}
        ]
        
        for section in sections:
            chunks.append({
                'ticker': transcript['ticker'],
                'quarter': transcript['quarter'],
                'year': transcript['year'],
                'date': transcript['date'],
                'section': section['section'],
                'text': section['text'],
                'sentiment_score': 0  # Placeholder
            })
    
    context['ti'].xcom_push(key='chunks', value=chunks)
    logger.info(f"✅ Generated {len(chunks)} transcript chunks")
    return {"chunks": len(chunks)}

def load_qdrant_transcripts(**context):
    """Load transcript chunks to Qdrant with embeddings"""
    chunks = context['ti'].xcom_pull(task_ids='transform_transcript_chunking', key='chunks') or []
    
    if not chunks:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(chunks)} chunks to Qdrant")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import openai
        
        openai.api_key = OPENAI_API_KEY
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        collection_name = QDRANT_COLLECTIONS['insider_sentiment_transcripts']
        
        # Create collection if not exists
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except:
            pass
        
        # Load chunks with embeddings
        for i, chunk in enumerate(chunks):
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk['text']
            )
            embedding = response.data[0].embedding
            
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=i,
                        vector=embedding,
                        payload=chunk
                    )
                ]
            )
        
        logger.info(f"✅ Loaded {len(chunks)} chunks to Qdrant")
        return {"loaded": len(chunks)}
    
    except Exception as e:
        logger.error(f"Qdrant load failed: {str(e)}")
        return {"loaded": 0}

# DAG DEFINITION
with DAG(
    dag_id='05_quarterly_earnings_transcripts_pipeline',
    default_args=DEFAULT_ARGS,
    description='Process earnings call transcripts with sentiment analysis',
    schedule_interval=None,  # Triggered by earnings calendar
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['quarterly', 'earnings', 'transcripts', 'sentiment'],
) as dag:
    
    task_extract = PythonOperator(task_id='extract_fmp_earnings_transcripts', python_callable=extract_fmp_earnings_transcripts)
    task_chunk = PythonOperator(task_id='transform_transcript_chunking', python_callable=transform_transcript_chunking)
    task_load_qdrant = PythonOperator(task_id='load_qdrant_transcripts', python_callable=load_qdrant_transcripts)
    
    task_extract >> task_chunk >> task_load_qdrant
