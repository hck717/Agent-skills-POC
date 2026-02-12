"""
DAG 5: Quarterly Earnings Transcripts Pipeline
Schedule: Triggered when earnings date detected
Purpose: Process earnings call transcripts for Insider & Sentiment agent
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import config

def extract_fmp_earnings_transcripts(**context):
    """Fetch earnings call transcripts for companies reporting this week"""
    import requests
    from datetime import datetime, timedelta
    
    execution_date = datetime.fromisoformat(context['ds'])
    
    all_transcripts = []
    for ticker in config.DEFAULT_TICKERS:
        # FMP Earnings Transcripts API
        url = f"{config.FMP_BASE_URL}/earning_call_transcript/{ticker}"
        params = {"apikey": config.FMP_API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            transcripts = response.json()
            
            # Get most recent transcript
            if transcripts and len(transcripts) > 0:
                latest = transcripts[0]
                latest['ticker'] = ticker
                all_transcripts.append(latest)
                
        except Exception as e:
            print(f"Error fetching transcript for {ticker}: {e}")
    
    context['ti'].xcom_push(key='transcripts', value=all_transcripts)
    print(f"Extracted {len(all_transcripts)} earnings transcripts")
    return True

def extract_earnings_dates(**context):
    """Fetch FMP earnings calendar for upcoming quarter"""
    import requests
    from datetime import datetime, timedelta
    
    # Get earnings calendar for next 90 days
    from_date = datetime.fromisoformat(context['ds'])
    to_date = from_date + timedelta(days=90)
    
    url = f"{config.FMP_BASE_URL}/earning_calendar"
    params = {
        "from": from_date.strftime('%Y-%m-%d'),
        "to": to_date.strftime('%Y-%m-%d'),
        "apikey": config.FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
        response.raise_for_status()
        earnings_calendar = response.json()
        
        context['ti'].xcom_push(key='earnings_calendar', value=earnings_calendar)
        print(f"Extracted {len(earnings_calendar)} earnings dates")
        return True
    except Exception as e:
        print(f"Error fetching earnings calendar: {e}")
        return False

def transform_transcript_chunking(**context):
    """Split by speaker turn and tag chunks"""
    import re
    
    ti = context['ti']
    transcripts = ti.xcom_pull(key='transcripts', task_ids='extract_fmp_earnings_transcripts')
    
    all_chunks = []
    
    for transcript in transcripts:
        text = transcript.get('content', '')
        ticker = transcript.get('ticker')
        quarter = transcript.get('quarter')
        year = transcript.get('year')
        
        # Split by speaker patterns
        # Pattern: "Speaker Name:\n" or "Q:\n" or "A:\n"
        speaker_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+|Q|A):\\n'
        chunks = re.split(speaker_pattern, text)
        
        current_speaker = None
        for i, chunk in enumerate(chunks):
            if i % 2 == 1:  # Speaker name
                current_speaker = chunk
            elif chunk.strip():  # Content
                # Determine speaker role
                role = 'Unknown'
                if current_speaker:
                    if 'CEO' in chunk or current_speaker in ['Chief Executive']:
                        role = 'CEO'
                    elif 'CFO' in chunk or 'Financial' in chunk:
                        role = 'CFO'
                    elif current_speaker == 'Q':
                        role = 'Analyst'
                
                # Determine section
                section = 'qa' if current_speaker in ['Q', 'A'] else 'prepared_remarks'
                
                chunk_data = {
                    'ticker': ticker,
                    'quarter': quarter,
                    'fiscal_year': year,
                    'speaker_role': role,
                    'transcript_section': section,
                    'content': chunk.strip(),
                    'speaker_name': current_speaker
                }
                all_chunks.append(chunk_data)
    
    context['ti'].xcom_push(key='chunked_transcripts', value=all_chunks)
    print(f"Created {len(all_chunks)} transcript chunks")
    return True

def extract_sentiment_scores(**context):
    """Run sentiment analysis on each chunk"""
    ti = context['ti']
    chunks = ti.xcom_pull(key='chunked_transcripts', task_ids='transform_transcript_chunking')
    
    # Placeholder for sentiment analysis
    # In production: Use FinBERT or FMP sentiment API
    for chunk in chunks:
        content = chunk['content'].lower()
        
        # Simple keyword-based sentiment (replace with FinBERT)
        positive_words = ['growth', 'increase', 'strong', 'positive', 'beat', 'exceed']
        negative_words = ['decline', 'decrease', 'weak', 'negative', 'miss', 'below']
        uncertain_words = ['may', 'might', 'could', 'uncertain', 'challenging']
        forward_words = ['expect', 'forecast', 'guidance', 'outlook', 'anticipate']
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        
        # Calculate sentiment score (-1 to 1)
        total = pos_count + neg_count
        if total > 0:
            chunk['sentiment_score'] = (pos_count - neg_count) / total
        else:
            chunk['sentiment_score'] = 0.0
        
        # Count mentions
        chunk['guidance_mentions'] = sum(1 for word in forward_words if word in content)
        chunk['uncertainty_mentions'] = sum(1 for word in uncertain_words if word in content)
    
    context['ti'].xcom_push(key='sentiment_chunks', value=chunks)
    print(f"Analyzed sentiment for {len(chunks)} chunks")
    return True

def generate_transcript_embeddings(**context):
    """Embed each chunk using local Ollama (llama3.2:latest)"""
    import requests
    
    ti = context['ti']
    chunks = ti.xcom_pull(key='sentiment_chunks', task_ids='extract_sentiment_scores')
    
    # Ollama embeddings API
    ollama_url = "http://host.docker.internal:11434/api/embeddings"
    
    for chunk in chunks:
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": "llama3.2:latest",
                    "prompt": chunk['content'][:2000]  # Limit to avoid timeout
                },
                timeout=30
            )
            response.raise_for_status()
            chunk['embedding'] = response.json().get('embedding', [])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            chunk['embedding'] = [0.0] * 2048  # llama3.2 embedding dimension
    
    context['ti'].xcom_push(key='embedded_chunks', value=chunks)
    print(f"Generated {len(chunks)} embeddings using Ollama llama3.2")
    return True

def load_qdrant_transcripts(**context):
    """Upsert into Qdrant collection"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid
    
    ti = context['ti']
    chunks = ti.xcom_pull(key='embedded_chunks', task_ids='generate_transcript_embeddings')
    
    # Connect to Qdrant
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['insider_sentiment_transcripts']
    
    # Create collection if not exists
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.EMBEDDING_DIMENSIONS, distance=Distance.COSINE)
        )
    except Exception:
        pass  # Collection already exists
    
    # Prepare points
    points = []
    for chunk in chunks:
        if chunk.get('embedding'):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk['embedding'],
                payload={
                    'ticker': chunk['ticker'],
                    'quarter': chunk['quarter'],
                    'fiscal_year': chunk['fiscal_year'],
                    'speaker_role': chunk['speaker_role'],
                    'transcript_section': chunk['transcript_section'],
                    'sentiment_score': chunk['sentiment_score'],
                    'content': chunk['content'][:1000],  # Truncate for storage
                    'guidance_mentions': chunk.get('guidance_mentions', 0),
                    'uncertainty_mentions': chunk.get('uncertainty_mentions', 0)
                }
            )
            points.append(point)
    
    # Upload to Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"Loaded {len(points)} chunks into Qdrant")
    return True

def calculate_temporal_drift(**context):
    """Calculate narrative drift from previous quarter"""
    from qdrant_client import QdrantClient
    import numpy as np
    
    ti = context['ti']
    chunks = ti.xcom_pull(key='embedded_chunks', task_ids='generate_transcript_embeddings')
    
    # Connect to Qdrant
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['insider_sentiment_transcripts']
    
    drift_alerts = []
    
    for chunk in chunks:
        if not chunk.get('embedding'):
            continue
        
        # Query for previous quarter's guidance
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=chunk['embedding'],
                query_filter={
                    "must": [
                        {"key": "ticker", "match": {"value": chunk['ticker']}},
                        {"key": "transcript_section", "match": {"value": "prepared_remarks"}}
                    ]
                },
                limit=5
            )
            
            # Calculate similarity with previous quarter
            if results:
                max_similarity = max([r.score for r in results])
                
                # Flag if similarity < 0.8 (narrative drift)
                if max_similarity < 0.8:
                    drift_alerts.append({
                        'ticker': chunk['ticker'],
                        'quarter': chunk['quarter'],
                        'similarity': max_similarity,
                        'message': 'Narrative Drift Detected'
                    })
        except Exception as e:
            print(f"Error calculating drift: {e}")
    
    context['ti'].xcom_push(key='drift_alerts', value=drift_alerts)
    print(f"Detected {len(drift_alerts)} narrative drift alerts")
    return True

def cross_reference_insider_trades(**context):
    """Query Postgres for insider trades around earnings date"""
    import psycopg2
    from datetime import timedelta
    
    ti = context['ti']
    drift_alerts = ti.xcom_pull(key='drift_alerts', task_ids='calculate_temporal_drift')
    
    # Connect to Postgres
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    divergence_scores = []
    
    for alert in drift_alerts:
        # Query insider transactions 30 days before earnings
        query = """
            SELECT COUNT(*) as sell_count, SUM(shares * price) as sell_value
            FROM insider_transactions
            WHERE ticker = %s
              AND transaction_type = 'Sale'
              AND transaction_date >= (CURRENT_DATE - INTERVAL '30 days')
        """
        
        try:
            cur.execute(query, (alert['ticker'],))
            result = cur.fetchone()
            
            if result and result[0] > 0:
                sell_count = result[0]
                sell_value = result[1] or 0
                
                # Calculate divergence score
                sentiment_drift = 1 - alert['similarity']
                insider_sell_ratio = min(sell_count / 10.0, 1.0)  # Normalize
                
                divergence_score = sentiment_drift * insider_sell_ratio
                
                divergence_scores.append({
                    'ticker': alert['ticker'],
                    'divergence_score': divergence_score,
                    'sentiment_drift': sentiment_drift,
                    'insider_sell_count': sell_count,
                    'insider_sell_value': sell_value
                })
        except Exception as e:
            print(f"Error querying insider trades: {e}")
    
    cur.close()
    conn.close()
    
    context['ti'].xcom_push(key='divergence_scores', value=divergence_scores)
    print(f"Calculated {len(divergence_scores)} divergence scores")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_05_quarterly_earnings_transcripts_pipeline",
    default_args=default_args,
    description="Process earnings call transcripts for Insider & Sentiment agent",
    schedule_interval=None,  # Triggered by earnings dates
    catchup=False,
    max_active_runs=1,
    tags=["quarterly", "earnings", "transcripts", "sentiment"],
) as dag:

    wait_for_fundamentals = ExternalTaskSensor(
        task_id="wait_for_weekly_fundamentals",
        external_dag_id="dag_03_weekly_fundamental_data_pipeline",
        external_task_id="load_postgres_financials",
        execution_delta=timedelta(days=1),
        timeout=7200,
        mode='reschedule',
    )

    extract_transcripts = PythonOperator(
        task_id="extract_fmp_earnings_transcripts",
        python_callable=extract_fmp_earnings_transcripts,
    )

    extract_dates = PythonOperator(
        task_id="extract_earnings_dates",
        python_callable=extract_earnings_dates,
    )

    chunk_transcripts = PythonOperator(
        task_id="transform_transcript_chunking",
        python_callable=transform_transcript_chunking,
    )

    sentiment_analysis = PythonOperator(
        task_id="extract_sentiment_scores",
        python_callable=extract_sentiment_scores,
    )

    gen_embeddings = PythonOperator(
        task_id="generate_transcript_embeddings",
        python_callable=generate_transcript_embeddings,
    )

    load_qdrant = PythonOperator(
        task_id="load_qdrant_transcripts",
        python_callable=load_qdrant_transcripts,
    )

    calc_drift = PythonOperator(
        task_id="calculate_temporal_drift",
        python_callable=calculate_temporal_drift,
    )

    cross_ref = PythonOperator(
        task_id="cross_reference_insider_trades",
        python_callable=cross_reference_insider_trades,
    )

    wait_for_fundamentals >> [extract_transcripts, extract_dates]
    extract_transcripts >> chunk_transcripts >> sentiment_analysis >> gen_embeddings
    gen_embeddings >> load_qdrant >> calc_drift >> cross_ref
