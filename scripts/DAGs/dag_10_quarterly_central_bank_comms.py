"""
DAG 10: Quarterly Central Bank Communications Pipeline
Schedule: Triggered after FOMC/ECB/BOJ policy meetings
Purpose: Process central bank statements for Macro Agent
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import config

def extract_fomc_minutes(**context):
    """Scrape Federal Reserve website for FOMC minutes"""
    import requests
    from bs4 import BeautifulSoup
    
    # Fed FOMC calendar URL
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalend ar.htm"
    
    statements = []
    
    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract latest statement (simplified - real implementation would parse properly)
        statements.append({
            'central_bank': 'Federal Reserve',
            'type': 'FOMC Minutes',
            'date': datetime.now().date(),
            'content': 'Placeholder for FOMC statement text',
            'rate_decision': None,
            'rate_change': None
        })
        
    except Exception as e:
        print(f"Error scraping FOMC minutes: {e}")
    
    context['ti'].xcom_push(key='fomc_statements', value=statements)
    print(f"Extracted {len(statements)} FOMC statements")
    return True

def extract_ecb_statements(**context):
    """Scrape ECB policy announcements"""
    statements = []
    
    # Placeholder - would scrape ECB website
    statements.append({
        'central_bank': 'European Central Bank',
        'type': 'Policy Statement',
        'date': datetime.now().date(),
        'content': 'Placeholder for ECB statement',
        'rate_decision': None
    })
    
    context['ti'].xcom_push(key='ecb_statements', value=statements)
    print(f"Extracted {len(statements)} ECB statements")
    return True

def extract_boj_pboc_statements(**context):
    """Extract BOJ and PBoC policy communications"""
    statements = []
    
    # Placeholder for BOJ
    statements.append({
        'central_bank': 'Bank of Japan',
        'type': 'Policy Statement',
        'date': datetime.now().date(),
        'content': 'Placeholder for BOJ statement'
    })
    
    # Placeholder for PBoC
    statements.append({
        'central_bank': 'Peoples Bank of China',
        'type': 'Policy Statement',
        'date': datetime.now().date(),
        'content': 'Placeholder for PBoC statement'
    })
    
    context['ti'].xcom_push(key='boj_pboc_statements', value=statements)
    print(f"Extracted {len(statements)} BOJ/PBoC statements")
    return True

def transform_policy_tone_analysis(**context):
    """Sentiment analysis: Hawkish vs Dovish classification"""
    ti = context['ti']
    fomc = ti.xcom_pull(key='fomc_statements', task_ids='extract_fomc_minutes') or []
    ecb = ti.xcom_pull(key='ecb_statements', task_ids='extract_ecb_statements') or []
    boj_pboc = ti.xcom_pull(key='boj_pboc_statements', task_ids='extract_boj_pboc_statements') or []
    
    all_statements = fomc + ecb + boj_pboc
    
    # Simple keyword-based tone analysis
    hawkish_words = ['inflation', 'tighten', 'raise', 'restrictive', 'vigilant', 'monitor']
    dovish_words = ['accommodative', 'support', 'lower', 'ease', 'patient', 'data-dependent']
    
    for statement in all_statements:
        content = statement.get('content', '').lower()
        
        hawk_count = sum(1 for word in hawkish_words if word in content)
        dove_count = sum(1 for word in dovish_words if word in content)
        
        # Calculate tone score (-1 = dovish, 0 = neutral, 1 = hawkish)
        total = hawk_count + dove_count
        if total > 0:
            tone_score = (hawk_count - dove_count) / total
        else:
            tone_score = 0
        
        statement['tone_score'] = tone_score
        statement['classification'] = 'Hawkish' if tone_score > 0.2 else ('Dovish' if tone_score < -0.2 else 'Neutral')
        
        # Extract key phrases
        if 'data-dependent' in content:
            statement['key_phrases'] = ['data-dependent']
        elif 'restrictive' in content:
            statement['key_phrases'] = ['restrictive policy']
        else:
            statement['key_phrases'] = []
        
        # Detect policy pivots (would compare to previous statements)
        statement['policy_pivot'] = False
    
    context['ti'].xcom_push(key='analyzed_statements', value=all_statements)
    print(f"Analyzed tone for {len(all_statements)} statements")
    return True

def generate_policy_embeddings(**context):
    """Embed policy statements using local Ollama (llama3.2:latest)"""
    import requests
    
    ti = context['ti']
    statements = ti.xcom_pull(key='analyzed_statements', task_ids='transform_policy_tone_analysis') or []
    
    # Ollama embeddings API
    ollama_url = "http://host.docker.internal:11434/api/embeddings"
    
    for statement in statements:
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": "llama3.2:latest",
                    "prompt": statement['content'][:2000]
                },
                timeout=30
            )
            response.raise_for_status()
            statement['embedding'] = response.json().get('embedding', [])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            statement['embedding'] = [0.0] * 2048
    
    context['ti'].xcom_push(key='embedded_statements', value=statements)
    print(f"Generated {len(statements)} embeddings using Ollama llama3.2")
    return True

    
    ti = context['ti']
    statements = ti.xcom_pull(key='analyzed_statements', task_ids='transform_policy_tone_analysis') or []
    
    for statement in statements:
        try:
            response = openai.Embedding.create(
                model=config.EMBEDDING_MODEL,
                input=statement['content'][:8000]
            )
            statement['embedding'] = response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            statement['embedding'] = None
    
    context['ti'].xcom_push(key='embedded_statements', value=statements)
    print(f"Generated embeddings for {len(statements)} statements")
    return True

def load_qdrant_central_bank_comms(**context):
    """Upsert into Qdrant collection"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid
    
    ti = context['ti']
    statements = ti.xcom_pull(key='embedded_statements', task_ids='generate_policy_embeddings') or []
    
    # Connect to Qdrant
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['macro_central_bank_comms']
    
    # Create collection if not exists
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.EMBEDDING_DIMENSIONS, distance=Distance.COSINE)
        )
    except Exception:
        pass
    
    # Prepare points
    points = []
    for statement in statements:
        if statement.get('embedding'):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=statement['embedding'],
                payload={
                    'central_bank': statement['central_bank'],
                    'type': statement['type'],
                    'date': str(statement['date']),
                    'content': statement['content'][:1000],
                    'tone_score': statement['tone_score'],
                    'classification': statement['classification'],
                    'rate_decision': statement.get('rate_decision'),
                    'rate_change': statement.get('rate_change'),
                    'key_phrases': statement.get('key_phrases', []),
                    'policy_pivot': statement.get('policy_pivot', False)
                }
            )
            points.append(point)
    
    # Upload to Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"Loaded {len(points)} central bank communications into Qdrant")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_10_quarterly_central_bank_comms_pipeline",
    default_args=default_args,
    description="Process central bank statements for Macro Agent",
    schedule_interval=None,  # Triggered after policy meetings
    catchup=False,
    max_active_runs=1,
    tags=["quarterly", "central_bank", "fomc", "ecb", "macro"],
) as dag:

    wait_for_macro = ExternalTaskSensor(
        task_id="wait_for_macro_indicators",
        external_dag_id="dag_09_weekly_macro_indicators_pipeline",
        external_task_id="detect_macro_regime_changes",
        execution_delta=timedelta(days=1),
        timeout=7200,
        mode='reschedule',
    )

    extract_fomc = PythonOperator(
        task_id="extract_fomc_minutes",
        python_callable=extract_fomc_minutes,
    )

    extract_ecb = PythonOperator(
        task_id="extract_ecb_statements",
        python_callable=extract_ecb_statements,
    )

    extract_boj_pboc = PythonOperator(
        task_id="extract_boj_pboc_statements",
        python_callable=extract_boj_pboc_statements,
    )

    analyze_tone = PythonOperator(
        task_id="transform_policy_tone_analysis",
        python_callable=transform_policy_tone_analysis,
    )

    gen_embeddings = PythonOperator(
        task_id="generate_policy_embeddings",
        python_callable=generate_policy_embeddings,
    )

    load_qdrant = PythonOperator(
        task_id="load_qdrant_central_bank_comms",
        python_callable=load_qdrant_central_bank_comms,
    )

    wait_for_macro >> [extract_fomc, extract_ecb, extract_boj_pboc]
    [extract_fomc, extract_ecb, extract_boj_pboc] >> analyze_tone >> gen_embeddings >> load_qdrant
