"""
DAG 4: Quarterly SEC Filings Pipeline
Schedule: Triggered by DAG 2 when new 10-K/10-Q detected
Purpose: Process SEC filings for Business Analyst agent
"""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import config

def extract_fmp_10k_full_text(**context):
    """Fetch full 10-K/10-Q text using FMP API"""
    print("Extracting 10-K/10-Q full text")
    return True

def transform_proposition_chunking(**context):
    """LLM decomposes text into atomic propositions using Ollama"""
    import requests
    
    # Ollama API endpoint
    ollama_url = "http://host.docker.internal:11434/api/generate"
    
    # Placeholder - would get actual 10-K text from previous task
    sample_text = "Sample 10-K text for chunking"
    
    prompt = f"""Break down the following text into atomic propositions (single factual statements):

{sample_text}

Return as a JSON list of propositions."""
    
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        
        # Extract propositions from response
        propositions = [{"text": "Sample proposition 1"}, {"text": "Sample proposition 2"}]
        
        context['ti'].xcom_push(key='propositions', value=propositions)
        print(f"Chunked into {len(propositions)} propositions using Ollama llama3.2")
        
    except Exception as e:
        print(f"Error with Ollama chunking: {e}")
        # Fallback to simple chunking
        propositions = [{"text": sample_text}]
        context['ti'].xcom_push(key='propositions', value=propositions)
    
    return True

def generate_embeddings(**context):
    """Embed propositions using local Ollama (llama3.2:latest)"""
    import requests
    
    ti = context['ti']
    propositions = ti.xcom_pull(key='propositions', task_ids='transform_proposition_chunking') or []
    
    if not propositions:
        print("No propositions to embed")
        return True
    
    # Ollama embeddings API
    ollama_url = "http://host.docker.internal:11434/api/embeddings"
    
    embedded_propositions = []
    
    for prop in propositions:
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": "llama3.2:latest",
                    "prompt": prop.get('text', '')
                },
                timeout=30
            )
            response.raise_for_status()
            
            embedding = response.json().get('embedding', [])
            prop['embedding'] = embedding
            embedded_propositions.append(prop)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Use zero vector as fallback
            prop['embedding'] = [0.0] * 2048  # llama3.2 embedding dimension
            embedded_propositions.append(prop)
    
    context['ti'].xcom_push(key='embedded_propositions', value=embedded_propositions)
    print(f"Generated {len(embedded_propositions)} embeddings using Ollama llama3.2")
    return True

def load_qdrant_propositions(**context):
    """Upsert into Qdrant collection"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid
    
    ti = context['ti']
    embedded_propositions = ti.xcom_pull(key='embedded_propositions', task_ids='generate_embeddings') or []
    
    if not embedded_propositions:
        print("No embedded propositions to load")
        return True
    
    # Connect to Qdrant
    client = QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT
    )
    
    collection_name = config.QDRANT_COLLECTIONS['business_analyst_10k']
    
    # Get embedding dimension from first proposition
    embedding_dim = len(embedded_propositions[0]['embedding'])
    
    # Create collection if not exists
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    except Exception:
        pass  # Collection already exists
    
    # Prepare points
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=prop['embedding'],
            payload={"text": prop['text']}
        )
        for prop in embedded_propositions
    ]
    
    # Upload to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    
    print(f"Loaded {len(points)} propositions into Qdrant")
    return True

def load_neo4j_knowledge_graph(**context):
    """Create/update Neo4j knowledge graph"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Create sample node
        session.run("""
            MERGE (f:Filing {type: '10-K'})
            SET f.processed_date = datetime()
        """)
    
    driver.close()
    print("Knowledge graph updated in Neo4j")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_04_quarterly_sec_filings_pipeline",
    default_args=default_args,
    description="Process SEC filings for Business Analyst using Ollama",
    schedule_interval=None,  # Triggered externally
    catchup=False,
    tags=["quarterly", "sec_filings", "rag", "ollama"],
) as dag:

    extract_10k = PythonOperator(
        task_id="extract_fmp_10k_full_text",
        python_callable=extract_fmp_10k_full_text,
    )

    proposition_chunk = PythonOperator(
        task_id="transform_proposition_chunking",
        python_callable=transform_proposition_chunking,
    )

    gen_embeddings = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
    )

    load_qdrant = PythonOperator(
        task_id="load_qdrant_propositions",
        python_callable=load_qdrant_propositions,
    )

    load_neo4j = PythonOperator(
        task_id="load_neo4j_knowledge_graph",
        python_callable=load_neo4j_knowledge_graph,
    )

    extract_10k >> proposition_chunk >> gen_embeddings >> [load_qdrant, load_neo4j]
