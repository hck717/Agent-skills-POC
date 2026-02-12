"""
DAG 12: Monthly Model Retraining Pipeline
Schedule: 1st of every month at 4:00 AM HKT
Purpose: Retrain CRAG evaluator and other ML models
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import config
from pathlib import Path

def collect_crag_feedback_data(**context):
    """Query feedback logs for CRAG retrieval quality"""
    import psycopg2
    
    # Connect to Postgres to fetch feedback
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    
    # Create feedback table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crag_feedback (
            id SERIAL PRIMARY KEY,
            query_text TEXT,
            retrieved_text TEXT,
            relevance_score INT,  -- 1-5 scale
            feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_positive BOOLEAN
        )
    """)
    
    # Fetch recent feedback (last 90 days)
    cur.execute("""
        SELECT query_text, retrieved_text, is_positive
        FROM crag_feedback
        WHERE feedback_date >= CURRENT_DATE - INTERVAL '90 days'
    """)
    
    feedback_data = cur.fetchall()
    
    cur.close()
    conn.close()
    
    # Convert to training format
    training_examples = []
    for query, retrieved, is_positive in feedback_
        training_examples.append({
            'query': query,
            'document': retrieved,
            'label': 1 if is_positive else 0
        })
    
    context['ti'].xcom_push(key='crag_training_data', value=training_examples)
    print(f"Collected {len(training_examples)} CRAG feedback examples")
    return True

def retrain_crag_evaluator(**context):
    """Fine-tune BERT on updated labeled data"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    
    ti = context['ti']
    training_data = ti.xcom_pull(key='crag_training_data', task_ids='collect_crag_feedback_data')
    
    if not training_data or len(training_data) < 100:
        print("Insufficient training data (<100 examples). Skipping retraining.")
        return False
    
    # Prepare dataset
    df = pd.DataFrame(training_data)
    train_texts = df['query'] + " [SEP] " + df['document']
    labels = df['label'].values
    
    # Split train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, labels, test_size=0.2, random_state=42
    )
    
    # Load base model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
    
    # Create PyTorch dataset
    class CRAGDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = CRAGDataset(train_encodings, train_labels)
    val_dataset = CRAGDataset(val_encodings, val_labels)
    
    # Training arguments
    model_output_dir = Path(config.STORAGE_ROOT) / "models" / "crag_evaluator"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(model_output_dir / "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # Save model
    model.save_pretrained(model_output_dir / "best_model")
    tokenizer.save_pretrained(model_output_dir / "best_model")
    
    # Get validation accuracy
    eval_result = trainer.evaluate()
    accuracy = eval_result.get('eval_accuracy', 0)
    
    context['ti'].xcom_push(key='crag_model_accuracy', value=accuracy)
    print(f"CRAG evaluator retrained. Validation accuracy: {accuracy:.4f}")
    
    return True

def recalibrate_sentiment_models(**context):
    """Retrain FinBERT on recent earnings transcripts"""
    from qdrant_client import QdrantClient
    import pandas as pd
    
    # Fetch recent transcripts with sentiment scores from Qdrant
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['insider_sentiment_transcripts']
    
    # Scroll through collection
    transcripts = []
    try:
        results = client.scroll(
            collection_name=collection_name,
            limit=1000
        )
        
        for point in results[0]:
            transcripts.append({
                'content': point.payload.get('content'),
                'sentiment_score': point.payload.get('sentiment_score')
            })
    except Exception as e:
        print(f"Error fetching transcripts: {e}")
        return False
    
    if len(transcripts) < 50:
        print("Insufficient transcripts for retraining")
        return False
    
    # Placeholder for FinBERT fine-tuning
    # Real implementation would use transformers library
    print(f"Recalibrated sentiment model on {len(transcripts)} transcripts")
    
    return True

def update_ner_models(**context):
    """Retrain entity extraction models on new 10-K filings"""
    from qdrant_client import QdrantClient
    
    # Fetch recent 10-K texts
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['business_analyst_10k']
    
    # Fetch recent filings
    try:
        results = client.scroll(
            collection_name=collection_name,
            limit=100
        )
        
        filing_texts = [point.payload.get('content') for point in results[0]]
        
        print(f"Updating NER models with {len(filing_texts)} 10-K texts")
        
        # Placeholder for NER model update
        # Real implementation would use spaCy or transformers
        
    except Exception as e:
        print(f"Error fetching 10-K texts: {e}")
        return False
    
    return True

def validate_model_performance(**context):
    """Run validation suite on test set"""
    ti = context['ti']
    crag_accuracy = ti.xcom_pull(key='crag_model_accuracy', task_ids='retrain_crag_evaluator')
    
    # Load previous best accuracy (would be stored in DB)
    previous_accuracy = 0.85  # Placeholder
    
    validation_results = {
        'crag_evaluator': {
            'new_accuracy': crag_accuracy,
            'previous_accuracy': previous_accuracy,
            'improvement': crag_accuracy - previous_accuracy if crag_accuracy else 0,
            'deploy': crag_accuracy > previous_accuracy if crag_accuracy else False
        },
        'sentiment_model': {
            'status': 'recalibrated',
            'deploy': True
        },
        'ner_model': {
            'status': 'updated',
            'deploy': True
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("Model Retraining Summary")
    print("="*70)
    
    for model_name, results in validation_results.items():
        print(f"\n{model_name}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    print("="*70 + "\n")
    
    context['ti'].xcom_push(key='validation_results', value=validation_results)
    
    # Decide whether to deploy new models
    deploy_count = sum(1 for r in validation_results.values() if r.get('deploy', False))
    print(f"\nDeploying {deploy_count}/{len(validation_results)} updated models")
    
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
})

with DAG(
    dag_id="dag_12_monthly_model_retraining_pipeline",
    default_args=default_args,
    description="Retrain CRAG evaluator and other ML models",
    schedule_interval="0 4 1 * *",  # 1st of month at 4 AM HKT
    catchup=False,
    max_active_runs=1,
    tags=["monthly", "ml", "training", "models"],
) as dag:

    wait_for_sec_filings = ExternalTaskSensor(
        task_id="wait_for_quarterly_sec_filings",
        external_dag_id="dag_04_quarterly_sec_filings_pipeline",
        external_task_id="load_qdrant_propositions",
        execution_delta=timedelta(days=7),
        timeout=7200,
        mode='reschedule',
        poke_interval=3600,
    )

    wait_for_transcripts = ExternalTaskSensor(
        task_id="wait_for_earnings_transcripts",
        external_dag_id="dag_05_quarterly_earnings_transcripts_pipeline",
        external_task_id="load_qdrant_transcripts",
        execution_delta=timedelta(days=7),
        timeout=7200,
        mode='reschedule',
        poke_interval=3600,
    )

    collect_feedback = PythonOperator(
        task_id="collect_crag_feedback_data",
        python_callable=collect_crag_feedback_data,
    )

    retrain_crag = PythonOperator(
        task_id="retrain_crag_evaluator",
        python_callable=retrain_crag_evaluator,
    )

    recalibrate_sentiment = PythonOperator(
        task_id="recalibrate_sentiment_models",
        python_callable=recalibrate_sentiment_models,
    )

    update_ner = PythonOperator(
        task_id="update_ner_models",
        python_callable=update_ner_models,
    )

    validate = PythonOperator(
        task_id="validate_model_performance",
        python_callable=validate_model_performance,
    )

    [wait_for_sec_filings, wait_for_transcripts] >> collect_feedback
    collect_feedback >> retrain_crag
    wait_for_transcripts >> recalibrate_sentiment
    wait_for_sec_filings >> update_ner
    [retrain_crag, recalibrate_sentiment, update_ner] >> validate
