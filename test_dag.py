#!/usr/bin/env python3
"""
Simple DAG tester - Run DAGs like regular Python scripts
Usage: python test_dag.py [dag_number]
  Example: python test_dag.py 0   # Test DAG 00
  Example: python test_dag.py 5   # Test DAG 05
  Example: python test_dag.py     # Test all DAGs
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Airflow environment
os.environ['AIRFLOW_HOME'] = str(Path.cwd())
os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = str(Path.cwd() / 'scripts' / 'DAGs')
os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
os.environ['AIRFLOW__CORE__EXECUTOR'] = 'SequentialExecutor'
os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = os.getenv('POSTGRES_URL')
os.environ['AIRFLOW__CORE__FERNET_KEY'] = '81HqDtbqAywKSOumSha3BhWNOdQ26slT6K0YaZeZyPs='

# Import after setting env vars
from airflow.models import DagBag
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Actual DAG IDs (some have _pipeline suffix, some don't)
DAG_IDS = [
    '00_connection_test',
    '01_daily_market_data',
    'dag_02_daily_news_sentiment_pipeline',
    'dag_03_weekly_fundamental_data_pipeline',
    'dag_04_quarterly_sec_filings_pipeline',
    'dag_05_quarterly_earnings_transcripts_pipeline',
    '06_monthly_insider_trading',
    'dag_07_monthly_institutional_holdings_pipeline',
    'dag_08_quarterly_supply_chain_graph_pipeline',
    'dag_09_weekly_macro_indicators_pipeline',
    'dag_10_quarterly_central_bank_comms_pipeline',
    'dag_11_daily_data_quality_monitoring',
    '12_monthly_model_retraining',
    'dag_13_neo4j_auto_ingest_pipeline',
]

def test_dag(dag_id):
    """Test a single DAG by loading and validating it"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing DAG: {dag_id}")
    print(f"{'='*70}\n")
    
    try:
        # Load DAG
        dagbag = DagBag(dag_folder=os.environ['AIRFLOW__CORE__DAGS_FOLDER'])
        
        if dag_id not in dagbag.dags:
            print(f"❌ DAG '{dag_id}' not found")
            print(f"Available DAGs: {list(dagbag.dags.keys())}")
            return False
        
        dag = dagbag.get_dag(dag_id)
        
        # Validate DAG structure
        if not dag.tasks:
            print(f"❌ DAG has no tasks")
            return False
        
        print(f"✅ DAG loaded successfully")
        print(f"   Tasks: {len(dag.tasks)}")
        print(f"   Task IDs: {[t.task_id for t in dag.tasks]}")
        
        # Test each task can be instantiated
        for task in dag.tasks:
            try:
                # Just check task is valid
                _ = task.task_id
                _ = task.task_type
            except Exception as e:
                print(f"❌ Task '{task.task_id}' validation failed: {e}")
                return False
        
        print(f"\n✅ {dag_id} PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ {dag_id} FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Check if specific DAG number provided
    if len(sys.argv) > 1:
        dag_num = int(sys.argv[1])
        
        # Find matching DAG ID
        dag_id = None
        for did in DAG_IDS:
            if did.startswith(f"{dag_num:02d}_") or did.startswith(f"dag_{dag_num:02d}_"):
                dag_id = did
                break
        
        if not dag_id:
            print(f"❌ Invalid DAG number: {dag_num}")
            print(f"Available: 0-{len(DAG_IDS)-1}")
            sys.exit(1)
        
        success = test_dag(dag_id)
        sys.exit(0 if success else 1)
    
    # Test all DAGs
    print("🧪 Testing all DAGs (00-13)...\n")
    
    passed = 0
    failed = 0
    failed_dags = []
    
    for dag_id in DAG_IDS:
        if test_dag(dag_id):
            passed += 1
        else:
            failed += 1
            failed_dags.append(dag_id)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed DAGs:")
        for dag in failed_dags:
            print(f"  - {dag}")
        sys.exit(1)
    else:
        print("\n🎉 All DAGs passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
