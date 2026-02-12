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
from airflow import DAG
from airflow.models import DagBag
from airflow.utils.state import State
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All DAG IDs
DAG_IDS = [
    '00_connection_test',
    '01_daily_market_data',
    '02_daily_news_sentiment',
    '03_weekly_fundamental_data',
    '04_quarterly_sec_filings',
    '05_quarterly_earnings_transcripts',
    '06_monthly_insider_trading',
    '07_monthly_institutional_holdings',
    '08_quarterly_supply_chain_graph',
    '09_weekly_macro_indicators',
    '10_quarterly_central_bank_comms',
    '11_daily_data_quality',
    '12_monthly_model_retraining',
    '13_neo4j_auto_ingest',
]

def test_dag(dag_id):
    """Test a single DAG"""
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
        
        # Test the DAG
        from datetime import datetime
        execution_date = datetime(2026, 2, 12)
        
        dag.test(
            execution_date=execution_date,
            run_conf=None,
            conn_file_path=None,
            variable_file_path=None,
            session=None,
        )
        
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
        dag_id = f"{dag_num:02d}_" + [d for d in DAG_IDS if d.startswith(f"{dag_num:02d}_")][0].split('_', 1)[1]
        
        if dag_id not in DAG_IDS:
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
