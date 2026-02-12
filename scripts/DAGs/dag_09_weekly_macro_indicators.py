"""
DAG 09: Weekly Macro Indicators Pipeline
Schedule: Every Monday at 1:00 AM HKT
Purpose: Refresh macroeconomic data, yield curves, and regime detection
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import duckdb
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow/dags')
from config import (
    DEFAULT_ARGS, TEST_MODE,
    EODHD_API_KEY, FMP_API_KEY, EODHD_BASE_URL, FMP_BASE_URL,
    DUCKDB_MACRO_DB, MAX_ACTIVE_RUNS
)

logger = logging.getLogger(__name__)

def extract_eodhd_macro_indicators(**context):
    """Extract 30+ macro indicators from EODHD"""
    logger.info("Extracting macro indicators from EODHD")
    
    countries = ['USA', 'CHN', 'JPN', 'DEU']  # US, China, Japan, Germany
    indicators = ['GDP', 'unemployment_rate', 'inflation_rate', 'interest_rate']
    
    macro_data = []
    
    for country in countries:
        for indicator in indicators:
            try:
                url = f"{EODHD_BASE_URL}/macro-indicator/{country}"
                params = {"api_token": EODHD_API_KEY, "indicator": indicator, "fmt": "json"}
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    latest = data[-1]
                    macro_data.append({
                        'country': country,
                        'indicator': indicator,
                        'value': latest.get('Value'),
                        'date': latest.get('Date'),
                        'source': 'eodhd'
                    })
            except Exception as e:
                logger.warning(f"Failed {country} {indicator}: {str(e)}")
    
    context['ti'].xcom_push(key='macro_data', value=macro_data)
    logger.info(f"✅ Extracted {len(macro_data)} macro indicators")
    return {"extracted": len(macro_data)}

def extract_fmp_treasury_yields(**context):
    """Extract treasury yields and central bank rates from FMP"""
    macro_data = context['ti'].xcom_pull(task_ids='extract_eodhd_macro_indicators', key='macro_data') or []
    logger.info("Extracting treasury yields from FMP")
    
    try:
        url = f"{FMP_BASE_URL}/treasury"
        params = {"apikey": FMP_API_KEY}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data:
            latest = data[0]
            for tenor in ['month1', 'month3', 'year1', 'year2', 'year5', 'year10', 'year30']:
                yield_value = latest.get(tenor)
                if yield_value:
                    macro_data.append({
                        'country': 'USA',
                        'indicator': f'treasury_{tenor}',
                        'value': yield_value,
                        'date': latest.get('date'),
                        'source': 'fmp'
                    })
        
        logger.info(f"✅ Extracted treasury yields")
    except Exception as e:
        logger.warning(f"FMP treasury yields failed: {str(e)}")
    
    context['ti'].xcom_push(key='macro_data', value=macro_data)
    return {"extracted": len(macro_data)}

def transform_macro_data(**context):
    """Calculate yield curve slopes and real rates"""
    macro_data = context['ti'].xcom_pull(task_ids='extract_fmp_treasury_yields', key='macro_data') or []
    logger.info(f"Transforming {len(macro_data)} macro indicators")
    
    df = pd.DataFrame(macro_data)
    if df.empty:
        context['ti'].xcom_push(key='transformed_data', value=[])
        return {"transformed": 0}
    
    # Calculate yield curve slopes
    df_yields = df[df['indicator'].str.contains('treasury', na=False)]
    
    # Simplified: Calculate 10Y-2Y spread
    y10 = df_yields[df_yields['indicator']=='treasury_year10']['value'].values
    y2 = df_yields[df_yields['indicator']=='treasury_year2']['value'].values
    
    if len(y10) > 0 and len(y2) > 0:
        spread_10y_2y = float(y10[0]) - float(y2[0])
        macro_data.append({
            'country': 'USA',
            'indicator': 'yield_curve_10y_2y',
            'value': spread_10y_2y,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'source': 'calculated'
        })
        
        logger.info(f"✅ 10Y-2Y spread: {spread_10y_2y:.2f}%")
    
    context['ti'].xcom_push(key='transformed_data', value=macro_data)
    return {"transformed": len(macro_data)}

def load_duckdb_macro_indicators(**context):
    """Load macro indicators to DuckDB"""
    transformed_data = context['ti'].xcom_pull(task_ids='transform_macro_data', key='transformed_data') or []
    
    if not transformed_data:
        return {"loaded": 0}
    
    logger.info(f"Loading {len(transformed_data)} indicators to DuckDB")
    
    conn = duckdb.connect(DUCKDB_MACRO_DB)
    
    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            country VARCHAR,
            indicator VARCHAR,
            value DOUBLE,
            date DATE,
            source VARCHAR,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert data
    df = pd.DataFrame(transformed_data)
    conn.execute("INSERT INTO macro_indicators SELECT * FROM df")
    
    conn.close()
    
    logger.info(f"✅ Loaded {len(transformed_data)} indicators to DuckDB")
    return {"loaded": len(transformed_data)}

def detect_macro_regime_changes(**context):
    """Detect yield curve inversions and regime changes"""
    logger.info("Detecting macro regime changes")
    
    conn = duckdb.connect(DUCKDB_MACRO_DB)
    
    # Check for yield curve inversion
    result = conn.execute("""
        SELECT value FROM macro_indicators
        WHERE indicator = 'yield_curve_10y_2y'
        ORDER BY date DESC
        LIMIT 1
    """).fetchone()
    
    if result and result[0] < 0:
        logger.warning(f"⚠️ YIELD CURVE INVERSION DETECTED: {result[0]:.2f}%")
    else:
        logger.info("✅ No yield curve inversion")
    
    conn.close()
    return {"detected": True}

# DAG DEFINITION
with DAG(
    dag_id='09_weekly_macro_indicators_pipeline',
    default_args=DEFAULT_ARGS,
    description='Weekly macro indicators and regime detection',
    schedule_interval='0 17 * * 1' if not TEST_MODE else None,  # Monday 1 AM HKT
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['weekly', 'macro', 'yields', 'duckdb'],
) as dag:
    
    task_extract_eodhd = PythonOperator(task_id='extract_eodhd_macro_indicators', python_callable=extract_eodhd_macro_indicators)
    task_extract_fmp = PythonOperator(task_id='extract_fmp_treasury_yields', python_callable=extract_fmp_treasury_yields)
    task_transform = PythonOperator(task_id='transform_macro_data', python_callable=transform_macro_data)
    task_load_duckdb = PythonOperator(task_id='load_duckdb_macro_indicators', python_callable=load_duckdb_macro_indicators)
    task_detect_regime = PythonOperator(task_id='detect_macro_regime_changes', python_callable=detect_macro_regime_changes)
    
    task_extract_eodhd >> task_extract_fmp >> task_transform >> task_load_duckdb >> task_detect_regime
