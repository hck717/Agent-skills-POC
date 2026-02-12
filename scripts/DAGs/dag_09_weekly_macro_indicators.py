"""
DAG 9: Weekly Macro Indicators Pipeline
Schedule: Every Monday at 1:00 AM HKT
Purpose: Refresh macroeconomic data for Macro Agent
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import config

def extract_eodhd_macro_indicators(**context):
    """Fetch 30+ indicators for major economies"""
    import requests
    
    # Major economies to track
    countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA']
    
    # Indicators to fetch
    indicators = [
        'GDP', 'GDPR', 'CPI', 'PPI', 'UNRATE', 'RETAIL_SALES',
        'INDUSTRIAL_PRODUCTION', 'REAL_INTEREST_RATE', 'TRADE_BALANCE', 'DEBT_TO_GDP'
    ]
    
    all_data = []
    
    for country in countries:
        for indicator in indicators:
            url = f"{config.EODHD_BASE_URL}/macro-indicator/{country}"
            params = {
                "api_token": config.EODHD_API_KEY,
                "indicator": indicator,
                "fmt": "json"
            }
            
            try:
                response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
                response.raise_for_status()
                data = response.json()
                
                # Get latest value
                if data and len(data) > 0:
                    latest = data[0]
                    latest['country'] = country
                    latest['indicator'] = indicator
                    all_data.append(latest)
                    
            except Exception as e:
                print(f"Error fetching {indicator} for {country}: {e}")
    
    context['ti'].xcom_push(key='macro_indicators', value=all_data)
    print(f"Extracted {len(all_data)} macro indicators")
    return True

def extract_fmp_treasury_yields(**context):
    """Fetch daily treasury yields"""
    import requests
    
    url = f"{config.FMP_BASE_URL}/treasury"
    params = {"apikey": config.FMP_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
        response.raise_for_status()
        yields_data = response.json()
        
        context['ti'].xcom_push(key='treasury_yields', value=yields_data)
        print(f"Extracted treasury yields: {len(yields_data)} records")
        return True
    except Exception as e:
        print(f"Error fetching treasury yields: {e}")
        return False

def extract_fmp_economic_calendar(**context):
    """Fetch upcoming economic events (next 30 days)"""
    import requests
    from datetime import datetime, timedelta
    
    from_date = datetime.fromisoformat(context['ds'])
    to_date = from_date + timedelta(days=30)
    
    url = f"{config.FMP_BASE_URL}/economic_calendar"
    params = {
        "from": from_date.strftime('%Y-%m-%d'),
        "to": to_date.strftime('%Y-%m-%d'),
        "apikey": config.FMP_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
        response.raise_for_status()
        calendar = response.json()
        
        context['ti'].xcom_push(key='economic_calendar', value=calendar)
        print(f"Extracted {len(calendar)} economic calendar events")
        return True
    except Exception as e:
        print(f"Error fetching economic calendar: {e}")
        return False

def extract_fx_rates(**context):
    """Fetch daily FX rates for major pairs"""
    import requests
    
    # Major currency pairs
    pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCNY', 'AUDUSD', 'USDCAD']
    
    all_fx = []
    
    for pair in pairs:
        url = f"{config.EODHD_BASE_URL}/real-time/{pair}.FOREX"
        params = {
            "api_token": config.EODHD_API_KEY,
            "fmt": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            fx_data = response.json()
            fx_data['pair'] = pair
            all_fx.append(fx_data)
        except Exception as e:
            print(f"Error fetching FX rate for {pair}: {e}")
    
    context['ti'].xcom_push(key='fx_rates', value=all_fx)
    print(f"Extracted {len(all_fx)} FX rates")
    return True

def transform_macro_data(**context):
    """Calculate derived metrics and classify economic cycle"""
    import pandas as pd
    
    ti = context['ti']
    indicators = ti.xcom_pull(key='macro_indicators', task_ids='extract_eodhd_macro_indicators') or []
    yields_data = ti.xcom_pull(key='treasury_yields', task_ids='extract_fmp_treasury_yields') or []
    
    df_indicators = pd.DataFrame(indicators)
    df_yields = pd.DataFrame(yields_data)
    
    derived_metrics = []
    
    # Calculate yield curve slopes
    if not df_yields.empty and 'year10' in df_yields.columns and 'year2' in df_yields.columns:
        slope_10y_2y = df_yields['year10'].iloc[0] - df_yields['year2'].iloc[0]
        
        derived_metrics.append({
            'metric': 'yield_curve_10y_2y',
            'value': slope_10y_2y,
            'signal': 'recession_warning' if slope_10y_2y < 0 else 'normal'
        })
    
    # Calculate real interest rates (nominal - CPI)
    for country in df_indicators['country'].unique():
        country_data = df_indicators[df_indicators['country'] == country]
        
        cpi = country_data[country_data['indicator'] == 'CPI']['Value'].values
        
        if len(cpi) > 0:
            cpi_rate = float(cpi[0]) if len(cpi) > 0 else 0
            
            # Get policy rate (would need to fetch separately)
            nominal_rate = 5.0  # Placeholder
            real_rate = nominal_rate - cpi_rate
            
            derived_metrics.append({
                'country': country,
                'metric': 'real_interest_rate',
                'value': real_rate
            })
    
    # Classify economic cycle phase
    # Simple classification: GDP growth + Unemployment trend
    cycle_phases = []
    
    for country in df_indicators['country'].unique():
        country_data = df_indicators[df_indicators['country'] == country]
        
        gdp = country_data[country_data['indicator'] == 'GDPR']
        unemployment = country_data[country_data['indicator'] == 'UNRATE']
        
        if not gdp.empty:
            gdp_growth = float(gdp['Value'].iloc[0])
            
            if gdp_growth > 2.5:
                phase = 'expansion'
            elif gdp_growth > 0:
                phase = 'slowdown'
            else:
                phase = 'contraction'
            
            cycle_phases.append({
                'country': country,
                'cycle_phase': phase,
                'gdp_growth': gdp_growth
            })
    
    context['ti'].xcom_push(key='derived_metrics', value=derived_metrics)
    context['ti'].xcom_push(key='cycle_phases', value=cycle_phases)
    
    print(f"Calculated {len(derived_metrics)} derived metrics, classified {len(cycle_phases)} cycles")
    return True

def load_duckdb_macro_indicators(**context):
    """Load macro data into DuckDB"""
    import duckdb
    import pandas as pd
    from pathlib import Path
    
    ti = context['ti']
    indicators = ti.xcom_pull(key='macro_indicators', task_ids='extract_eodhd_macro_indicators') or []
    fx_rates = ti.xcom_pull(key='fx_rates', task_ids='extract_fx_rates') or []
    calendar = ti.xcom_pull(key='economic_calendar', task_ids='extract_fmp_economic_calendar') or []
    
    # Ensure DuckDB directory exists
    Path(config.DUCKDB_PATH).mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(config.DUCKDB_MACRO_DB)
    
    # Create macro_indicators table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            country VARCHAR(10),
            indicator VARCHAR(50),
            date DATE,
            value DOUBLE,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Load indicators
    if indicators:
        df_indicators = pd.DataFrame(indicators)
        conn.execute("INSERT INTO macro_indicators SELECT country, indicator, Date as date, Value as value, CURRENT_TIMESTAMP FROM df_indicators")
    
    # Create fx_rates table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            pair VARCHAR(10),
            date DATE,
            rate DOUBLE,
            PRIMARY KEY (pair, date)
        )
    """)
    
    # Load FX rates
    if fx_rates:
        for fx in fx_rates:
            conn.execute("""
                INSERT OR REPLACE INTO fx_rates (pair, date, rate)
                VALUES (?, CURRENT_DATE, ?)
            """, [fx['pair'], fx.get('close', 0)])
    
    # Create economic_calendar table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS economic_calendar (
            event_name VARCHAR(200),
            country VARCHAR(50),
            event_date DATE,
            actual DOUBLE,
            estimate DOUBLE,
            previous DOUBLE,
            PRIMARY KEY (event_name, country, event_date)
        )
    """)
    
    # Load calendar events
    if calendar:
        df_calendar = pd.DataFrame(calendar)
        if not df_calendar.empty:
            conn.execute("""
                INSERT OR REPLACE INTO economic_calendar 
                SELECT event as event_name, country, date as event_date, 
                       actual, estimate, previous 
                FROM df_calendar
            """)
    
    conn.close()
    print(f"Loaded macro data into DuckDB")
    return True

def detect_macro_regime_changes(**context):
    """Flag significant macro regime changes"""
    import duckdb
    
    conn = duckdb.connect(config.DUCKDB_MACRO_DB)
    
    alerts = []
    
    # Check for yield curve inversion
    # Would query from treasury yields table
    # Placeholder logic
    
    # Check for high inflation (> 5%)
    result = conn.execute("""
        SELECT country, value as cpi_rate
        FROM macro_indicators
        WHERE indicator = 'CPI'
          AND date = (SELECT MAX(date) FROM macro_indicators WHERE indicator = 'CPI')
          AND value > 5
    """).fetchall()
    
    for row in result:
        alerts.append({
            'alert_type': 'high_inflation',
            'country': row[0],
            'value': row[1],
            'message': f"High inflation regime: {row[1]:.2f}%"
        })
    
    conn.close()
    
    context['ti'].xcom_push(key='regime_alerts', value=alerts)
    print(f"Detected {len(alerts)} macro regime changes")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_09_weekly_macro_indicators_pipeline",
    default_args=default_args,
    description="Refresh macroeconomic data for Macro Agent",
    schedule_interval="0 1 * * 1",  # Every Monday at 1 AM HKT
    catchup=False,
    max_active_runs=1,
    tags=["weekly", "macro", "economics"],
) as dag:

    extract_macro = PythonOperator(
        task_id="extract_eodhd_macro_indicators",
        python_callable=extract_eodhd_macro_indicators,
    )

    extract_yields = PythonOperator(
        task_id="extract_fmp_treasury_yields",
        python_callable=extract_fmp_treasury_yields,
    )

    extract_calendar = PythonOperator(
        task_id="extract_fmp_economic_calendar",
        python_callable=extract_fmp_economic_calendar,
    )

    extract_fx = PythonOperator(
        task_id="extract_fx_rates",
        python_callable=extract_fx_rates,
    )

    transform = PythonOperator(
        task_id="transform_macro_data",
        python_callable=transform_macro_data,
    )

    load_duckdb = PythonOperator(
        task_id="load_duckdb_macro_indicators",
        python_callable=load_duckdb_macro_indicators,
    )

    detect_regimes = PythonOperator(
        task_id="detect_macro_regime_changes",
        python_callable=detect_macro_regime_changes,
    )

    [extract_macro, extract_yields, extract_calendar, extract_fx] >> transform
    transform >> load_duckdb >> detect_regimes
