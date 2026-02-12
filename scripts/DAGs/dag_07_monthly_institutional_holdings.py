"""
DAG 7: Monthly Institutional Holdings Pipeline
Schedule: 15th of every month (after 13F filing deadline)
Purpose: Update Supply Chain Graph with institutional ownership
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import config

def extract_fmp_13f_filings(**context):
    """Fetch latest 13F filings from large institutions"""
    import requests
    
    all_filings = []
    
    # Get institutional holders for each ticker
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.FMP_BASE_URL}/institutional-holder/{ticker}"
        params = {"apikey": config.FMP_API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            holders = response.json()
            
            for holder in holders:
                holder['ticker'] = ticker
                all_filings.append(holder)
                
        except Exception as e:
            print(f"Error fetching 13F for {ticker}: {e}")
    
    context['ti'].xcom_push(key='fmp_13f_filings', value=all_filings)
    print(f"Extracted {len(all_filings)} institutional holdings from FMP")
    return True

def extract_eodhd_institutional_holders(**context):
    """Fetch institutional holder data for all tickers"""
    import requests
    
    all_holders = []
    
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.EODHD_BASE_URL}/fundamentals/{ticker}.US"
        params = {
            "api_token": config.EODHD_API_KEY,
            "filter": "Holders"
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if 'Holders' in data and 'Institutions' in data['Holders']:
                for holder in data['Holders']['Institutions']:
                    holder['ticker'] = ticker
                    all_holders.append(holder)
                    
        except Exception as e:
            print(f"Error fetching EODHD holders for {ticker}: {e}")
    
    context['ti'].xcom_push(key='eodhd_holders', value=all_holders)
    print(f"Extracted {len(all_holders)} institutional holders from EODHD")
    return True

def extract_etf_constituents(**context):
    """Fetch ETF holdings (which ETFs hold each stock)"""
    import requests
    
    all_etf_holdings = []
    
    # List of major ETFs to check
    major_etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI']
    
    for etf in major_etfs:
        url = f"{config.FMP_BASE_URL}/etf-holder/{etf}"
        params = {"apikey": config.FMP_API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            holdings = response.json()
            
            for holding in holdings:
                holding['etf'] = etf
                all_etf_holdings.append(holding)
                
        except Exception as e:
            print(f"Error fetching ETF constituents for {etf}: {e}")
    
    context['ti'].xcom_push(key='etf_constituents', value=all_etf_holdings)
    print(f"Extracted {len(all_etf_holdings)} ETF holdings")
    return True

def transform_ownership_data(**context):
    """Calculate ownership concentration and detect changes"""
    import pandas as pd
    
    ti = context['ti']
    fmp_filings = ti.xcom_pull(key='fmp_13f_filings', task_ids='extract_fmp_13f_filings') or []
    eodhd_holders = ti.xcom_pull(key='eodhd_holders', task_ids='extract_eodhd_institutional_holders') or []
    
    # Combine data
    all_holdings = fmp_filings + eodhd_holders
    df = pd.DataFrame(all_holdings)
    
    if df.empty:
        print("No institutional holdings to process")
        context['ti'].xcom_push(key='transformed_holdings', value=[])
        return True
    
    # Calculate ownership concentration (top 10 institutions %)
    concentration_stats = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Sort by shares or value
        if 'shares' in ticker_df.columns:
            ticker_df = ticker_df.sort_values('shares', ascending=False)
        elif 'value' in ticker_df.columns:
            ticker_df = ticker_df.sort_values('value', ascending=False)
        
        # Top 10 concentration
        top_10 = ticker_df.head(10)
        total_shares = ticker_df['shares'].sum() if 'shares' in ticker_df.columns else 0
        top_10_shares = top_10['shares'].sum() if 'shares' in top_10.columns else 0
        
        concentration_pct = (top_10_shares / total_shares * 100) if total_shares > 0 else 0
        
        concentration_stats.append({
            'ticker': ticker,
            'top_10_concentration_pct': concentration_pct,
            'total_institutions': len(ticker_df),
            'total_shares_held': total_shares
        })
    
    # Detect significant changes (>20% change in holdings)
    # This would require historical data - placeholder for now
    changes = []
    
    context['ti'].xcom_push(key='transformed_holdings', value=df.to_dict('records'))
    context['ti'].xcom_push(key='concentration_stats', value=concentration_stats)
    context['ti'].xcom_push(key='significant_changes', value=changes)
    
    print(f"Transformed {len(df)} holdings, calculated concentration for {len(concentration_stats)} tickers")
    return True

def load_neo4j_ownership_graph(**context):
    """Create/update Neo4j ownership graph"""
    from neo4j import GraphDatabase
    
    ti = context['ti']
    holdings = ti.xcom_pull(key='transformed_holdings', task_ids='transform_ownership_data')
    
    if not holdings:
        print("No holdings to load into Neo4j")
        return True
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Create constraints
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) 
            REQUIRE i.name IS UNIQUE
        """)
        
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) 
            REQUIRE c.ticker IS UNIQUE
        """)
        
        # Load holdings
        for holding in holdings:
            institution_name = holding.get('holder') or holding.get('institution')
            ticker = holding.get('ticker')
            shares = holding.get('shares', 0)
            value = holding.get('value', 0)
            report_date = holding.get('dateReported') or holding.get('date')
            change_pct = holding.get('change', 0)
            
            if institution_name and ticker:
                query = """
                    MERGE (i:Institution {name: $institution_name})
                    MERGE (c:Company {ticker: $ticker})
                    MERGE (i)-[r:HOLDS]->(c)
                    SET r.shares = $shares,
                        r.value = $value,
                        r.report_date = $report_date,
                        r.change_pct = $change_pct,
                        r.last_updated = datetime()
                """
                
                session.run(
                    query,
                    institution_name=institution_name,
                    ticker=ticker,
                    shares=shares,
                    value=value,
                    report_date=report_date,
                    change_pct=change_pct
                )
    
    driver.close()
    print(f"Loaded {len(holdings)} ownership relationships into Neo4j")
    return True

def calculate_ownership_metrics(**context):
    """Calculate institutional ownership metrics in Neo4j"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Calculate total institutional ownership % per company
        session.run("""
            MATCH (c:Company)<-[r:HOLDS]-(i:Institution)
            WITH c, SUM(r.shares) as total_inst_shares
            SET c.institutional_ownership_shares = total_inst_shares
        """)
        
        # Calculate ownership concentration (HHI index)
        # HHI = sum of squared market shares
        session.run("""
            MATCH (c:Company)<-[r:HOLDS]-(i:Institution)
            WITH c, COLLECT(r.shares) as shares_list
            WITH c, shares_list, REDUCE(s = 0.0, x IN shares_list | s + x) as total
            WITH c, REDUCE(hhi = 0.0, x IN shares_list | hhi + (x / total) * (x / total)) as hhi_index
            SET c.ownership_hhi = hhi_index
        """)
        
        # Detect shared ownership patterns
        # Find institutions that hold similar portfolios
        session.run("""
            MATCH (i1:Institution)-[:HOLDS]->(c:Company)<-[:HOLDS]-(i2:Institution)
            WHERE id(i1) < id(i2)
            WITH i1, i2, COUNT(c) as shared_holdings
            WHERE shared_holdings >= 5
            MERGE (i1)-[r:SIMILAR_PORTFOLIO]->(i2)
            SET r.shared_holdings = shared_holdings
        """)
    
    driver.close()
    print("Ownership metrics calculated in Neo4j")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_07_monthly_institutional_holdings_pipeline",
    default_args=default_args,
    description="Update Supply Chain Graph with institutional ownership",
    schedule_interval="0 3 15 * *",  # 15th of month at 3 AM HKT
    catchup=False,
    max_active_runs=1,
    tags=["monthly", "institutional", "13f", "ownership"],
) as dag:

    extract_13f = PythonOperator(
        task_id="extract_fmp_13f_filings",
        python_callable=extract_fmp_13f_filings,
    )

    extract_holders = PythonOperator(
        task_id="extract_eodhd_institutional_holders",
        python_callable=extract_eodhd_institutional_holders,
    )

    extract_etf = PythonOperator(
        task_id="extract_etf_constituents",
        python_callable=extract_etf_constituents,
    )

    transform = PythonOperator(
        task_id="transform_ownership_data",
        python_callable=transform_ownership_data,
    )

    load_neo4j = PythonOperator(
        task_id="load_neo4j_ownership_graph",
        python_callable=load_neo4j_ownership_graph,
    )

    calc_metrics = PythonOperator(
        task_id="calculate_ownership_metrics",
        python_callable=calculate_ownership_metrics,
    )

    [extract_13f, extract_holders, extract_etf] >> transform >> load_neo4j >> calc_metrics
