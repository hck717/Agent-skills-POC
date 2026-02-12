"""
DAG 8: Quarterly Supply Chain Graph Pipeline
Schedule: Every quarter after 10-K/10-Q filings processed
Purpose: Build supply chain relationships and calculate graph metrics
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import config

def extract_10k_customer_supplier_mentions(**context):
    """Parse 10-K Item 1 for customer/supplier names"""
    # This would parse previously stored 10-K texts from Qdrant
    from qdrant_client import QdrantClient
    import re
    
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['business_analyst_10k']
    
    relationships = []
    
    # Search for business description sections
    for ticker in config.DEFAULT_TICKERS:
        try:
            # Query for business description sections
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {"key": "ticker", "match": {"value": ticker}},
                        {"key": "section", "match": {"value": "Business Description"}}
                    ]
                },
                limit=10
            )
            
            for point in results[0]:
                content = point.payload.get('content', '')
                
                # Extract revenue concentration patterns
                # Pattern: "Customer A represents X% of revenue"
                pattern = r'([A-Z][\w\s]+)\s+(?:represents|accounts for|comprised)\s+(\d+)%\s+of\s+(?:total\s+)?revenue'
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    customer_name = match.group(1).strip()
                    revenue_pct = float(match.group(2))
                    
                    relationships.append({
                        'ticker': ticker,
                        'relationship_type': 'customer',
                        'entity': customer_name,
                        'revenue_pct': revenue_pct,
                        'source': 'Item 1'
                    })
        except Exception as e:
            print(f"Error extracting relationships for {ticker}: {e}")
    
    context['ti'].xcom_push(key='customer_supplier_mentions', value=relationships)
    print(f"Extracted {len(relationships)} customer/supplier relationships")
    return True

def extract_fmp_stock_peers(**context):
    """Fetch FMP stock peers API (companies in same sector/industry)"""
    import requests
    
    all_peers = []
    
    for ticker in config.DEFAULT_TICKERS:
        url = f"{config.FMP_BASE_URL}/stock_peers"
        params = {
            "symbol": ticker,
            "apikey": config.FMP_API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            peers = response.json()
            
            if peers and len(peers) > 0:
                peer_list = peers[0].get('peersList', [])
                for peer in peer_list:
                    all_peers.append({
                        'ticker': ticker,
                        'peer': peer,
                        'relationship': 'competitor'
                    })
        except Exception as e:
            print(f"Error fetching peers for {ticker}: {e}")
    
    context['ti'].xcom_push(key='stock_peers', value=all_peers)
    print(f"Extracted {len(all_peers)} peer relationships")
    return True

def extract_product_geography_data(**context):
    """Parse 10-K segment reporting for products and geographies"""
    from qdrant_client import QdrantClient
    import re
    
    if config.QDRANT_API_KEY:
        client = QdrantClient(url=f"https://{config.QDRANT_HOST}", api_key=config.QDRANT_API_KEY)
    else:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    collection_name = config.QDRANT_COLLECTIONS['business_analyst_10k']
    
    segments = []
    
    for ticker in config.DEFAULT_TICKERS:
        try:
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {"key": "ticker", "match": {"value": ticker}},
                        {"key": "section", "match": {"value": "Segment Reporting"}}
                    ]
                },
                limit=5
            )
            
            for point in results[0]:
                content = point.payload.get('content', '')
                
                # Extract geographic segments
                geo_pattern = r'(North America|Europe|Asia|China|United States)\s+revenue\s+(?:was|totaled|comprised)\s+(\d+)%'
                geo_matches = re.finditer(geo_pattern, content, re.IGNORECASE)
                
                for match in geo_matches:
                    geography = match.group(1)
                    pct = float(match.group(2))
                    
                    segments.append({
                        'ticker': ticker,
                        'segment_type': 'geography',
                        'segment_name': geography,
                        'revenue_pct': pct
                    })
        except Exception as e:
            print(f"Error extracting segments for {ticker}: {e}")
    
    context['ti'].xcom_push(key='product_geography_segments', value=segments)
    print(f"Extracted {len(segments)} segment exposures")
    return True

def transform_supply_chain_entities(**context):
    """NER to extract company names and resolve aliases"""
    ti = context['ti']
    relationships = ti.xcom_pull(key='customer_supplier_mentions', task_ids='extract_10k_customer_supplier_mentions') or []
    peers = ti.xcom_pull(key='stock_peers', task_ids='extract_fmp_stock_peers') or []
    segments = ti.xcom_pull(key='product_geography_segments', task_ids='extract_product_geography_data') or []
    
    # Simple entity resolution (in production, use advanced NER)
    entity_aliases = {
        'Apple Inc.': 'AAPL',
        'Apple': 'AAPL',
        'Microsoft Corporation': 'MSFT',
        'Microsoft': 'MSFT',
        'Amazon.com': 'AMZN',
        'Amazon': 'AMZN',
    }
    
    # Resolve entities
    for rel in relationships:
        entity = rel.get('entity', '')
        if entity in entity_aliases:
            rel['resolved_entity'] = entity_aliases[entity]
        else:
            rel['resolved_entity'] = entity
    
    context['ti'].xcom_push(key='resolved_relationships', value=relationships)
    context['ti'].xcom_push(key='peer_relationships', value=peers)
    context['ti'].xcom_push(key='segment_data', value=segments)
    
    print(f"Transformed {len(relationships)} relationships, {len(peers)} peers, {len(segments)} segments")
    return True

def load_neo4j_supply_chain_graph(**context):
    """Create supply chain nodes and relationships in Neo4j"""
    from neo4j import GraphDatabase
    
    ti = context['ti']
    relationships = ti.xcom_pull(key='resolved_relationships', task_ids='transform_supply_chain_entities') or []
    peers = ti.xcom_pull(key='peer_relationships', task_ids='transform_supply_chain_entities') or []
    segments = ti.xcom_pull(key='segment_data', task_ids='transform_supply_chain_entities') or []
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Create Company nodes
        for ticker in config.DEFAULT_TICKERS:
            session.run("""
                MERGE (c:Company {ticker: $ticker})
            """, ticker=ticker)
        
        # Create customer/supplier relationships
        for rel in relationships:
            if rel['relationship_type'] == 'customer':
                session.run("""
                    MERGE (supplier:Company {ticker: $ticker})
                    MERGE (customer:Entity {name: $customer_name})
                    MERGE (supplier)-[r:CUSTOMER_OF]->(customer)
                    SET r.revenue_pct = $revenue_pct,
                        r.source = $source
                """, 
                ticker=rel['ticker'],
                customer_name=rel['resolved_entity'],
                revenue_pct=rel['revenue_pct'],
                source=rel['source'])
        
        # Create competitor relationships
        for peer in peers:
            session.run("""
                MERGE (c1:Company {ticker: $ticker})
                MERGE (c2:Company {ticker: $peer})
                MERGE (c1)-[r:COMPETES_WITH]->(c2)
            """, ticker=peer['ticker'], peer=peer['peer'])
        
        # Create geography exposure
        for seg in segments:
            if seg['segment_type'] == 'geography':
                session.run("""
                    MERGE (c:Company {ticker: $ticker})
                    MERGE (g:Geography {name: $geo_name})
                    MERGE (c)-[r:EXPOSED_TO]->(g)
                    SET r.revenue_pct = $revenue_pct
                """, 
                ticker=seg['ticker'],
                geo_name=seg['segment_name'],
                revenue_pct=seg['revenue_pct'])
    
    driver.close()
    print("Supply chain graph loaded into Neo4j")
    return True

def calculate_graph_centrality_metrics(**context):
    """Calculate PageRank, betweenness, degree centrality"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Create graph projection
        try:
            session.run("""
                CALL gds.graph.project(
                    'supply_chain_graph',
                    'Company',
                    ['SUPPLIES_TO', 'CUSTOMER_OF', 'COMPETES_WITH']
                )
            """)
        except:
            pass  # Graph already exists
        
        # Calculate PageRank
        session.run("""
            CALL gds.pageRank.write('supply_chain_graph', {
                writeProperty: 'pagerank',
                maxIterations: 20,
                dampingFactor: 0.85
            })
        """)
        
        # Calculate Betweenness Centrality
        session.run("""
            CALL gds.betweenness.write('supply_chain_graph', {
                writeProperty: 'betweenness'
            })
        """)
        
        # Calculate Degree
        session.run("""
            MATCH (c:Company)
            SET c.degree = size((c)--())
        """)
    
    driver.close()
    print("Graph centrality metrics calculated")
    return True

def detect_communities(**context):
    """Run Louvain algorithm to detect clusters"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Run Louvain community detection
        session.run("""
            CALL gds.louvain.write('supply_chain_graph', {
                writeProperty: 'community_id'
            })
        """)
    
    driver.close()
    print("Communities detected")
    return True

def calculate_concentration_risks(**context):
    """Calculate customer, supplier, geographic concentration risks"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Flag customer concentration risk (single customer > 10%)
        session.run("""
            MATCH (c:Company)-[r:CUSTOMER_OF]->(customer)
            WHERE r.revenue_pct > 10
            SET c.customer_concentration_risk = true
        """)
        
        # Flag geographic concentration (single country > 40%)
        session.run("""
            MATCH (c:Company)-[r:EXPOSED_TO]->(g:Geography)
            WHERE r.revenue_pct > 40
            SET c.geographic_concentration_risk = true
        """)
    
    driver.close()
    print("Concentration risks calculated")
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_08_quarterly_supply_chain_graph_pipeline",
    default_args=default_args,
    description="Build supply chain relationships and calculate graph metrics",
    schedule_interval="@quarterly",
    catchup=False,
    max_active_runs=1,
    tags=["quarterly", "supply_chain", "graph", "neo4j"],
) as dag:

    wait_for_sec_filings = ExternalTaskSensor(
        task_id="wait_for_sec_filings_processed",
        external_dag_id="dag_04_quarterly_sec_filings_pipeline",
        external_task_id="load_neo4j_knowledge_graph",
        execution_delta=timedelta(days=1),
        timeout=7200,
        mode='reschedule',
    )

    extract_mentions = PythonOperator(
        task_id="extract_10k_customer_supplier_mentions",
        python_callable=extract_10k_customer_supplier_mentions,
    )

    extract_peers = PythonOperator(
        task_id="extract_fmp_stock_peers",
        python_callable=extract_fmp_stock_peers,
    )

    extract_segments = PythonOperator(
        task_id="extract_product_geography_data",
        python_callable=extract_product_geography_data,
    )

    transform = PythonOperator(
        task_id="transform_supply_chain_entities",
        python_callable=transform_supply_chain_entities,
    )

    load_graph = PythonOperator(
        task_id="load_neo4j_supply_chain_graph",
        python_callable=load_neo4j_supply_chain_graph,
    )

    calc_centrality = PythonOperator(
        task_id="calculate_graph_centrality_metrics",
        python_callable=calculate_graph_centrality_metrics,
    )

    detect_comm = PythonOperator(
        task_id="detect_communities",
        python_callable=detect_communities,
    )

    calc_risks = PythonOperator(
        task_id="calculate_concentration_risks",
        python_callable=calculate_concentration_risks,
    )

    wait_for_sec_filings >> [extract_mentions, extract_peers, extract_segments]
    [extract_mentions, extract_peers, extract_segments] >> transform >> load_graph
    load_graph >> calc_centrality >> detect_comm >> calc_risks
