
"""
DAG 13: Neo4j Auto Ingestion Pipeline
Schedule: Daily at 11 PM HKT
Purpose: Automatically ingest CSV files from import folders into Neo4j
"""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import config
from pathlib import Path

def scan_import_folders(**context):
    """Scan all Neo4j import folders for new CSV files"""
    import_paths = [
        config.DATA_ROOT / "01_business_analyst/neo4j/import",
        config.DATA_ROOT / "03_supply_chain_graph/neo4j/import",
    ]
    
    new_files = []
    for import_path in import_paths:
        if import_path.exists():
            csv_files = list(import_path.rglob("*.csv"))
            new_files.extend([(str(f), import_path.name) for f in csv_files])
    
    context["ti"].xcom_push(key="import_files", value=new_files)
    print(f"Found {len(new_files)} CSV files to import")
    return True

def ingest_business_analyst_graph(**context):
    """Ingest Business Analyst knowledge graph"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="business_analyst") as session:
        # Create constraints
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) 
            REQUIRE c.name IS UNIQUE
        """)
        
        # Load nodes
        import_path = config.DATA_ROOT / "01_business_analyst/neo4j/import/nodes"
        
        node_files = {
            "concepts.csv": "Concept",
            "strategies.csv": "Strategy",
            "risks.csv": "Risk",
            "technologies.csv": "Technology",
            "markets.csv": "Market",
            "regulations.csv": "Regulation",
        }
        
        for filename, label in node_files.items():
            file_path = import_path / filename
            if file_path.exists():
                query = f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row
                    MERGE (n:{label} {{name: row.name}})
                    SET n.extracted_from_filing = row.extracted_from_filing,
                        n.mention_frequency = toInteger(row.mention_frequency),
                        n.temporal_validity = row.temporal_validity
                """
                session.run(query)
                print(f"Loaded {label} nodes from {filename}")
        
        # Load relationships
        rel_path = import_path.parent / "relationships"
        
        rel_files = {
            "depends_on.csv": "DEPENDS_ON",
            "impacts.csv": "IMPACTS",
            "affects.csv": "AFFECTS",
        }
        
        for filename, rel_type in rel_files.items():
            file_path = rel_path / filename
            if file_path.exists():
                query = f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row
                    MATCH (a {{name: row.source}})
                    MATCH (b {{name: row.target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.strength = toFloat(row.strength)
                """
                session.run(query)
                print(f"Loaded {rel_type} relationships from {filename}")
    
    driver.close()
    print("Business Analyst graph ingested")
    return True

def ingest_supply_chain_graph(**context):
    """Ingest Supply Chain graph"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    with driver.session(database="supply_chain") as session:
        # Create constraints
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) 
            REQUIRE c.ticker IS UNIQUE
        """)
        
        # Load company nodes
        import_path = config.DATA_ROOT / "03_supply_chain_graph/neo4j/import/nodes"
        
        companies_file = import_path / "companies.csv"
        if companies_file.exists():
            query = """
                LOAD CSV WITH HEADERS FROM 'file:///{0}' AS row
                MERGE (c:Company {{ticker: row.ticker}})
                SET c.name = row.name,
                    c.sector = row.sector,
                    c.industry = row.industry
            """.format(companies_file)
            session.run(query)
            print("Loaded Company nodes")
        
        # Load supply relationships
        rel_path = import_path.parent / "relationships"
        
        supplies_file = rel_path / "supplies_to.csv"
        if supplies_file.exists():
            query = """
                LOAD CSV WITH HEADERS FROM 'file:///{0}' AS row
                MATCH (supplier:Company {{ticker: row.supplier_ticker}})
                MATCH (customer:Company {{ticker: row.customer_ticker}})
                MERGE (supplier)-[r:SUPPLIES_TO]->(customer)
                SET r.annual_value = toFloat(row.annual_value),
                    r.dependency_score = toFloat(row.dependency_score)
            """.format(supplies_file)
            session.run(query)
            print("Loaded SUPPLIES_TO relationships")
    
    driver.close()
    print("Supply Chain graph ingested")
    return True

def calculate_graph_metrics(**context):
    """Calculate PageRank and centrality metrics"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    # Run graph algorithms
    with driver.session(database="supply_chain") as session:
        # PageRank
        session.run("""
            CALL gds.pageRank.write({
                nodeProjection: 'Company',
                relationshipProjection: 'SUPPLIES_TO',
                writeProperty: 'pagerank'
            })
        """)
        
        # Betweenness Centrality
        session.run("""
            CALL gds.betweenness.write({
                nodeProjection: 'Company',
                relationshipProjection: 'SUPPLIES_TO',
                writeProperty: 'betweenness'
            })
        """)
        
        print("Graph metrics calculated")
    
    driver.close()
    return True

default_args = config.DEFAULT_ARGS.copy()
default_args.update({"start_date": datetime(2026, 1, 1)})

with DAG(
    dag_id="dag_13_neo4j_auto_ingest_pipeline",
    default_args=default_args,
    description="Auto-ingest CSV files into Neo4j graphs",
    schedule_interval="0 23 * * *",  # 11 PM HKT daily
    catchup=False,
    tags=["daily", "neo4j", "graph"],
) as dag:

    scan_files = PythonOperator(
        task_id="scan_import_folders",
        python_callable=scan_import_folders,
    )

    ingest_ba_graph = PythonOperator(
        task_id="ingest_business_analyst_graph",
        python_callable=ingest_business_analyst_graph,
    )

    ingest_sc_graph = PythonOperator(
        task_id="ingest_supply_chain_graph",
        python_callable=ingest_supply_chain_graph,
    )

    calc_metrics = PythonOperator(
        task_id="calculate_graph_metrics",
        python_callable=calculate_graph_metrics,
    )

    scan_files >> [ingest_ba_graph, ingest_sc_graph] >> calc_metrics
