import os
import argparse
from datetime import datetime
from neo4j import GraphDatabase

def seed(uri: str, user: str, password: str, ticker: str, reset: bool = False):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    strategies = [
        {"id": "S1", "title": "Services expansion", "description": "Grow Services revenue via subscriptions, payments, and ecosystem monetization."},
        {"id": "S2", "title": "On-device AI", "description": "Integrate on-device AI features across iPhone/Mac to improve UX and retention."},
        {"id": "S3", "title": "Ecosystem lock-in", "description": "Deepen integration across devices and software to reduce churn and increase ARPU."},
    ]
    risks = [
        {"id": "R1", "title": "Regulatory / antitrust", "description": "App Store/platform rules may face antitrust actions and forced changes."},
        {"id": "R2", "title": "Supply chain concentration", "description": "Manufacturing concentration creates geopolitical and disruption exposure."},
        {"id": "R3", "title": "Hardware demand cyclicality", "description": "Upgrade cycles and macro shocks can pressure revenue and margins."},
    ]
    segments = [
        {"id": "SEG1", "title": "iPhone", "description": "Core hardware revenue driver; grows installed base for Services."},
        {"id": "SEG2", "title": "Services", "description": "Recurring revenue from subscriptions, commissions, ads, and payments."},
    ]

    now = datetime.utcnow().isoformat()

    with driver.session() as session:
        if reset:
            session.run("MATCH (c:Company {ticker:$ticker}) DETACH DELETE c", ticker=ticker)

        session.run(
            """
            MERGE (c:Company {ticker:$ticker})
            ON CREATE SET c.name=$name, c.created_at=$now
            ON MATCH  SET c.updated_at=$now
            """,
            ticker=ticker, name=f"{ticker} (seeded)", now=now
        )

        for s in strategies:
            session.run(
                """
                MATCH (c:Company {ticker:$ticker})
                MERGE (n:Strategy {id:$id, ticker:$ticker})
                SET n.title=$title, n.description=$description, n.updated_at=$now, n.source="seed"
                MERGE (c)-[:HAS_STRATEGY]->(n)
                """,
                ticker=ticker, id=s["id"], title=s["title"], description=s["description"], now=now
            )

        for r in risks:
            session.run(
                """
                MATCH (c:Company {ticker:$ticker})
                MERGE (n:Risk {id:$id, ticker:$ticker})
                SET n.title=$title, n.description=$description, n.updated_at=$now, n.source="seed"
                MERGE (c)-[:FACES_RISK]->(n)
                """,
                ticker=ticker, id=r["id"], title=r["title"], description=r["description"], now=now
            )

        for seg in segments:
            session.run(
                """
                MATCH (c:Company {ticker:$ticker})
                MERGE (n:Segment {id:$id, ticker:$ticker})
                SET n.title=$title, n.description=$description, n.updated_at=$now, n.source="seed"
                MERGE (c)-[:HAS_SEGMENT]->(n)
                """,
                ticker=ticker, id=seg["id"], title=seg["title"], description=seg["description"], now=now
            )

    driver.close()
    print(f"✅ Seeded Neo4j BA graph for {ticker}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    seed(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        ticker=args.ticker,
        reset=args.reset
    )
