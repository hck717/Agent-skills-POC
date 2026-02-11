import os
from neo4j import GraphDatabase
from tabulate import tabulate # Optional, falls back to print if not installed

# Configuration
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def check_graph():
    print(f"\n🏥 NEO4J GRAPH HEALTH CHECK")
    print(f"==========================================")
    print(f"🔌 Connecting to {URI}...")
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        with driver.session() as session:
            # 1. Connectivity Test
            print("✅ Connection Successful!")
            
            # 2. Node Counts
            print(f"\n📊 NODE COUNTS by Label")
            print(f"------------------------------------------")
            result = session.run("MATCH (n) RETURN labels(n) as Label, count(*) as Count ORDER BY Count DESC")
            records = [r for r in result]
            if not records:
                print("⚠️ Graph is EMPTY! (Did seeding run?)")
            else:
                for r in records:
                    label = r["Label"][0] if r["Label"] else "No Label"
                    print(f"   • {label:<15} : {r['Count']}")

            # 3. Relationship Counts
            print(f"\n🔗 RELATIONSHIPS")
            print(f"------------------------------------------")
            result = session.run("MATCH ()-[r]->() RETURN type(r) as Type, count(*) as Count")
            for r in result:
                print(f"   • {r['Type']:<15} : {r['Count']}")

            # 4. Content Sampling (Quality Check)
            print(f"\n📝 CONTENT SAMPLING (Top 2 per category)")
            print(f"------------------------------------------")
            
            for label in ["Strategy", "Risk", "Segment"]:
                print(f"\n🔹 {label.upper()} Samples:")
                query = f"MATCH (n:{label}) RETURN n.title as title, n.description as desc, n.source as source LIMIT 2"
                samples = session.run(query)
                found = False
                for s in samples:
                    found = True
                    print(f"   Title: {s['title']}")
                    print(f"   Source: {s['source']}")
                    # Truncate description for display
                    desc_preview = (s['desc'][:100] + '...') if s['desc'] and len(s['desc']) > 100 else s['desc']
                    print(f"   Desc:  {desc_preview}\n")
                
                if not found:
                    print("   (No nodes found)")

            # 5. Orphan Check
            orphans = session.run("MATCH (n) WHERE NOT (n)--() RETURN count(n) as c").single()["c"]
            if orphans > 0:
                print(f"\n⚠️ WARNING: Found {orphans} disconnected (orphan) nodes.")
            else:
                print(f"\n✅ Connectivity: All nodes are connected.")

        driver.close()
        print(f"\n==========================================")
        print(f"🏥 Check Complete")

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("💡 Check if Docker is running: docker ps")

if __name__ == "__main__":
    check_graph()
