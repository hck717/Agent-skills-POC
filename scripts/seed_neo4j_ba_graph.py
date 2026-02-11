import os
import argparse
import glob
from datetime import datetime
from neo4j import GraphDatabase
import PyPDF2  # Ensure you have pypdf installed
import requests

# Import Semantic Chunker
# Note: In a real deployment, we'd import this from the skills module.
# For this script, we'll implement a lightweight local version or try import.
try:
    from skills.business_analyst_crag.ingestion import SemanticChunker
    from orchestrator_react import OllamaClient
    HAS_SEMANTIC_CHUNKER = True
except ImportError:
    HAS_SEMANTIC_CHUNKER = False
    print("⚠️ Semantic Chunker import failed. Falling back to basic chunking.")

class GraphSeeder:
    def __init__(self, uri, user, password, ticker):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.ticker = ticker
        self.data_dir = os.path.join("data", ticker)
        
        # Initialize Ollama for semantic chunking if available
        if HAS_SEMANTIC_CHUNKER:
            self.ollama_client = OllamaClient(base_url="http://localhost:11434", model="deepseek-r1:8b")
            self.chunker = SemanticChunker(self.ollama_client)
        else:
            self.chunker = None

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
        return text

    def get_filing_files(self):
        """Get list of PDF/TXT files in the ticker's data directory."""
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory not found: {self.data_dir}")
            return []
        
        files = glob.glob(os.path.join(self.data_dir, "*.pdf")) + \
                glob.glob(os.path.join(self.data_dir, "*.txt")) + \
                glob.glob(os.path.join(self.data_dir, "*.docx"))
        return files

    def semantic_extraction(self, text, source_file):
        """
        Use LLM to extract structured entities (Strategy, Risk, Segment) 
        from the raw text chunk.
        """
        if not HAS_SEMANTIC_CHUNKER:
            return [] # Fallback skipped for brevity
            
        # Limit text length for extraction prompt to avoid context overflow
        # In a real system, we'd iterate over chunks. 
        # Here we take the first 4000 chars as a representative sample for the POC seed.
        sample_text = text[:8000] 
        
        prompt = f"""
        Extract key business entities from the following text (e.g. from a 10-K).
        Return a valid JSON list of objects.
        
        Entities to extract:
        1. STRATEGY (e.g., "AI Expansion", "Services Growth")
        2. RISK (e.g., "Supply Chain", "Antitrust")
        3. SEGMENT (e.g., "iPhone", "Azure")
        
        TEXT:
        {sample_text}
        
        OUTPUT FORMAT (Strict JSON):
        [
            {{"type": "Strategy", "title": "...", "description": "..."}},
            {{"type": "Risk", "title": "...", "description": "..."}},
            {{"type": "Segment", "title": "...", "description": "..."}}
        ]
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.ollama_client.chat(messages, temperature=0.1, num_predict=2000)
            
            # Clean JSON
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
                
            import json
            entities = json.loads(json_str)
            return entities
            
        except Exception as e:
            print(f"⚠️ Extraction failed: {e}")
            return []

    def seed(self, reset=False):
        """Main seeding logic."""
        files = self.get_filing_files()
        if not files:
            print(f"⚠️ No files found for {self.ticker}. Skipping ingestion.")
            return

        now = datetime.utcnow().isoformat()
        
        with self.driver.session() as session:
            # 1. Reset if requested
            if reset:
                print(f"🧹 Wiping graph data for {self.ticker}...")
                session.run("MATCH (c:Company {ticker:$ticker}) DETACH DELETE c", ticker=self.ticker)
                # Also delete related nodes to ensure clean slate
                session.run("MATCH (n {ticker:$ticker}) DETACH DELETE n", ticker=self.ticker)

            # 2. Create Company Node
            session.run(
                """
                MERGE (c:Company {ticker:$ticker})
                ON CREATE SET c.name=$ticker, c.created_at=$now
                ON MATCH  SET c.updated_at=$now
                """,
                ticker=self.ticker, now=now
            )

            # 3. Process Files
            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"📄 Processing {filename}...")
                
                # Extract Text
                if file_path.endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', errors='ignore') as f:
                        text = f.read()
                
                if not text:
                    continue

                # AI Extraction
                print(f"   🧠 Extracting graph entities from {filename}...")
                entities = self.semantic_extraction(text, filename)
                
                print(f"   🔹 Found {len(entities)} entities. Writing to Neo4j...")
                
                # Write to Neo4j
                for entity in entities:
                    label = entity.get("type", "Unknown").capitalize()
                    title = entity.get("title", "Untitled")
                    desc = entity.get("description", "")
                    
                    if label == "Strategy":
                        cypher = """
                        MATCH (c:Company {ticker:$ticker})
                        MERGE (n:Strategy {title:$title, ticker:$ticker})
                        SET n.description=$desc, n.source=$source, n.updated_at=$now
                        MERGE (c)-[:HAS_STRATEGY]->(n)
                        """
                    elif label == "Risk":
                        cypher = """
                        MATCH (c:Company {ticker:$ticker})
                        MERGE (n:Risk {title:$title, ticker:$ticker})
                        SET n.description=$desc, n.source=$source, n.updated_at=$now
                        MERGE (c)-[:FACES_RISK]->(n)
                        """
                    elif label == "Segment":
                        cypher = """
                        MATCH (c:Company {ticker:$ticker})
                        MERGE (n:Segment {title:$title, ticker:$ticker})
                        SET n.description=$desc, n.source=$source, n.updated_at=$now
                        MERGE (c)-[:HAS_SEGMENT]->(n)
                        """
                    else:
                        continue
                        
                    session.run(
                        cypher,
                        ticker=self.ticker,
                        title=title,
                        desc=desc,
                        source=filename,
                        now=now
                    )
                    
        self.driver.close()
        print(f"✅ Seeding complete for {self.ticker}")

# Function wrapper for external import compatibility
def seed(uri, user, password, ticker, reset=False):
    seeder = GraphSeeder(uri, user, password, ticker)
    seeder.seed(reset=reset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    seed(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        ticker=args.ticker,
        reset=args.reset
    )
