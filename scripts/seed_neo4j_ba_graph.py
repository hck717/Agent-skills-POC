import os
import argparse
import glob
from datetime import datetime
from neo4j import GraphDatabase
import PyPDF2  # Ensure you have pypdf installed
import requests
import re
import json

# Import Semantic Chunker
# Note: In a real deployment, we'd import this from the skills module.
# For this script, we'll implement a lightweight local version or try import.
try:
    from skills.business_analyst_crag.ingestion import SemanticChunker
    from orchestrator_react import OllamaClient
    HAS_SEMANTIC_CHUNKER = True
except ImportError:
    HAS_SEMANTIC_CHUNKER = False
    
    # Define a local fallback if import fails (so we can still run)
    class OllamaClient:
        def __init__(self, base_url="http://localhost:11434", model="deepseek-r1:8b"):
            self.base_url = base_url
            self.model = model
        
        def chat(self, messages, temperature=0.2, num_predict=4000):
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict}
            }
            try:
                response = requests.post(url, json=payload, timeout=600)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                print(f"Ollama Error: {e}")
                return ""

class GraphSeeder:
    def __init__(self, uri, user, password, ticker):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.ticker = ticker
        self.data_dir = os.path.join("data", ticker)
        
        # Always use our local client for this script to ensure configuration
        self.ollama_client = OllamaClient(base_url="http://localhost:11434", model="deepseek-r1:8b")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # Read first 10 pages for Seed (sufficient for Strategy/Risk/Biz Overview)
                # Reading whole 100+ page 10K takes too long for a demo seed
                limit = min(15, len(reader.pages))
                for i in range(limit): 
                    text += reader.pages[i].extract_text() + "\n"
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
        # Limit text length to avoid context overflow for the extraction model
        sample_text = text[:12000] 
        
        # 🔥 SIMPLIFIED PROMPT FOR DEEPSEEK-R1
        # Removing complex JSON schema instructions that confuse "Thinking" models.
        prompt = f"""
        Analyze this 10-K text for {self.ticker}.
        Extract 5-10 key business entities (Strategies, Risks, Segments).
        
        TEXT SAMPLE:
        {sample_text}
        
        INSTRUCTIONS:
        Return a valid JSON list. 
        Do not include any 'thinking' text or markdown blocks outside the JSON.
        
        FORMAT:
        [
            {{"type": "Strategy", "title": "AI Expansion", "description": "Investing in generative AI..."}},
            {{"type": "Risk", "title": "Supply Chain", "description": "Dependence on China..."}},
            {{"type": "Segment", "title": "Services", "description": "App Store and Cloud growth..."}}
        ]
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.ollama_client.chat(messages, temperature=0.1)
            
            # 🔥 CLEANING: Remove <think> tags
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # Extract JSON block
            if "```json" in clean_response:
                json_str = clean_response.split("```json")[1].split("```")[0]
            elif "```" in clean_response:
                json_str = clean_response.split("```")[1].split("```")[0]
            else:
                json_str = clean_response
                
            entities = json.loads(json_str)
            return entities
            
        except Exception as e:
            print(f"⚠️ Extraction failed: {e}")
            # print(f"DEBUG Response: {response[:200]}...")
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
                    print("   ⚠️ No text extracted.")
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
