import os
import argparse
import glob
from datetime import datetime
from neo4j import GraphDatabase
import PyPDF2
import requests
import re
import json

# Local fallback client
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
        self.ollama_client = OllamaClient(base_url="http://localhost:11434", model="deepseek-r1:8b")

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                limit = min(15, len(reader.pages))
                for i in range(limit): 
                    text += reader.pages[i].extract_text() + "\n"
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
        return text

    def get_filing_files(self):
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory not found: {self.data_dir}")
            return []
        return glob.glob(os.path.join(self.data_dir, "*.pdf")) + \
               glob.glob(os.path.join(self.data_dir, "*.txt")) + \
               glob.glob(os.path.join(self.data_dir, "*.docx"))

    def semantic_extraction(self, text, source_file):
        sample_text = text[:15000] 
        
        # Expanded prompt to explicitly allow Products and Metrics
        prompt = f"""
        Analyze this 10-K text for {self.ticker}.
        Extract 10-15 key business entities.
        
        Allowed Types: 'Strategy', 'Risk', 'Segment', 'Product', 'Metric'.
        
        TEXT SAMPLE:
        {sample_text}
        
        INSTRUCTIONS:
        Return a valid JSON list. 
        Do not include any 'thinking' text or markdown blocks outside the JSON.
        
        FORMAT:
        [
            {{"type": "Strategy", "title": "AI Expansion", "description": "Investing in generative AI..."}},
            {{"type": "Risk", "title": "Supply Chain", "description": "Dependence on China..."}},
            {{"type": "Product", "title": "Azure", "description": "Cloud computing platform..."}},
            {{"type": "Metric", "title": "Revenue Growth", "description": "18% year-over-year..."}}
        ]
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.ollama_client.chat(messages, temperature=0.1)
            
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
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
            return []

    def seed(self, reset=False):
        files = self.get_filing_files()
        if not files:
            print(f"⚠️ No files found for {self.ticker}. Skipping ingestion.")
            return

        now = datetime.utcnow().isoformat()
        
        with self.driver.session() as session:
            if reset:
                print(f"🧹 Wiping graph data for {self.ticker}...")
                # 1. Delete nodes with the specific ticker property
                session.run("MATCH (n {ticker:$ticker}) DETACH DELETE n", ticker=self.ticker)
                # 2. Cleanup: Delete any Company node with this ticker (even if property missing on some labels)
                session.run("MATCH (c:Company {name:$ticker}) DETACH DELETE c", ticker=self.ticker)

            session.run(
                """
                MERGE (c:Company {ticker:$ticker})
                ON CREATE SET c.name=$ticker, c.created_at=$now
                ON MATCH  SET c.updated_at=$now
                """,
                ticker=self.ticker, now=now
            )

            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"📄 Processing {filename}...")
                
                if file_path.endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', errors='ignore') as f:
                        text = f.read()
                
                if not text: continue

                print(f"   🧠 Extracting graph entities from {filename}...")
                entities = self.semantic_extraction(text, filename)
                print(f"   🔹 Found {len(entities)} entities. Writing to Neo4j...")
                
                for entity in entities:
                    label = entity.get("type", "Unknown").capitalize()
                    title = entity.get("title", "Untitled")
                    desc = entity.get("description", "")
                    
                    # Define Cypher based on Label
                    if label == "Strategy":
                        rel = "[:HAS_STRATEGY]"
                        node_label = "Strategy"
                    elif label == "Risk":
                        rel = "[:FACES_RISK]"
                        node_label = "Risk"
                    elif label == "Segment":
                        rel = "[:HAS_SEGMENT]"
                        node_label = "Segment"
                    elif label == "Product":
                        rel = "[:OFFERS_PRODUCT]"
                        node_label = "Product"
                    elif label == "Metric":
                        rel = "[:TRACKS_METRIC]"
                        node_label = "Metric"
                    else:
                        continue # Skip unknown types
                        
                    cypher = f"""
                    MATCH (c:Company {{ticker:$ticker}})
                    MERGE (n:{node_label} {{title:$title, ticker:$ticker}})
                    SET n.description=$desc, n.source=$source, n.updated_at=$now
                    MERGE (c)-{rel}->(n)
                    """
                        
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
