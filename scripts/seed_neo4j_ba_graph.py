import os
import argparse
import glob
import time
from datetime import datetime
from neo4j import GraphDatabase
import PyPDF2
import requests
import re
import json

class GraphSeeder:
    def __init__(self, uri, user, password, ticker):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.ticker = ticker.strip().upper()
        self.data_dir = os.path.join("data", self.ticker)
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "deepseek-r1:8b"

    def chat(self, prompt, retries=3):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 4000, "num_ctx": 8192} 
        }
        
        for attempt in range(retries):
            try:
                # REMOVED timeout=600 to allow unlimited processing time
                response = requests.post(self.ollama_url, json=payload) 
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                print(f"   ⚠️ Ollama Error (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2) 
        
        print("   ❌ Ollama failed after all retries.")
        return ""

    def extract_text(self, file_path):
        text = ""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for i in range(min(5, len(reader.pages))): 
                        text += reader.pages[i].extract_text() + "\n"
            else:
                with open(file_path, 'r', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
        return text

    def get_entities(self, text):
        chunk = text[:6000] 
        
        prompt = f"""
        Extract 10 key business entities from this text for {self.ticker}.
        Allowed Types: Strategy, Risk, Segment, Product, Metric.
        
        Text: {chunk}...
        
        Output ONLY valid JSON list:
        [
            {{"type": "Product", "title": "Azure", "description": "Cloud platform..."}},
            {{"type": "Metric", "title": "Revenue Growth", "description": "20% increase..."}}
        ]
        """
        response = self.chat(prompt)
        clean_json = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        match = re.search(r'\[.*\]', clean_json, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return []

    def seed(self, reset=False):
        files = glob.glob(os.path.join(self.data_dir, "*"))
        if not files:
            print(f"⚠️ No files in {self.data_dir}")
            return

        with self.driver.session() as session:
            if reset:
                print(f"🧹 Nuclear Wipe for {self.ticker}...")
                session.run("MATCH (c:Company {ticker: $t}) DETACH DELETE c", t=self.ticker)
                session.run("MATCH (n {ticker: $t}) DETACH DELETE n", t=self.ticker)

            print(f"🏗️  Creating Company Node: {self.ticker}")
            session.run("""
                MERGE (c:Company {ticker: $t})
                ON CREATE SET c.name = $t, c.created_at = datetime()
                ON MATCH SET c.updated_at = datetime()
            """, t=self.ticker)

            for file_path in files:
                if not os.path.isfile(file_path): continue
                filename = os.path.basename(file_path)
                print(f"📄 Processing {filename}...")
                
                text = self.extract_text(file_path)
                if not text: continue
                
                entities = self.get_entities(text)
                print(f"   🔹 Found {len(entities)} entities. Linking to Graph...")
                
                for e in entities:
                    etype = e.get("type", "Unknown").capitalize()
                    title = e.get("title", "Untitled")
                    desc = e.get("description", "")
                    
                    rel_map = {
                        "Strategy": "HAS_STRATEGY",
                        "Risk": "FACES_RISK",
                        "Segment": "HAS_SEGMENT",
                        "Product": "OFFERS_PRODUCT",
                        "Metric": "TRACKS_METRIC"
                    }
                    rel_type = rel_map.get(etype)
                    if not rel_type: continue

                    query = f"""
                        MATCH (c:Company {{ticker: $ticker}})
                        MERGE (n:{etype} {{title: $title, ticker: $ticker}})
                        ON CREATE SET n.description = $desc, n.source = $source
                        MERGE (c)-[:{rel_type}]->(n)
                    """
                    session.run(query, ticker=self.ticker, title=title, desc=desc, source=filename)
        
        self.driver.close()
        print(f"✅ Seeding Complete: {self.ticker}")

def seed(uri, user, password, ticker, reset=False):
    GraphSeeder(uri, user, password, ticker).seed(reset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    seed(os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
         os.getenv("NEO4J_USER", "neo4j"), 
         os.getenv("NEO4J_PASSWORD", "password"), 
         args.ticker, args.reset)
