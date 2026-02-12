#!/usr/bin/env python3
"""
Proper Document Ingestion for Business Analyst v4.2

Uses proposition-based chunking with embeddings for CRAG retrieval.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from skills.business_analyst_crag.ingestion import SemanticChunker
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from dotenv import load_dotenv

load_dotenv()

class DocumentIngester:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, ticker):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.ticker = ticker.strip().upper()
        self.data_dir = os.path.join("data", self.ticker)
        
        # Initialize chunker and embedder
        print("\n🔧 Initializing components...")
        
        # Create simple Ollama client for chunker
        import ollama
        self.chunker = SemanticChunker(ollama_client=ollama)
        print("  ✅ Semantic chunker ready (using Ollama)")
        
        # Use CPU for embeddings (M3 Mac compatible)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("  ✅ Embedder ready (CPU mode)\n")
    
    def extract_text_from_file(self, file_path):
        """Extract text from PDF or DOCX"""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    # Read first 20 pages (or all if less)
                    for i in range(min(20, len(reader.pages))):
                        text += reader.pages[i].extract_text() + "\n"
                    return text
            
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            
            else:
                # Try reading as text
                with open(file_path, 'r', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            return ""
    
    def ingest(self, reset=False, max_chunks=None):
        """Ingest documents with proposition-based chunking"""
        
        # Find files
        files = glob.glob(os.path.join(self.data_dir, "*"))
        files = [f for f in files if os.path.isfile(f) and not f.endswith('.DS_Store')]
        
        if not files:
            print(f"❌ No files found in {self.data_dir}")
            return
        
        print(f"📁 Found {len(files)} file(s) in {self.data_dir}\n")
        
        with self.driver.session() as session:
            # Reset if requested
            if reset:
                print(f"🧹 Clearing existing {self.ticker} data...")
                session.run("""
                    MATCH (c:Company {ticker: $ticker})
                    DETACH DELETE c
                """, ticker=self.ticker)
                session.run("""
                    MATCH (n:Chunk {ticker: $ticker})
                    DELETE n
                """, ticker=self.ticker)
                print("  ✅ Cleared\n")
            
            # Create company node
            print(f"🏗️  Creating Company node: {self.ticker}")
            session.run("""
                MERGE (c:Company {ticker: $ticker})
                ON CREATE SET c.name = $ticker
            """, ticker=self.ticker)
            print("  ✅ Company created\n")
            
            # Process each file
            total_chunks = 0
            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"📄 Processing: {filename}")
                
                # Extract text
                text = self.extract_text_from_file(file_path)
                if not text or len(text) < 100:
                    print(f"  ⚠️  Skipped (empty or too short)\n")
                    continue
                
                print(f"  📝 Extracted {len(text)} characters")
                
                # Limit text size (first 50k chars to avoid timeout)
                if len(text) > 50000:
                    text = text[:50000]
                    print(f"  ✂️  Truncated to 50k chars")
                
                # Chunk with proposition-based method
                print(f"  🔪 Chunking with LLM (proposition-based)...")
                try:
                    chunks = self.chunker.chunk_by_proposition(text)
                    print(f"  ✅ Created {len(chunks)} proposition chunks")
                except Exception as e:
                    print(f"  ❌ Chunking failed: {e}")
                    print(f"  🔄 Falling back to sentence splitting...")
                    try:
                        chunks = self.chunker._fallback_sentence_split(text)
                        print(f"  ✅ Created {len(chunks)} sentence chunks (fallback)")
                    except Exception as e2:
                        print(f"  ❌ Fallback also failed: {e2}\n")
                        continue
                
                # Limit chunks if requested
                if max_chunks and len(chunks) > max_chunks:
                    chunks = chunks[:max_chunks]
                    print(f"  ✂️  Limited to {max_chunks} chunks")
                
                # Create chunk nodes with embeddings
                print(f"  🧬 Generating embeddings...")
                for i, chunk_text in enumerate(chunks):
                    # Generate embedding
                    try:
                        embedding = self.embedder.encode(chunk_text, device='cpu', show_progress_bar=False)
                        embedding_list = embedding.tolist()
                    except Exception as e:
                        print(f"    ⚠️  Embedding failed for chunk {i}: {e}")
                        continue
                    
                    # Verify embedding is valid
                    if sum(embedding_list) == 0:
                        print(f"    ⚠️  Zero embedding for chunk {i}, skipping")
                        continue
                    
                    # Create chunk node
                    session.run("""
                        MATCH (c:Company {ticker: $ticker})
                        CREATE (chunk:Chunk {
                            text: $text,
                            embedding: $embedding,
                            ticker: $ticker,
                            chunk_id: $chunk_id,
                            source: $source
                        })
                        CREATE (c)-[:HAS_CHUNK]->(chunk)
                    """, 
                        ticker=self.ticker,
                        text=chunk_text,
                        embedding=embedding_list,
                        chunk_id=f"{self.ticker}_{filename}_{i}",
                        source=filename
                    )
                    
                    total_chunks += 1
                    if (i + 1) % 5 == 0:
                        print(f"    ✅ {i + 1}/{len(chunks)} chunks created")
                
                print(f"  ✅ {len(chunks)} chunks ingested\n")
            
            # Create vector index
            print("📊 Creating vector index...")
            try:
                session.run("""
                    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                    FOR (n:Chunk) ON (n.embedding)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: 384,
                      `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("  ✅ Vector index created\n")
            except Exception as e:
                print(f"  ⚠️  Vector index warning: {e}\n")
        
        # Verify
        print("="*80)
        print("📊 Verification")
        print("="*80 + "\n")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})
                RETURN c.name as name
            """, ticker=self.ticker)
            company = result.single()
            print(f"✅ Company: {company['name'] if company else 'NOT FOUND'}")
            
            result = session.run("""
                MATCH (n:Chunk {ticker: $ticker})
                RETURN count(n) as count
            """, ticker=self.ticker)
            chunk_count = result.single()['count']
            print(f"✅ Chunks: {chunk_count}")
            
            if chunk_count > 0:
                result = session.run("""
                    MATCH (n:Chunk {ticker: $ticker})
                    RETURN n.text as text, n.embedding[0] as first_val
                    LIMIT 1
                """, ticker=self.ticker)
                sample = result.single()
                if sample:
                    print(f"\n✅ Sample chunk:")
                    print(f"  Text: {sample['text'][:80]}...")
                    print(f"  Embedding[0]: {sample['first_val']:.6f}")
        
        self.driver.close()
        
        print("\n" + "="*80)
        print(f"✅ Ingestion Complete: {self.ticker}")
        print("="*80)
        print(f"\n📊 Total chunks created: {total_chunks}")
        print(f"🚀 Ready to test Business Analyst!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents with proposition-based chunking")
    parser.add_argument("--ticker", required=True, help="Company ticker (e.g., MSFT)")
    parser.add_argument("--reset", action="store_true", help="Clear existing data before ingestion")
    parser.add_argument("--max-chunks", type=int, help="Limit chunks per document (for testing)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"Document Ingestion for Business Analyst v4.2")
    print("="*80)
    print(f"\nTicker: {args.ticker.upper()}")
    print(f"Reset: {args.reset}")
    print(f"Max Chunks: {args.max_chunks or 'Unlimited'}")
    
    ingester = DocumentIngester(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pass=os.getenv("NEO4J_PASSWORD", "password"),
        ticker=args.ticker
    )
    
    ingester.ingest(reset=args.reset, max_chunks=args.max_chunks)
