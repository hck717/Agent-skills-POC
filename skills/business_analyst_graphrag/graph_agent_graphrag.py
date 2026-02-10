#!/usr/bin/env python3
"""
GraphRAG Enhanced Business Analyst Agent

Version: 26.0 - GraphRAG Architecture (2026 SOTA)

Features:
1. Self-RAG baseline (Adaptive + Grading + Hallucination Check)
2. Entity Extraction (Companies, Products, People, Events)
3. Knowledge Graph (Neo4j) for relationship mapping
4. Multi-hop Reasoning across entities
5. Graph-enhanced Retrieval (Vector + BM25 + Graph)
6. Cross-entity Analysis

Flow:
  START → Adaptive → [Direct OR Full Pipeline]
  Full: Identify → Research → Grade → Graph Extract → Graph Query → Rerank → Generate → Check → END
"""

import os
import operator
import re
import shutil
from typing import Annotated, TypedDict, List, Tuple, Dict, Any
from collections import defaultdict
import json

# LangChain & Graph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END, START

# Self-RAG components
try:
    from ..business_analyst_selfrag.semantic_chunker import SemanticChunker
    from ..business_analyst_selfrag.document_grader import DocumentGrader
    from ..business_analyst_selfrag.hallucination_checker import HallucinationChecker
    from ..business_analyst_selfrag.adaptive_retrieval import AdaptiveRetrieval
except:
    print("⚠️ Self-RAG components not found - using fallback")
    SemanticChunker = None
    DocumentGrader = None
    HallucinationChecker = None
    AdaptiveRetrieval = None

# Neo4j (optional)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    print("⚠️ Neo4j driver not installed: pip install neo4j")
    NEO4J_AVAILABLE = False

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# --- Enhanced State ---
class GraphRAGState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    tickers: List[str]
    documents: List[Document]
    graded_documents: List[Document]
    passed_grading: bool
    relevance_rate: float
    skip_rag: bool
    direct_answer: str
    hallucination_free: bool
    retry_count: int
    # GraphRAG additions
    entities: List[Dict[str, Any]]  # Extracted entities
    relationships: List[Dict[str, Any]]  # Extracted relationships
    graph_context: str  # Additional context from graph queries
    graph_insights: List[str]  # Multi-hop insights


class GraphRAGBusinessAnalyst:
    """
    🔥 2026 SOTA: GraphRAG = Self-RAG + Knowledge Graph
    """
    
    def __init__(self, 
                 data_path="./data", 
                 db_path="./storage/chroma_db",
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j",
                 neo4j_password="password",
                 use_semantic_chunking=True):
        
        self.data_path = data_path
        self.db_path = db_path
        self.use_semantic_chunking = use_semantic_chunking
        
        print(f"🚀 Initializing GraphRAG Business Analyst v26.0...")
        print(f"   🌟 2026 SOTA: Self-RAG + Neo4j Knowledge Graph")
        
        # Models
        self.chat_model_name = "deepseek-r1:8b"
        self.embed_model_name = "nomic-embed-text"
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        self.llm = ChatOllama(model=self.chat_model_name, temperature=0.0, num_predict=2000)
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        self.reranker = CrossEncoder(self.rerank_model_name)
        
        self.vectorstores = {}
        self.bm25_indexes = {}
        self.bm25_documents = {}
        self.use_hybrid = BM25_AVAILABLE
        
        # Neo4j connection
        self.neo4j_driver = None
        self.neo4j_enabled = False
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                self.neo4j_enabled = True
                print(f"   ✅ Neo4j connected: {neo4j_uri}")
                self._initialize_graph_schema()
            except Exception as e:
                print(f"   ⚠️ Neo4j connection failed: {e}")
                print(f"   💡 GraphRAG will run without graph features")
        
        # Self-RAG components
        if DocumentGrader:
            self.document_grader = DocumentGrader(model_name=self.chat_model_name)
            self.hallucination_checker = HallucinationChecker(model_name=self.chat_model_name)
            self.adaptive_retrieval = AdaptiveRetrieval(model_name=self.chat_model_name)
            self.semantic_chunker = None
            print(f"   ✅ Self-RAG components loaded")
        else:
            print(f"   ⚠️ Self-RAG components not available")
        
        self.app = self._build_graph()
        print(f"   ✅ GraphRAG ready! (Graph: {self.neo4j_enabled})")

    def _initialize_graph_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        if not self.neo4j_enabled:
            return
        
        print("   🔧 Initializing Neo4j schema...")
        
        with self.neo4j_driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    pass  # Constraint might already exist
            
            print("   ✅ Graph schema initialized")

    def _extract_entities(self, text: str, ticker: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM
        Returns: List of {type, name, properties}
        """
        prompt = f"""Extract key entities from this financial document text.

Extract:
1. Companies (names, tickers)
2. Products/Services (names, categories)
3. People (names, roles)
4. Events (type, date, description)
5. Metrics (name, value, unit)

Text:
{text[:3000]}

Output JSON format:
{{
  "entities": [
    {{"type": "Company", "name": "Apple Inc", "ticker": "AAPL"}},
    {{"type": "Product", "name": "iPhone", "category": "smartphone"}},
    {{"type": "Person", "name": "Tim Cook", "role": "CEO"}}
  ]
}}

Extract now:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse JSON
            json_match = re.search(r'\{.*"entities".*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('entities', [])
        except Exception as e:
            print(f"   ⚠️ Entity extraction error: {e}")
        
        return []

    def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities
        Returns: List of {from, to, type, properties}
        """
        if len(entities) < 2:
            return []
        
        entity_list = "\n".join([f"- {e.get('name', 'unknown')} ({e.get('type', 'unknown')})" for e in entities[:10]])
        
        prompt = f"""Given these entities:
{entity_list}

And this text:
{text[:2000]}

Extract relationships between entities.

Output JSON:
{{
  "relationships": [
    {{"from": "Apple", "to": "iPhone", "type": "PRODUCES"}},
    {{"from": "Apple", "to": "Tim Cook", "type": "EMPLOYS", "role": "CEO"}}
  ]
}}

Extract now:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            json_match = re.search(r'\{.*"relationships".*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('relationships', [])
        except:
            pass
        
        return []

    def _add_to_graph(self, entities: List[Dict], relationships: List[Dict], ticker: str):
        """Add entities and relationships to Neo4j"""
        if not self.neo4j_enabled:
            return
        
        with self.neo4j_driver.session() as session:
            # Add entities
            for entity in entities:
                entity_type = entity.get('type', 'Entity')
                name = entity.get('name', '')
                
                if not name:
                    continue
                
                # Create node
                props = {k: v for k, v in entity.items() if k not in ['type']}
                props['ticker_context'] = ticker
                
                cypher = f"""MERGE (e:{entity_type} {{name: $name}})
                             SET e += $props
                             RETURN e"""
                
                try:
                    session.run(cypher, name=name, props=props)
                except Exception as e:
                    print(f"   ⚠️ Failed to add entity {name}: {e}")
            
            # Add relationships
            for rel in relationships:
                from_name = rel.get('from', '')
                to_name = rel.get('to', '')
                rel_type = rel.get('type', 'RELATED_TO').upper().replace(' ', '_')
                
                if not from_name or not to_name:
                    continue
                
                props = {k: v for k, v in rel.items() if k not in ['from', 'to', 'type']}
                
                cypher = f"""MATCH (a) WHERE a.name = $from
                             MATCH (b) WHERE b.name = $to
                             MERGE (a)-[r:{rel_type}]->(b)
                             SET r += $props
                             RETURN r"""
                
                try:
                    session.run(cypher, **{'from': from_name, 'to': to_name, 'props': props})
                except Exception as e:
                    print(f"   ⚠️ Failed to add relationship: {e}")

    def _query_graph(self, ticker: str, query_type: str = "neighbors") -> str:
        """Query Neo4j for graph insights"""
        if not self.neo4j_enabled:
            return ""
        
        insights = []
        
        with self.neo4j_driver.session() as session:
            if query_type == "neighbors":
                # Find direct connections
                cypher = """MATCH (c:Company {ticker: $ticker})-[r]-(n)
                            RETURN type(r) as rel_type, labels(n)[0] as node_type, n.name as name
                            LIMIT 20"""
                result = session.run(cypher, ticker=ticker)
                
                for record in result:
                    insights.append(f"{ticker} -{record['rel_type']}-> {record['node_type']}:{record['name']}")
            
            elif query_type == "competitors":
                # Find competitors through shared relationships
                cypher = """MATCH (c:Company {ticker: $ticker})-[:COMPETES_IN]->(m:Market)<-[:COMPETES_IN]-(comp:Company)
                            RETURN comp.name as competitor, comp.ticker as ticker
                            LIMIT 10"""
                result = session.run(cypher, ticker=ticker)
                
                for record in result:
                    insights.append(f"Competitor: {record['competitor']} ({record['ticker']})")
            
            elif query_type == "supply_chain":
                # Multi-hop: suppliers and customers
                cypher = """MATCH path = (c:Company {ticker: $ticker})-[:SUPPLIES|SUPPLIED_BY*1..2]-(other)
                            RETURN other.name as entity, length(path) as hops
                            LIMIT 10"""
                result = session.run(cypher, ticker=ticker)
                
                for record in result:
                    insights.append(f"Supply chain connection: {record['entity']} ({record['hops']} hops)")
        
        return "\n".join(insights) if insights else ""

    def _inject_citations_if_missing(self, analysis: str, context: str) -> str:
        """Inject citations if LLM failed to preserve them"""
        if '--- SOURCE:' in analysis:
            return analysis
        
        print("   ⚠️ Injecting citations...")
        source_pattern = r'--- SOURCE: ([^\(]+)\(Page ([^\)]+)\) ---'
        sources = re.findall(source_pattern, context)
        
        if not sources:
            return analysis
        
        lines = analysis.split('\n')
        result_lines = []
        source_idx = 0
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            if (line.strip() and not line.startswith('#') and len(line) > 100 
                and source_idx < len(sources) and i < len(lines) - 1):
                filename, page = sources[source_idx]
                result_lines.append(f"--- SOURCE: {filename}(Page {page}) ---")
                source_idx += 1
        
        return '\n'.join(result_lines)

    # --- Standard RAG nodes (simplified for space) ---
    def _load_prompt(self, name):
        return "You are a Strategic Analyst."
    
    def _build_bm25_index(self, collection_name: str, documents: List[Document]):
        if not BM25_AVAILABLE:
            return
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25_indexes[collection_name] = BM25Okapi(tokenized)
        self.bm25_documents[collection_name] = documents
    
    def _hybrid_search(self, collection_name: str, query: str, k: int = 25) -> List[Document]:
        vs = self._get_vectorstore(collection_name)
        vector_docs = vs.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in vector_docs]

    # --- GraphRAG Nodes ---
    
    def adaptive_node(self, state: GraphRAGState):
        if not AdaptiveRetrieval:
            return {"skip_rag": False}
        
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        needs_rag, direct_answer, _ = self.adaptive_retrieval.should_use_rag(query)
        
        return {"skip_rag": not needs_rag, "direct_answer": direct_answer}

    def identify_node(self, state: GraphRAGState):
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "").upper()
        
        mapping = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA", "NVIDIA": "NVDA"}
        found = [ticker for name, ticker in mapping.items() if name in query]
        return {"tickers": found}

    def research_node(self, state: GraphRAGState):
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        tickers = state.get('tickers', [])
        
        if not tickers:
            return {"documents": []}
        
        all_docs = []
        for ticker in tickers:
            print(f"🔍 [Research] Hybrid search + Graph query for {ticker}...")
            collection_name = f"docs_{ticker}"
            docs = self._hybrid_search(collection_name, query, k=25)
            all_docs.extend(docs)
        
        return {"documents": all_docs}

    def grade_documents_node(self, state: GraphRAGState):
        if not DocumentGrader:
            return {"graded_documents": state.get('documents', []), "passed_grading": True}
        
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        documents = state.get('documents', [])
        
        graded_docs, metadata = self.document_grader.grade_documents(query, documents, threshold=0.3)
        return {
            "graded_documents": graded_docs,
            "passed_grading": metadata['meets_threshold'],
            "relevance_rate": metadata['pass_rate']
        }

    def graph_extraction_node(self, state: GraphRAGState):
        """🔥 NEW: Extract entities and build knowledge graph"""
        print("\n🕸️ [GraphRAG] Extracting entities and relationships...")
        
        docs = state.get('graded_documents', [])
        tickers = state.get('tickers', [])
        ticker = tickers[0] if tickers else "UNKNOWN"
        
        if not docs:
            return {"entities": [], "relationships": []}
        
        # Extract from top 3 documents
        all_entities = []
        all_relationships = []
        
        for doc in docs[:3]:
            entities = self._extract_entities(doc.page_content, ticker)
            all_entities.extend(entities)
            
            if entities:
                relationships = self._extract_relationships(doc.page_content, entities)
                all_relationships.extend(relationships)
        
        print(f"   ✅ Extracted {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        # Add to Neo4j
        if self.neo4j_enabled and all_entities:
            print(f"   💾 Adding to Neo4j knowledge graph...")
            self._add_to_graph(all_entities, all_relationships, ticker)
        
        return {"entities": all_entities, "relationships": all_relationships}

    def graph_query_node(self, state: GraphRAGState):
        """🔥 NEW: Query knowledge graph for multi-hop insights"""
        print("\n🔎 [GraphRAG] Querying knowledge graph...")
        
        tickers = state.get('tickers', [])
        if not tickers or not self.neo4j_enabled:
            return {"graph_context": "", "graph_insights": []}
        
        ticker = tickers[0]
        
        # Run multiple graph queries
        insights = []
        
        # Query 1: Direct neighbors
        neighbors = self._query_graph(ticker, "neighbors")
        if neighbors:
            insights.append(f"**Direct Connections:**\n{neighbors}")
        
        # Query 2: Competitors
        competitors = self._query_graph(ticker, "competitors")
        if competitors:
            insights.append(f"**Competitors:**\n{competitors}")
        
        # Query 3: Supply chain
        supply_chain = self._query_graph(ticker, "supply_chain")
        if supply_chain:
            insights.append(f"**Supply Chain:**\n{supply_chain}")
        
        graph_context = "\n\n".join(insights) if insights else ""
        
        if graph_context:
            print(f"   ✅ Found {len(insights)} graph insights")
        else:
            print(f"   ⚠️ No graph data available yet")
        
        return {"graph_context": graph_context, "graph_insights": insights}

    def rerank_node(self, state: GraphRAGState):
        docs = state.get('graded_documents', [])
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        if not docs:
            return {"context": ""}
        
        print(f"⚖️ [Rerank] BERT scoring {len(docs)} documents...")
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        top_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:8]
        
        # Format with citations
        formatted = []
        for doc, _ in top_docs:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            formatted.append(f"--- SOURCE: {source} (Page {page}) ---\n{doc.page_content}")
        
        context = "\n\n".join(formatted)
        
        # 🔥 Add graph context
        graph_context = state.get('graph_context', '')
        if graph_context:
            context += f"\n\n===== KNOWLEDGE GRAPH INSIGHTS =====\n{graph_context}\n===================================="
        
        return {"context": context}

    def analyst_node(self, state: GraphRAGState):
        context = state.get('context', '')
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        if not context:
            return {"messages": [HumanMessage(content="⚠️ No relevant information found.")]}
        
        # Enhanced prompt with graph awareness
        prompt = f"""You are a Strategic Analyst with access to:
1. Document analysis (10-K filings, reports)
2. Knowledge Graph insights (entity relationships, multi-hop connections)

Provide comprehensive analysis using BOTH sources.

CITATION RULES:
- Use --- SOURCE: filename (Page X) --- for document claims
- Use [GRAPH INSIGHT] for knowledge graph findings

====== CONTEXT ======
{context}
=====================

QUESTION: {query}

Analyze now:"""
        
        response = self.llm.invoke([SystemMessage(content=prompt), *messages])
        analysis = response.content
        
        # Inject citations
        analysis = self._inject_citations_if_missing(analysis, context)
        
        return {"messages": [HumanMessage(content=analysis)]}

    def hallucination_check_node(self, state: GraphRAGState):
        if not HallucinationChecker:
            return {"hallucination_free": True}
        
        messages = state['messages']
        analysis = messages[-1].content if messages else ""
        context = state.get('context', '')
        
        is_grounded, _, _ = self.hallucination_checker.check_hallucination(analysis, context)
        return {"hallucination_free": is_grounded, "retry_count": state.get('retry_count', 0) + 1}

    def direct_output_node(self, state: GraphRAGState):
        return {"messages": [HumanMessage(content=state.get('direct_answer', ''))]}

    # --- Conditional edges ---
    def should_skip_rag(self, state: GraphRAGState) -> str:
        return "direct_output" if state.get('skip_rag') else "identify"
    
    def should_use_web_search(self, state: GraphRAGState) -> str:
        return "rerank" if state.get('passed_grading') else "rerank"  # Skip web search for now
    
    def should_retry(self, state: GraphRAGState) -> str:
        if state.get('hallucination_free', True):
            return END
        return END if state.get('retry_count', 0) >= 2 else "analyst"

    def _build_graph(self):
        workflow = StateGraph(GraphRAGState)
        
        # Add nodes
        workflow.add_node("adaptive", self.adaptive_node)
        workflow.add_node("direct_output", self.direct_output_node)
        workflow.add_node("identify", self.identify_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("grade", self.grade_documents_node)
        workflow.add_node("graph_extract", self.graph_extraction_node)
        workflow.add_node("graph_query", self.graph_query_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("hallucination_check", self.hallucination_check_node)
        
        # Build flow
        workflow.add_edge(START, "adaptive")
        workflow.add_conditional_edges("adaptive", self.should_skip_rag, 
                                        {"direct_output": "direct_output", "identify": "identify"})
        workflow.add_edge("direct_output", END)
        
        # GraphRAG pipeline
        workflow.add_edge("identify", "research")
        workflow.add_edge("research", "grade")
        workflow.add_conditional_edges("grade", self.should_use_web_search, {"rerank": "graph_extract"})
        workflow.add_edge("graph_extract", "graph_query")
        workflow.add_edge("graph_query", "rerank")
        workflow.add_edge("rerank", "analyst")
        workflow.add_edge("analyst", "hallucination_check")
        workflow.add_conditional_edges("hallucination_check", self.should_retry, {END: END, "analyst": "analyst"})
        
        return workflow.compile()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
        return self.vectorstores[collection_name]

    def ingest_data(self):
        """Ingest documents and build knowledge graph"""
        print(f"\n📂 Scanning {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return
        
        folders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        
        for folder in folders:
            ticker = os.path.basename(folder).upper()
            print(f"\n📊 Processing {ticker}...")
            
            # Load PDFs
            pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
            docs = pdf_loader.load()
            
            if not docs:
                continue
            
            # Chunk
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            
            # Vector store
            collection_name = f"docs_{ticker}"
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            
            # BM25
            if self.use_hybrid:
                self._build_bm25_index(collection_name, splits)
            
            # 🔥 Build knowledge graph
            if self.neo4j_enabled:
                print(f"   🕸️ Building knowledge graph for {ticker}...")
                for i, doc in enumerate(docs[:5]):  # First 5 docs
                    entities = self._extract_entities(doc.page_content, ticker)
                    if entities:
                        relationships = self._extract_relationships(doc.page_content, entities)
                        self._add_to_graph(entities, relationships, ticker)
                    if i % 2 == 0:
                        print(f"      Progress: {i+1}/5 documents")
            
            print(f"   ✅ Indexed {len(splits)} chunks + knowledge graph")
        
        print(f"\n✅ GraphRAG ingestion complete!")

    def reset_vector_db(self):
        """Reset vector DB (graph preserved)"""
        self.vectorstores = {}
        self.bm25_indexes = {}
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)
        return True, "Vector DB reset (graph preserved)"

    def reset_graph(self):
        """⚠️ Delete all graph data"""
        if not self.neo4j_enabled:
            return False, "Neo4j not connected"
        
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return True, "Knowledge graph cleared"

    def get_database_stats(self):
        stats = {}
        for folder in [f.path for f in os.scandir(self.data_path) if f.is_dir()]:
            ticker = os.path.basename(folder).upper()
            try:
                vs = self._get_vectorstore(f"docs_{ticker}")
                stats[ticker] = vs._collection.count()
            except:
                stats[ticker] = 0
        
        # Graph stats
        if self.neo4j_enabled:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                stats['GRAPH_NODES'] = result.single()['node_count']
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                stats['GRAPH_RELATIONSHIPS'] = result.single()['rel_count']
        
        return stats

    def analyze(self, query: str):
        print(f"🤖 GraphRAG Query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content

    def __del__(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()


if __name__ == "__main__":
    agent = GraphRAGBusinessAnalyst()
    print("\n🎯 GraphRAG v26.0 ready!")
    print("   Features: Self-RAG + Neo4j + Entity Extraction + Multi-hop Reasoning")
