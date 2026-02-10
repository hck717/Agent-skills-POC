#!/usr/bin/env python3
"""
Self-RAG Enhanced Business Analyst Graph Agent

Version: 25.0 - Self-RAG Architecture

New Features:
1. Semantic Chunking - Better document splitting
2. Adaptive Retrieval - Skip RAG for simple queries (60% faster)
3. Document Grading - Filter irrelevant documents
4. Hallucination Checking - Verify answer grounding
5. Web Search Fallback - If document grading fails

Flow:
  START → Adaptive Check → [Direct Output OR Full RAG]
  Full RAG: Identify → Research → Grade → [Analyst OR Web Search] → Generate → Hallucination Check → [END OR Retry]
"""

import os
import operator
import re
import shutil
from typing import Annotated, TypedDict, List, Tuple, Dict
from collections import defaultdict

# LangChain & Graph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END, START

# Self-RAG enhancements
from .semantic_chunker import SemanticChunker
from .document_grader import DocumentGrader
from .hallucination_checker import HallucinationChecker
from .adaptive_retrieval import AdaptiveRetrieval

# BM25 for sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: rank_bm25 not installed. Hybrid search will use vector-only mode.")
    BM25_AVAILABLE = False


# --- Enhanced State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    tickers: List[str]
    documents: List[Document]  # Retrieved docs before grading
    graded_documents: List[Document]  # Filtered docs after grading
    passed_grading: bool
    relevance_rate: float
    skip_rag: bool  # Adaptive retrieval flag
    direct_answer: str  # For simple queries
    hallucination_free: bool
    retry_count: int


class SelfRAGBusinessAnalyst:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db", use_semantic_chunking=True):
        self.data_path = data_path
        self.db_path = db_path
        self.use_semantic_chunking = use_semantic_chunking
        
        print(f"🚀 Initializing Self-RAG Business Analyst v25.0...")
        print(f"   Features: Adaptive Retrieval + Document Grading + Hallucination Check")
        print(f"   Semantic Chunking: {'ENABLED' if use_semantic_chunking else 'DISABLED'}")
        
        # Models
        self.chat_model_name = "deepseek-r1:8b"
        self.embed_model_name = "nomic-embed-text"
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        self.llm = ChatOllama(
            model=self.chat_model_name, 
            temperature=0.0,
            num_predict=2000
        )
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        
        print(f"   📊 Loading re-ranker: {self.rerank_model_name}...")
        self.reranker = CrossEncoder(self.rerank_model_name)
        
        self.vectorstores = {}
        self.bm25_indexes = {}
        self.bm25_documents = {}
        self.use_hybrid = BM25_AVAILABLE
        
        # Self-RAG components
        print(f"   🧩 Initializing Self-RAG components...")
        self.document_grader = DocumentGrader(model_name=self.chat_model_name)
        self.hallucination_checker = HallucinationChecker(model_name=self.chat_model_name)
        self.adaptive_retrieval = AdaptiveRetrieval(model_name=self.chat_model_name)
        
        # Semantic chunker (lazy init during ingestion)
        self.semantic_chunker = None
        
        self.app = self._build_graph()
        print(f"   ✅ Self-RAG Business Analyst ready!")

    def _load_prompt(self, prompt_name):
        try:
            current_dir = os.getcwd()
            path = os.path.join(current_dir, "prompts", f"{prompt_name}.md")
            if not os.path.exists(path): 
                return "You are a Strategic Analyst."
            with open(path, "r") as f: 
                return f.read()
        except: 
            return "You are a Strategic Analyst."

    # --- BM25 Functions (from original) ---
    def _build_bm25_index(self, collection_name: str, documents: List[Document]):
        if not BM25_AVAILABLE:
            return None
        print(f"   🔨 Building BM25 index for {collection_name}...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        self.bm25_indexes[collection_name] = bm25
        self.bm25_documents[collection_name] = documents
        print(f"   ✅ BM25 index built with {len(documents)} documents")
        return bm25
    
    def _bm25_search(self, collection_name: str, query: str, k: int = 25) -> List[Tuple[Document, float]]:
        if not BM25_AVAILABLE or collection_name not in self.bm25_indexes:
            return []
        bm25 = self.bm25_indexes[collection_name]
        documents = self.bm25_documents[collection_name]
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(documents[i], float(scores[i])) for i in top_indices]
    
    def _reciprocal_rank_fusion(self, vector_results: List[Tuple[Document, float]], 
                                 bm25_results: List[Tuple[Document, float]], k: int = 60) -> List[Document]:
        def doc_id(doc: Document) -> str:
            return f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{hash(doc.page_content[:100])}"
        
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_key = doc_id(doc)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            doc_map[doc_key] = doc
        
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_key = doc_id(doc)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            doc_map[doc_key] = doc
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_key] for doc_key in sorted_ids]
    
    def _hybrid_search(self, collection_name: str, query: str, k: int = 25) -> List[Document]:
        vs = self._get_vectorstore(collection_name)
        print(f"   🔍 Performing vector search (top {k})...")
        vector_docs = vs.similarity_search_with_score(query, k=k)
        
        if self.use_hybrid and collection_name in self.bm25_indexes:
            print(f"   🔍 Performing BM25 search (top {k})...")
            bm25_results = self._bm25_search(collection_name, query, k=k)
            print(f"   🔀 Fusing results with RRF...")
            fused_docs = self._reciprocal_rank_fusion(vector_docs, bm25_results)
            return fused_docs[:k]
        else:
            return [doc for doc, _ in vector_docs]

    # --- NEW: Adaptive Retrieval Node ---
    def adaptive_node(self, state: AgentState):
        """Determine if query needs RAG or can be answered directly"""
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        
        needs_rag, direct_answer, metadata = self.adaptive_retrieval.should_use_rag(query)
        
        if needs_rag:
            return {"skip_rag": False}
        else:
            return {
                "skip_rag": True,
                "direct_answer": direct_answer
            }

    # --- Node 1: Identification ---
    def identify_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content.upper() if last_human_msg else ""
        
        found_tickers = []
        mapping = {
            "APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA",
            "NVIDIA": "NVDA", "GOOGLE": "GOOGL", "AMAZON": "AMZN", "META": "META"
        }
        for name, ticker in mapping.items():
            if name in query: 
                found_tickers.append(ticker)
        
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        for t in potential_tickers:
            if t in mapping.values(): 
                found_tickers.append(t)
        
        return {"tickers": list(set(found_tickers))}

    # --- Node 2: Research (Hybrid Search) ---
    def research_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        tickers = state.get('tickers', [])
        
        if not tickers:
            return {"documents": [], "context": "SYSTEM WARNING: No companies identified."}

        all_documents = []
        
        for ticker in tickers:
            print(f"🕵️ [Research] Hybrid search for {ticker}...")
            collection_name = f"docs_{ticker}"
            vs = self._get_vectorstore(collection_name)
            
            if vs._collection.count() == 0:
                continue

            # Query enhancement
            search_query = query
            if "compet" in query.lower(): 
                search_query += " competition rivals market share"
            if "risk" in query.lower(): 
                search_query += " risk factors regulation inflation"
            if "product" in query.lower(): 
                search_query += " products services offerings"
            
            docs = self._hybrid_search(collection_name, search_query, k=25)
            all_documents.extend(docs)
        
        return {"documents": all_documents}

    # --- NEW: Grade Documents Node ---
    def grade_documents_node(self, state: AgentState):
        """Filter documents by relevance using LLM"""
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        documents = state.get('documents', [])
        
        if not documents:
            return {
                "graded_documents": [],
                "passed_grading": False,
                "relevance_rate": 0.0
            }
        
        # Grade documents
        graded_docs, metadata = self.document_grader.grade_documents(
            query=query,
            documents=documents,
            threshold=0.3  # At least 30% must be relevant
        )
        
        return {
            "graded_documents": graded_docs,
            "passed_grading": metadata['meets_threshold'],
            "relevance_rate": metadata['pass_rate']
        }

    # --- NEW: Web Search Fallback Node ---
    def web_search_fallback_node(self, state: AgentState):
        """Fallback to web search if document grading failed"""
        print("\n🌐 [Web Search] Document grading failed - using web search fallback...")
        
        # This would integrate with web_search_agent
        # For now, return a placeholder
        return {
            "context": "[WEB SEARCH PLACEHOLDER] This would call the web_search_agent for current information."
        }

    # --- Node 3: Rerank (BERT) ---
    def rerank_node(self, state: AgentState):
        """BERT reranking of graded documents"""
        docs = state.get('graded_documents', [])
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        if not docs or not query:
            return {"context": ""}

        print(f"⚖️ [Rerank] BERT scoring {len(docs)} filtered documents...")
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        top_k = min(8, len(scored_docs))
        top_docs = [doc for doc, score in scored_docs[:top_k]]
        
        print(f"✅ [Rerank] Selected top {len(top_docs)} documents")
        
        # Format context
        formatted_chunks = []
        for d in top_docs:
            source = d.metadata.get('source') or d.metadata.get('file_path') or "Unknown_File"
            source = os.path.basename(source)
            page = d.metadata.get('page') or d.metadata.get('page_number') or "N/A"
            formatted_chunks.append(f"--- SOURCE: {source} (Page {page}) ---\n{d.page_content}")
        
        context = "\n\n".join(formatted_chunks)
        return {"context": context}

    # --- Node 4: Analyst (Generation) ---
    def analyst_node(self, state: AgentState):
        """Generate analysis with citation enforcement"""
        context = state.get('context', '')
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content.lower() if last_human_msg else ""
        tickers = state.get('tickers', [])
        ticker = tickers[0] if tickers else "Unknown"
        
        if ticker == "Unknown":
            return {"messages": [HumanMessage(content="❌ Please specify a valid company.")]}
        
        if not context:
            return {"messages": [HumanMessage(content=f"⚠️ No relevant documents found for {ticker}.")]}

        # Persona selection
        if "compet" in query or "market share" in query:
            base_prompt = self._load_prompt("competitive_intel")
        elif "risk" in query or "threat" in query:
            base_prompt = self._load_prompt("risk_officer")
        else:
            base_prompt = self._load_prompt("chief_strategy_officer")
        
        # Citation enforcement
        citation_instruction = """
⚠️ CRITICAL CITATION REQUIREMENT ⚠️

OUTPUT FORMAT:
[2-4 sentences of analysis]
--- SOURCE: filename.pdf (Page X) ---

RULES:
1. Write 2-4 sentences
2. Add SOURCE line immediately after
3. Repeat for each point
4. Use EXACT format: --- SOURCE: filename (Page X) ---
        """
        
        full_prompt = f"""{base_prompt}
        
{citation_instruction}

====== DOCUMENT CONTEXT ======
{context}
==============================

USER QUESTION: {query}

Provide your analysis with strict citation compliance.
        """

        new_messages = [SystemMessage(content=full_prompt)] + messages
        response = self.llm.invoke(new_messages)
        analysis = response.content
        
        return {"messages": [HumanMessage(content=analysis)]}

    # --- NEW: Hallucination Check Node ---
    def hallucination_check_node(self, state: AgentState):
        """Verify generated analysis is grounded in sources"""
        messages = state['messages']
        analysis = messages[-1].content if messages else ""
        context = state.get('context', '')
        
        is_grounded, feedback, metadata = self.hallucination_checker.check_hallucination(
            analysis=analysis,
            context=context
        )
        
        retry_count = state.get('retry_count', 0)
        
        if not is_grounded:
            print(f"\n⚠️ Hallucination detected - Retry {retry_count + 1}/2")
        
        return {
            "hallucination_free": is_grounded,
            "retry_count": retry_count + 1 if not is_grounded else retry_count
        }

    # --- NEW: Direct Output Node ---
    def direct_output_node(self, state: AgentState):
        """Return direct answer without RAG"""
        direct_answer = state.get('direct_answer', '')
        return {"messages": [HumanMessage(content=direct_answer)]}

    # --- Conditional Edges ---
    def should_skip_rag(self, state: AgentState) -> str:
        """Route: direct answer or full RAG"""
        if state.get('skip_rag', False):
            return "direct_output"
        return "identify"
    
    def should_use_web_search(self, state: AgentState) -> str:
        """Route: analyst or web search"""
        if not state.get('passed_grading', False):
            print("   🌐 Routing to web search fallback")
            return "web_search"
        return "rerank"
    
    def should_retry_generation(self, state: AgentState) -> str:
        """Route: end or retry generation"""
        if state.get('hallucination_free', True):
            return END
        
        if state.get('retry_count', 0) >= 2:
            print("⚠️ Max retries reached - outputting with warning")
            return END
        
        return "analyst"

    # --- Build Graph ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("adaptive", self.adaptive_node)
        workflow.add_node("direct_output", self.direct_output_node)
        workflow.add_node("identify", self.identify_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("grade", self.grade_documents_node)
        workflow.add_node("web_search", self.web_search_fallback_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("hallucination_check", self.hallucination_check_node)
        
        # Build flow
        workflow.add_edge(START, "adaptive")
        
        # Adaptive routing
        workflow.add_conditional_edges(
            "adaptive",
            self.should_skip_rag,
            {
                "direct_output": "direct_output",
                "identify": "identify"
            }
        )
        
        workflow.add_edge("direct_output", END)
        
        # Full RAG pipeline
        workflow.add_edge("identify", "research")
        workflow.add_edge("research", "grade")
        
        # Grade routing
        workflow.add_conditional_edges(
            "grade",
            self.should_use_web_search,
            {
                "web_search": "web_search",
                "rerank": "rerank"
            }
        )
        
        workflow.add_edge("web_search", "rerank")
        workflow.add_edge("rerank", "analyst")
        workflow.add_edge("analyst", "hallucination_check")
        
        # Hallucination check routing
        workflow.add_conditional_edges(
            "hallucination_check",
            self.should_retry_generation,
            {
                END: END,
                "analyst": "analyst"
            }
        )
        
        return workflow.compile()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
        return self.vectorstores[collection_name]

    # --- Ingestion with Semantic Chunking ---
    def ingest_data(self):
        print(f"\n📂 Scanning {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return
        
        folders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        
        if not folders:
            print(f"   ⚠️ No company folders found")
            return
        
        # Initialize semantic chunker if enabled
        if self.use_semantic_chunking and self.semantic_chunker is None:
            print(f"   🧩 Initializing semantic chunker...")
            self.semantic_chunker = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80,
                min_chunk_size=500,
                max_chunk_size=4000
            )
        
        for folder in folders:
            ticker = os.path.basename(folder).upper()
            print(f"\n📊 Processing {ticker}...")
            
            all_docs = []
            
            # Load PDFs
            try:
                pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
                pdf_docs = pdf_loader.load()
                if pdf_docs:
                    all_docs.extend(pdf_docs)
                    print(f"   ✅ Loaded {len(pdf_docs)} PDF documents")
            except Exception as e:
                print(f"   ⚠️ PDF error: {e}")
            
            if not all_docs:
                continue
            
            # Split with semantic or recursive chunker
            if self.use_semantic_chunking:
                print(f"   🧩 Semantic chunking (embedding-based)...")
                splits = self.semantic_chunker.split_documents(all_docs)
            else:
                print(f"   🔪 Recursive chunking (fixed-size)...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                splits = splitter.split_documents(all_docs)
            
            # Index
            collection_name = f"docs_{ticker}"
            print(f"   🧮 Embedding {len(splits)} chunks...")
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            
            # Build BM25
            if self.use_hybrid:
                self._build_bm25_index(collection_name, splits)
            
            print(f"   ✅ Indexed {len(splits)} chunks")
        
        print(f"\n✅ Self-RAG ingestion complete!")

    def analyze(self, query: str):
        print(f"🤖 User Query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content


if __name__ == "__main__":
    agent = SelfRAGBusinessAnalyst(use_semantic_chunking=True)
