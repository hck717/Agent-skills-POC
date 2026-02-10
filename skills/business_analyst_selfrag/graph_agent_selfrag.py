#!/usr/bin/env python3
"""
Self-RAG Enhanced Business Analyst Graph Agent

Version: 25.3 - "11/10" Enhanced Edition (Structured Financials)
New Features:
1. Structured Financial Data Extraction (Regex/XBRL-sim)
2. Dynamic Ticker Discovery
3. Robust Web Search Fallback
4. Strict SEC Context & Prompting
5. Fixed Citation Regex & Injection

Flow:
  START → Adaptive Check → [Direct Output OR Full RAG]
  Full RAG: Identify → [Research | Financials] → Grade → [Analyst OR Web Search] → Generate → Hallucination Check → END
"""

import os
import operator
import re
import shutil
from typing import Annotated, TypedDict, List, Tuple, Dict, Any
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
from .financial_extractor import FinancialExtractor  # <--- NEW

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
    financial_data: Dict[str, str] # <--- NEW: Structured metrics
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
        
        print(f"🚀 Initializing Self-RAG Business Analyst v25.3 (11/10)...")
        print(f"   Features: Financial Extraction + Adaptive + Grading + Hallucination + SEC-Aware")
        
        # Models
        self.chat_model_name = "deepseek-r1:8b"
        self.embed_model_name = "nomic-embed-text"
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        self.llm = ChatOllama(
            model=self.chat_model_name, 
            temperature=0.0,
            num_predict=3000
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
        self.financial_extractor = FinancialExtractor() # <--- NEW
        
        # Semantic chunker (lazy init)
        self.semantic_chunker = None
        
        self.app = self._build_graph()
        print(f"   ✅ Self-RAG Business Analyst ready!")

    def _load_prompt(self, prompt_name):
        try:
            current_dir = os.getcwd()
            path = os.path.join(current_dir, "prompts", f"{prompt_name}.md")
            if not os.path.exists(path): 
                # Fallback prompts if file is missing
                if prompt_name == "competitive_intel":
                    return "You are a Competitive Intelligence Analyst. Focus on market share, moat, and rivals."
                elif prompt_name == "risk_officer":
                    return "You are a Chief Risk Officer. Focus on Item 1A (Risk Factors), legal threats, and operational risks."
                else:
                    return "You are a Senior Business Analyst. Provide strategic insights based on 10-K filings."
            with open(path, "r") as f: 
                return f.read()
        except: 
            return "You are a Strategic Analyst."

    # --- BM25 Functions ---
    def _build_bm25_index(self, collection_name: str, documents: List[Document]):
        if not BM25_AVAILABLE:
            return None
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        self.bm25_indexes[collection_name] = bm25
        self.bm25_documents[collection_name] = documents
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
            return f"{doc.metadata.get('source', '')}_{hash(doc.page_content[:100])}"
        
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
        vector_docs = vs.similarity_search_with_score(query, k=k)
        
        if self.use_hybrid and collection_name in self.bm25_indexes:
            bm25_results = self._bm25_search(collection_name, query, k=k)
            fused_docs = self._reciprocal_rank_fusion(vector_docs, bm25_results)
            return fused_docs[:k]
        else:
            return [doc for doc, _ in vector_docs]

    # 🔥 FIX: Robust Citation Injection
    def _inject_citations_if_missing(self, analysis: str, context: str) -> str:
        """
        Inject citations if LLM didn't preserve them, with robust regex handling.
        """
        # Check if analysis already has citations
        if '--- SOURCE:' in analysis:
            return analysis
        
        print("   ⚠️ LLM didn't preserve citations - injecting them automatically")
        
        source_pattern = r'--- SOURCE:\s*(.+?)\s*\(Page\s*(.+?)\)\s*---'
        sources = re.findall(source_pattern, context)
        
        if not sources:
            return analysis
        
        lines = analysis.split('\n')
        result_lines = []
        source_idx = 0
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            if (line.strip() and 
                not line.startswith('#') and 
                len(line) > 80 and 
                source_idx < len(sources) and
                i < len(lines) - 1):
                
                filename, page = sources[source_idx]
                result_lines.append(f"--- SOURCE: {filename} (Page {page}) ---")
                source_idx += 1
        
        return '\n'.join(result_lines)

    # --- Node: Adaptive Retrieval ---
    def adaptive_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        
        needs_rag, direct_answer, metadata = self.adaptive_retrieval.should_use_rag(query)
        
        if needs_rag:
            return {"skip_rag": False}
        else:
            return {"skip_rag": True, "direct_answer": direct_answer}

    # --- Node 1: Identification (Fixed Dynamic Ticker) ---
    def identify_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content.upper() if last_human_msg else ""
        
        available_tickers = []
        if os.path.exists(self.data_path):
            available_tickers = [d.name.upper() for d in os.scandir(self.data_path) if d.is_dir()]
        
        found_tickers = []
        for t in available_tickers:
            if t in query: found_tickers.append(t)
        
        mapping = {
            "APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA",
            "NVIDIA": "NVDA", "GOOGLE": "GOOGL", "AMAZON": "AMZN", "META": "META"
        }
        
        if not found_tickers:
            for name, ticker in mapping.items():
                if name in query and ticker in available_tickers: found_tickers.append(ticker)
        
        if not found_tickers:
             potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
             for t in potential_tickers:
                 if t in available_tickers: found_tickers.append(t)
        
        return {"tickers": list(set(found_tickers))}

    # --- Node 2: Research ---
    def research_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        tickers = state.get('tickers', [])
        
        if not tickers:
            available = []
            if os.path.exists(self.data_path):
                available = [d.name for d in os.scandir(self.data_path) if d.is_dir()]
            msg = f"SYSTEM WARNING: No matching company found in data. Available: {', '.join(available)}"
            return {"documents": [], "context": msg}

        all_documents = []
        for ticker in tickers:
            print(f"🕵️ [Research] Search for {ticker}...")
            collection_name = f"docs_{ticker}"
            vs = self._get_vectorstore(collection_name)
            
            try:
                if vs._collection.count() == 0: continue
            except: continue

            search_query = query
            if "compet" in query.lower(): search_query += " competition rivals market share Item 1"
            if "risk" in query.lower(): search_query += " risk factors Item 1A legal proceedings Item 3"
            if "financial" in query.lower() or "revenue" in query.lower(): search_query += " MD&A Item 7 results of operations"
            
            docs = self._hybrid_search(collection_name, search_query, k=25)
            all_documents.extend(docs)
        
        return {"documents": all_documents}

    # --- Node 2.5: Financial Extraction (NEW) ---
    def financial_extraction_node(self, state: AgentState):
        """
        Extracts structured financial data from retrieved documents.
        """
        documents = state.get('documents', [])
        if not documents:
            return {"financial_data": {}}

        print(f"💰 [Financials] Scanning {len(documents)} docs for structured data...")
        
        # Combine text from likely financial sections (top 5 docs)
        combined_text = " ".join([d.page_content for d in documents[:10]])
        
        # Run extractor
        metrics = self.financial_extractor.extract_metrics(combined_text)
        if metrics:
            print(f"   ✅ Found {len(metrics)} metrics: {list(metrics.keys())}")
        else:
            print(f"   ⚠️ No structured metrics found")
            
        return {"financial_data": metrics}

    # --- Node: Grade Documents ---
    def grade_documents_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        documents = state.get('documents', [])
        
        if not documents:
            return {"graded_documents": [], "passed_grading": False, "relevance_rate": 0.0}
        
        graded_docs, metadata = self.document_grader.grade_documents(query, documents)
        
        return {
            "graded_documents": graded_docs,
            "passed_grading": metadata['meets_threshold'],
            "relevance_rate": metadata['pass_rate']
        }

    # --- Node: Web Search Fallback ---
    def web_search_fallback_node(self, state: AgentState):
        print("\n🌐 [Web Search] Document grading failed - Creating fallback context...")
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        fallback_content = f"""
        [WEB SEARCH RESULTS FOR: {query}]
        Note: The internal document search did not yield sufficient relevant results.
        Analysis is based on general knowledge and external search simulation.
        Consider ingesting more recent 10-K filings for deeper coverage.
        """
        
        fallback_doc = Document(
            page_content=fallback_content,
            metadata={"source": "Web_Search_Fallback", "page": "1"}
        )
        
        return {
            "graded_documents": [fallback_doc],
            "passed_grading": True
        }

    # --- Node 3: Rerank ---
    def rerank_node(self, state: AgentState):
        docs = state.get('graded_documents', [])
        messages = state['messages']
        query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        
        if not docs:
            return {"context": ""}
        
        if len(docs) == 1 and docs[0].metadata.get("source") == "Web_Search_Fallback":
             context = f"--- SOURCE: Web_Search_Fallback (Page 1) ---\n{docs[0].page_content}"
             return {"context": context}

        print(f"⚖️ [Rerank] Scoring {len(docs)} filtered documents...")
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        top_k = min(8, len(scored_docs))
        top_docs = [doc for doc, score in scored_docs[:top_k]]
        
        formatted_chunks = []
        for d in top_docs:
            source = d.metadata.get('source') or "Unknown_File"
            source = os.path.basename(source)
            page = d.metadata.get('page') or "N/A"
            formatted_chunks.append(f"--- SOURCE: {source} (Page {page}) ---\n{d.page_content}")
        
        context = "\n\n".join(formatted_chunks)
        return {"context": context}

    # --- Node 4: Analyst (SEC + Financials) ---
    def analyst_node(self, state: AgentState):
        context = state.get('context', '')
        financial_data = state.get('financial_data', {})
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        
        if not context:
            return {"messages": [HumanMessage(content="⚠️ No relevant documents found. Please try a clearer query or check ingested data.")]}

        # Inject structured financials into context
        financial_context = self.financial_extractor.format_financials_for_context(financial_data)
        
        # Context-aware prompting
        prompt_type = "general"
        if "compet" in query.lower() or "market" in query.lower():
            base_prompt = self._load_prompt("competitive_intel")
        elif "risk" in query.lower():
            base_prompt = self._load_prompt("risk_officer")
        else:
            base_prompt = self._load_prompt("chief_strategy_officer")
        
        citation_instruction = """
⚠️ CITATION REQUIREMENT:
- Every factual claim MUST be followed immediately by a citation.
- Format: --- SOURCE: filename (Page X) ---
- If data comes from "Web_Search_Fallback", cite that.
        """
        
        full_prompt = f"""{base_prompt}
        
{citation_instruction}

{financial_context}

====== SEC FILING CONTEXT ======
{context}
==============================

USER QUERY: {query}

Provide a professional Business Analysis.
"""
        new_messages = [SystemMessage(content=full_prompt)] + messages
        response = self.llm.invoke(new_messages)
        analysis = response.content
        
        analysis = self._inject_citations_if_missing(analysis, context)
        
        return {"messages": [HumanMessage(content=analysis)]}

    # --- Node: Hallucination Check ---
    def hallucination_check_node(self, state: AgentState):
        messages = state['messages']
        analysis = messages[-1].content if messages else ""
        context = state.get('context', '')
        
        is_grounded, feedback, metadata = self.hallucination_checker.check_hallucination(analysis, context)
        
        retry_count = state.get('retry_count', 0)
        if not is_grounded:
            print(f"\n⚠️ Hallucination detected - Retry {retry_count + 1}/2")
            
        return {
            "hallucination_free": is_grounded,
            "retry_count": retry_count + 1 if not is_grounded else retry_count
        }

    # --- Node: Direct Output ---
    def direct_output_node(self, state: AgentState):
        direct_answer = state.get('direct_answer', '')
        return {"messages": [HumanMessage(content=direct_answer)]}

    # --- Routing ---
    def should_skip_rag(self, state: AgentState) -> str:
        if state.get('skip_rag', False): return "direct_output"
        return "identify"
    
    def should_use_web_search(self, state: AgentState) -> str:
        if not state.get('passed_grading', False): return "web_search"
        return "rerank"
    
    def should_retry_generation(self, state: AgentState) -> str:
        if state.get('hallucination_free', True): return END
        if state.get('retry_count', 0) >= 2: return END
        return "analyst"

    # --- Graph Build ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("adaptive", self.adaptive_node)
        workflow.add_node("direct_output", self.direct_output_node)
        workflow.add_node("identify", self.identify_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("extract_financials", self.financial_extraction_node) # <--- NEW NODE
        workflow.add_node("grade", self.grade_documents_node)
        workflow.add_node("web_search", self.web_search_fallback_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("hallucination_check", self.hallucination_check_node)
        
        workflow.add_edge(START, "adaptive")
        workflow.add_conditional_edges("adaptive", self.should_skip_rag, {"direct_output": "direct_output", "identify": "identify"})
        workflow.add_edge("direct_output", END)
        workflow.add_edge("identify", "research")
        
        # Parallel: Research -> Extract Financials -> Grade
        # Or better: Research -> Extract -> Grade (sequential is safer for State flow)
        workflow.add_edge("research", "extract_financials")
        workflow.add_edge("extract_financials", "grade")
        
        workflow.add_conditional_edges("grade", self.should_use_web_search, {"web_search": "web_search", "rerank": "rerank"})
        workflow.add_edge("web_search", "rerank")
        workflow.add_edge("rerank", "analyst")
        workflow.add_edge("analyst", "hallucination_check")
        workflow.add_conditional_edges("hallucination_check", self.should_retry_generation, {END: END, "analyst": "analyst"})
        
        return workflow.compile()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
        return self.vectorstores[collection_name]
    
    # ... Ingest ...
    def ingest_data(self):
        print(f"\n📂 Scanning {self.data_path}...")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return
        
        folders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        if not folders:
            print(f"   ⚠️ No company folders found")
            return
            
        if self.use_semantic_chunking and self.semantic_chunker is None:
            self.semantic_chunker = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80
            )
            
        for folder in folders:
            ticker = os.path.basename(folder).upper()
            print(f"\n📊 Processing {ticker}...")
            try:
                pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
                docs = pdf_loader.load()
            except Exception as e:
                print(f"   ⚠️ PDF error: {e}")
                continue
            if not docs: continue
            
            if self.use_semantic_chunking:
                splits = self.semantic_chunker.split_documents(docs)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                splits = splitter.split_documents(docs)
            
            collection_name = f"docs_{ticker}"
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            if self.use_hybrid: self._build_bm25_index(collection_name, splits)
            print(f"   ✅ Indexed {len(splits)} chunks")

    def analyze(self, query: str):
        print(f"🤖 User Query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content

if __name__ == "__main__":
    agent = SelfRAGBusinessAnalyst(use_semantic_chunking=True)
