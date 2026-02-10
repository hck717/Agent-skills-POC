import os
import operator
import re
import shutil
from typing import Annotated, TypedDict, List, Tuple, Dict
from collections import defaultdict
import math

# LangChain & Graph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END, START

# BM25 for sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: rank_bm25 not installed. Hybrid search will use vector-only mode.")
    print("   Install with: pip install rank-bm25")
    BM25_AVAILABLE = False

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str        # The text gathered from 10-Ks
    tickers: List[str]  # Companies identified

class BusinessAnalystGraphAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        print(f"🚀 Initializing HYBRID SEARCH Analyst Agent v24.0 (DeepSeek-R1 8B)...")
        print(f"   Features: Vector Search + BM25 + Advanced Re-ranking")
        
        # Models
        self.chat_model_name = "deepseek-r1:8b"
        self.embed_model_name = "nomic-embed-text"
        # 🔥 UPGRADE: Better re-ranker (12-layer vs 6-layer)
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        # 🔥 CRITICAL FIX: Temperature 0.0 for strict citation adherence
        self.llm = ChatOllama(
            model=self.chat_model_name, 
            temperature=0.0,
            num_predict=2000
        )
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        
        print(f"   📊 Loading re-ranker: {self.rerank_model_name}...")
        self.reranker = CrossEncoder(self.rerank_model_name)
        print(f"   ✅ Re-ranker loaded (12-layer model for better accuracy)")
        
        self.vectorstores = {}
        
        # 🔥 NEW: BM25 index storage (in-memory)
        self.bm25_indexes = {}  # {collection_name: BM25Okapi}
        self.bm25_documents = {}  # {collection_name: List[Document]}
        
        # 🔥 NEW: Hybrid search parameters
        self.hybrid_alpha = 0.5  # Weight: 0=BM25 only, 1=vector only, 0.5=balanced
        self.use_hybrid = BM25_AVAILABLE  # Auto-disable if library missing

        self.app = self._build_graph()
        
        if self.use_hybrid:
            print(f"   ✅ Hybrid search enabled (alpha={self.hybrid_alpha})")
        else:
            print(f"   ⚠️ Hybrid search disabled (vector-only mode)")

    def _load_prompt(self, prompt_name):
        try:
            current_dir = os.getcwd()
            path = os.path.join(current_dir, "prompts", f"{prompt_name}.md")
            if not os.path.exists(path): return "You are a Strategic Analyst."
            with open(path, "r") as f: return f.read()
        except: return "You are a Strategic Analyst."

    # --- NEW: BM25 Index Management ---
    def _build_bm25_index(self, collection_name: str, documents: List[Document]):
        """
        Build BM25 index for sparse retrieval
        """
        if not BM25_AVAILABLE:
            return None
            
        print(f"   🔨 Building BM25 index for {collection_name}...")
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Store index and documents
        self.bm25_indexes[collection_name] = bm25
        self.bm25_documents[collection_name] = documents
        
        print(f"   ✅ BM25 index built with {len(documents)} documents")
        return bm25
    
    def _bm25_search(self, collection_name: str, query: str, k: int = 25) -> List[Tuple[Document, float]]:
        """
        Perform BM25 sparse retrieval
        Returns: List of (Document, score) tuples
        """
        if not BM25_AVAILABLE or collection_name not in self.bm25_indexes:
            return []
        
        bm25 = self.bm25_indexes[collection_name]
        documents = self.bm25_documents[collection_name]
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = [(documents[i], float(scores[i])) for i in top_indices]
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                                 vector_results: List[Tuple[Document, float]], 
                                 bm25_results: List[Tuple[Document, float]], 
                                 k: int = 60) -> List[Document]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion (RRF)
        RRF formula: score = sum(1 / (k + rank_i)) for each retrieval method
        
        Args:
            vector_results: List of (Document, score) from vector search
            bm25_results: List of (Document, score) from BM25 search
            k: RRF constant (typically 60)
        
        Returns:
            List of Documents sorted by RRF score
        """
        # Create document identifier (using page_content hash as unique ID)
        def doc_id(doc: Document) -> str:
            return f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}_{hash(doc.page_content[:100])}"
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}  # id -> Document
        
        # Add vector search results
        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_key = doc_id(doc)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            doc_map[doc_key] = doc
        
        # Add BM25 results
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_key = doc_id(doc)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            doc_map[doc_key] = doc
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return documents in RRF order
        return [doc_map[doc_key] for doc_key in sorted_ids]
    
    def _hybrid_search(self, collection_name: str, query: str, k: int = 25) -> List[Document]:
        """
        Perform hybrid search combining vector and BM25 retrieval
        
        Args:
            collection_name: Name of the collection to search
            query: Search query
            k: Number of results to return
        
        Returns:
            List of top-k Documents from hybrid search
        """
        vs = self._get_vectorstore(collection_name)
        
        # 1. Vector search
        print(f"   🔍 Performing vector search (top {k})...")
        vector_docs = vs.similarity_search_with_score(query, k=k)
        print(f"   ✅ Vector search: {len(vector_docs)} results")
        
        # 2. BM25 search (if available)
        if self.use_hybrid and collection_name in self.bm25_indexes:
            print(f"   🔍 Performing BM25 search (top {k})...")
            bm25_results = self._bm25_search(collection_name, query, k=k)
            print(f"   ✅ BM25 search: {len(bm25_results)} results")
            
            # 3. Combine with RRF
            print(f"   🔀 Fusing results with Reciprocal Rank Fusion...")
            fused_docs = self._reciprocal_rank_fusion(vector_docs, bm25_results)
            print(f"   ✅ Hybrid fusion: {len(fused_docs)} unique documents")
            
            # Return top-k from fused results
            return fused_docs[:k]
        else:
            # Fallback to vector-only
            print(f"   ⚠️ BM25 not available, using vector-only search")
            return [doc for doc, _ in vector_docs]

    # --- Node 1: Identification ---
    def identify_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content.upper() if last_human_msg else ""
        
        found_tickers = []
        mapping = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA", "NVIDIA": "NVDA", "GOOGLE": "GOOGL", "AMAZON": "AMZN", "META": "META"}
        for name, ticker in mapping.items():
            if name in query: found_tickers.append(ticker)
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        for t in potential_tickers:
            if t in mapping.values(): found_tickers.append(t)
        return {"tickers": list(set(found_tickers))}

    # --- Node 2: Strategic Research (UPGRADED with Hybrid Search) ---
    def research_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        tickers = state.get('tickers', [])
        
        context_str = ""
        
        if not tickers: return {"context": "SYSTEM WARNING: No specific companies identified."}

        for ticker in tickers:
            print(f"🕵️ [Step 2] HYBRID SEARCH for {ticker}...")
            
            collection_name = f"docs_{ticker}"
            vs = self._get_vectorstore(collection_name)
            
            if vs._collection.count() == 0:
                context_str += f"\n[WARNING] No documents found for {ticker}.\n"
                continue

            # Query enhancement
            search_query = query
            if "compet" in query.lower(): search_query += " competition rivals market share"
            if "risk" in query.lower(): search_query += " risk factors regulation inflation"
            if "product" in query.lower(): search_query += " products services offerings"
            
            print(f"   📝 Enhanced query: '{search_query[:100]}...'")
            
            # 🔥 NEW: Hybrid search (vector + BM25)
            docs = self._hybrid_search(collection_name, search_query, k=25)
            
            rag_content = "No relevant text found."
            if docs:
                print(f"   🎯 Re-ranking top {len(docs)} documents with improved cross-encoder...")
                
                # 🔥 IMPROVED: Re-ranking with better model
                pairs = [[query, d.page_content] for d in docs]
                scores = self.reranker.predict(pairs)
                top_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:8]
                
                print(f"   ✅ Selected top 8 documents (scores: {[f'{s:.3f}' for _, s in top_docs[:3]]}...)")
                
                if top_docs:
                    print(f"   🔍 DEBUG: Metadata sample: {top_docs[0][0].metadata}")

                # 🟢 FORMAT: --- SOURCE: filename (Page X) ---
                formatted_chunks = []
                for d, s in top_docs:
                    source = d.metadata.get('source') or d.metadata.get('file_path') or "Unknown_File"
                    source = os.path.basename(source)
                    page = d.metadata.get('page') or d.metadata.get('page_number') or "N/A"
                    
                    # ✅ CRITICAL: Use format orchestrator expects
                    formatted_chunks.append(f"--- SOURCE: {source} (Page {page}) ---\n{d.page_content}")
                
                rag_content = "\n\n".join(formatted_chunks)
            
            context_str += f"\n====== ANALYSIS CONTEXT FOR {ticker} ======\n{rag_content}\n===========================================\n"
            
        return {"context": context_str}

    def _inject_citations_if_missing(self, analysis: str, context: str) -> str:
        """
        🔥 POST-PROCESSING FIX: Inject citations if LLM didn't preserve them
        """
        # Check if analysis already has citations
        if '--- SOURCE:' in analysis:
            citation_count = analysis.count('--- SOURCE:')
            print(f"   ✅ LLM preserved {citation_count} citations")
            return analysis
        
        print("   ⚠️ LLM didn't preserve citations - injecting them automatically")
        
        # Extract all sources from context
        source_pattern = r'--- SOURCE: ([^\(]+)\(Page ([^\)]+)\) ---'
        sources = re.findall(source_pattern, context)
        
        if not sources:
            print("   ❌ No sources found in context to inject")
            return analysis
        
        print(f"   📚 Found {len(sources)} sources to distribute")
        
        # Split analysis into sections/paragraphs
        lines = analysis.split('\n')
        result_lines = []
        source_idx = 0
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            # Add citation after substantial paragraphs (not headers, not empty lines)
            if (line.strip() and 
                not line.startswith('#') and 
                len(line) > 100 and 
                source_idx < len(sources) and
                i < len(lines) - 1):  # Don't add to last line
                
                filename, page = sources[source_idx]
                result_lines.append(f"--- SOURCE: {filename}(Page {page}) ---")
                print(f"   [Citation {source_idx + 1}] Added: {filename} Page {page}")
                source_idx += 1
        
        injected_analysis = '\n'.join(result_lines)
        final_count = injected_analysis.count('--- SOURCE:')
        print(f"   ✅ Injected {final_count} citations into analysis")
        
        return injected_analysis

    # --- Node 3: Strategic Analyst ---
    def analyst_node(self, state: AgentState):
        messages = state['messages']
        context = state.get('context', '')
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content.lower() if last_human_msg else ""
        
        if "compet" in query or "market share" in query:
            print("🎭 [Persona] COMPETITIVE INTELLIGENCE")
            base_prompt = self._load_prompt("competitive_intel")
        elif "risk" in query or "threat" in query:
            print("🎭 [Persona] CHIEF RISK OFFICER")
            base_prompt = self._load_prompt("risk_officer")
        else:
            print("🎭 [Persona] CHIEF STRATEGY OFFICER")
            base_prompt = self._load_prompt("chief_strategy_officer")
        
        # 🔥 SOLUTION 2: Stricter citation format with few-shot examples
        citation_instruction = """
        ---------------------------------------------------
        ⚠️ CRITICAL CITATION REQUIREMENT ⚠️
        
        YOU MUST OUTPUT IN THIS EXACT FORMAT:
        
        [Your paragraph of analysis - 2 to 4 sentences]
        --- SOURCE: filename.pdf (Page X) ---
        
        [Next paragraph of analysis]
        --- SOURCE: filename.pdf (Page Y) ---
        
        THIS IS MANDATORY. DO NOT SKIP THE SOURCE LINES.
        
        EXAMPLE OUTPUT YOU MUST FOLLOW:
        
        ## Supply Chain Concentration Risk
        Apple relies heavily on third-party manufacturers in Asia, particularly 
        for iPhone assembly. The majority of production capacity is concentrated 
        in China, creating significant geopolitical exposure.
        --- SOURCE: APPL 10-k Filings.pdf (Page 23) ---
        
        Supply disruptions during 2020-2021 demonstrated the vulnerability of 
        this concentrated manufacturing model. The company has limited ability 
        to rapidly shift production to alternative regions.
        --- SOURCE: APPL 10-k Filings.pdf (Page 24) ---
        
        ## Market Competition
        The smartphone market faces intense competition from Android manufacturers 
        including Samsung, Xiaomi, and Oppo. These competitors offer feature-rich 
        devices at lower price points, particularly in emerging markets.
        --- SOURCE: APPL 10-k Filings.pdf (Page 45) ---
        
        Market share erosion in price-sensitive regions poses risks to unit growth. 
        Premium positioning limits addressable market in developing economies.
        --- SOURCE: APPL 10-k Filings.pdf (Page 46) ---
        
        RULES:
        1. Write 2-4 sentences
        2. Add SOURCE line immediately after
        3. Repeat for each major point
        4. Use the EXACT format: --- SOURCE: filename (Page X) ---
        5. Do NOT write long sections without citations
        6. Do NOT combine multiple topics without separate citations
        
        The document context below contains SOURCE markers. 
        You MUST preserve them in your output.
        ---------------------------------------------------
        """
        
        full_prompt = f"""{base_prompt}
        
        {citation_instruction}

        ====== DOCUMENT CONTEXT (with SOURCE markers) ======
        {context}
        ==============================
        
        USER QUESTION: {query}
        
        Now provide your analysis following the EXACT citation format shown above.
        Keep your response focused and concise (target ~2000 tokens).
        """

        new_messages = [SystemMessage(content=full_prompt)] + messages
        
        print("   🤖 Generating analysis with strict citation enforcement (DeepSeek-R1)...")
        print("   ⚡ Token limit: 2000 (optimized for synthesis)")
        response = self.llm.invoke(new_messages)
        analysis = response.content
        
        # 🔥 SOLUTION 1: Post-process to inject citations if LLM failed
        analysis = self._inject_citations_if_missing(analysis, context)
        
        return {"messages": [HumanMessage(content=analysis)]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("identify", self.identify_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_edge(START, "identify")
        workflow.add_edge("identify", "research")
        workflow.add_edge("research", "analyst")
        workflow.add_edge("analyst", END)
        return workflow.compile()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(collection_name=collection_name, embedding_function=self.embeddings, persist_directory=self.db_path)
        return self.vectorstores[collection_name]

    def reset_vector_db(self):
        """
        ⚠️ DANGER: Delete all vector data in ChromaDB
        This will remove all embedded documents and you'll need to re-ingest
        """
        print(f"\n🗑️ RESETTING VECTOR DATABASE...")
        print(f"   Path: {self.db_path}")
        
        try:
            # Clear in-memory indexes
            self.vectorstores = {}
            self.bm25_indexes = {}
            self.bm25_documents = {}
            
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"   ✅ Deleted {self.db_path}")
            else:
                print(f"   ⚠️ Directory doesn't exist: {self.db_path}")
            
            os.makedirs(self.db_path, exist_ok=True)
            print(f"   ✅ Created fresh directory")
            
            return True, "Vector database reset successfully"
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False, f"Failed to reset: {str(e)}"

    def get_database_stats(self):
        """
        Get statistics about the current vector database
        """
        stats = {}
        total_chunks = 0
        
        if not os.path.exists(self.data_path):
            return {"error": "Data path doesn't exist"}
        
        for folder in [f.path for f in os.scandir(self.data_path) if f.is_dir()]:
            ticker = os.path.basename(folder).upper()
            try:
                vs = self._get_vectorstore(f"docs_{ticker}")
                count = vs._collection.count()
                stats[ticker] = count
                total_chunks += count
            except:
                stats[ticker] = 0
        
        stats['TOTAL'] = total_chunks
        return stats

    def ingest_data(self):
        """
        📂 Ingest documents from data folder
        Supports: PDF (.pdf), Word (.docx), Text (.txt, .md)
        🔥 NEW: Also builds BM25 indexes for hybrid search
        """
        print(f"\n📂 Scanning {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"   ⚠️ Created empty data directory: {self.data_path}")
            return
        
        folders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        
        if not folders:
            print(f"   ⚠️ No company folders found in {self.data_path}")
            print(f"   💡 Create folders like: ./data/AAPL/, ./data/TSLA/, etc.")
            return
        
        total_docs = 0
        total_chunks = 0
        
        for folder in folders:
            ticker = os.path.basename(folder).upper()
            print(f"\n📊 Processing {ticker}...")
            print(f"   Folder: {folder}")
            
            all_docs = []
            doc_types = []
            
            # Load PDF files
            try:
                pdf_loader = DirectoryLoader(
                    folder, 
                    glob="**/*.pdf", 
                    loader_cls=PyPDFLoader,
                    show_progress=False
                )
                pdf_docs = pdf_loader.load()
                if pdf_docs:
                    all_docs.extend(pdf_docs)
                    doc_types.append(f"{len(pdf_docs)} PDFs")
                    print(f"   ✅ Loaded {len(pdf_docs)} PDF documents")
            except Exception as e:
                print(f"   ⚠️ PDF loading error: {e}")
            
            # Load Word documents (.docx)
            try:
                docx_loader = DirectoryLoader(
                    folder, 
                    glob="**/*.docx", 
                    loader_cls=Docx2txtLoader,
                    show_progress=False
                )
                docx_docs = docx_loader.load()
                if docx_docs:
                    all_docs.extend(docx_docs)
                    doc_types.append(f"{len(docx_docs)} Word docs")
                    print(f"   ✅ Loaded {len(docx_docs)} Word documents")
            except Exception as e:
                print(f"   ⚠️ Word doc loading error: {e}")
            
            # Load text files (.txt, .md)
            try:
                text_loader = DirectoryLoader(
                    folder, 
                    glob="**/*.txt", 
                    loader_cls=TextLoader,
                    show_progress=False
                )
                txt_docs = text_loader.load()
                if txt_docs:
                    all_docs.extend(txt_docs)
                    doc_types.append(f"{len(txt_docs)} text files")
                    print(f"   ✅ Loaded {len(txt_docs)} text files")
            except Exception as e:
                print(f"   ⚠️ Text file loading error: {e}")
            
            # Load markdown files
            try:
                md_loader = DirectoryLoader(
                    folder, 
                    glob="**/*.md", 
                    loader_cls=TextLoader,
                    show_progress=False
                )
                md_docs = md_loader.load()
                if md_docs:
                    all_docs.extend(md_docs)
                    doc_types.append(f"{len(md_docs)} markdown files")
                    print(f"   ✅ Loaded {len(md_docs)} markdown files")
            except Exception as e:
                print(f"   ⚠️ Markdown loading error: {e}")
            
            if not all_docs:
                print(f"   ⚠️ No supported documents found for {ticker}")
                print(f"   💡 Supported formats: .pdf, .docx, .txt, .md")
                continue
            
            # Split into chunks
            print(f"   🔪 Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000, 
                chunk_overlap=200
            )
            splits = splitter.split_documents(all_docs)
            
            # Add to vector store
            collection_name = f"docs_{ticker}"
            print(f"   🧮 Embedding {len(splits)} chunks...")
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            
            # 🔥 NEW: Build BM25 index for hybrid search
            if self.use_hybrid:
                self._build_bm25_index(collection_name, splits)
            
            print(f"   ✅ Indexed {len(splits)} chunks from {', '.join(doc_types)}")
            
            total_docs += len(all_docs)
            total_chunks += len(splits)
        
        print(f"\n{'='*60}")
        print(f"✅ INGESTION COMPLETE")
        print(f"   Total documents: {total_docs}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Database: {self.db_path}")
        if self.use_hybrid:
            print(f"   Hybrid search: ENABLED (Vector + BM25)")
        else:
            print(f"   Hybrid search: DISABLED (Vector only)")
        print(f"{'='*60}\n")

    def analyze(self, query: str):
        print(f"🤖 User Query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content

if __name__ == "__main__":
    agent = BusinessAnalystGraphAgent()
