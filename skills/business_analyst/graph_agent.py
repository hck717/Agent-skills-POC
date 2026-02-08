import os
import operator
import re
from typing import Annotated, TypedDict, List

# LangChain & Graph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END, START

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str        # The text gathered from 10-Ks
    tickers: List[str]  # Companies identified

class BusinessAnalystGraphAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        print(f"🚀 Initializing STRATEGIC Analyst Agent v22.0 (Metadata Debug)...")
        
        # Models
        self.chat_model_name = "qwen2.5:7b"
        self.embed_model_name = "nomic-embed-text"
        self.rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        self.llm = ChatOllama(model=self.chat_model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        self.reranker = CrossEncoder(self.rerank_model_name)
        self.vectorstores = {}

        self.app = self._build_graph()

    def _load_prompt(self, prompt_name):
        try:
            current_dir = os.getcwd()
            path = os.path.join(current_dir, "prompts", f"{prompt_name}.md")
            if not os.path.exists(path): return "You are a Strategic Analyst."
            with open(path, "r") as f: return f.read()
        except: return "You are a Strategic Analyst."

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

    # --- Node 2: Strategic Research ---
    def research_node(self, state: AgentState):
        messages = state['messages']
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = last_human_msg.content if last_human_msg else ""
        tickers = state.get('tickers', [])
        
        context_str = ""
        
        if not tickers: return {"context": "SYSTEM WARNING: No specific companies identified."}

        for ticker in tickers:
            print(f"🕵️ [Step 2] Reading 10-K for {ticker}...")
            
            vs = self._get_vectorstore(f"docs_{ticker}")
            if vs._collection.count() == 0:
                context_str += f"\n[WARNING] No documents found for {ticker}.\n"
                continue

            search_query = query
            if "compet" in query.lower(): search_query += " competition rivals market share"
            if "risk" in query.lower(): search_query += " risk factors regulation inflation"
            
            retriever = vs.as_retriever(search_kwargs={"k": 25})
            docs = retriever.invoke(search_query)
            
            rag_content = "No relevant text found."
            if docs:
                pairs = [[query, d.page_content] for d in docs]
                scores = self.reranker.predict(pairs)
                top_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:8]
                
                # 🟢 DEBUG PRINT: Check metadata keys
                if top_docs:
                    print(f"   🔍 DEBUG: Metadata sample: {top_docs[0][0].metadata}")

                # 🟢 ROBUST METADATA HANDLING
                formatted_chunks = []
                for d, s in top_docs:
                    source = d.metadata.get('source') or d.metadata.get('file_path') or "Unknown_File"
                    source = os.path.basename(source)
                    page = d.metadata.get('page') or d.metadata.get('page_number') or "N/A"
                    formatted_chunks.append(f"--- SOURCE: {source} (Page {page}) ---\n{d.page_content}")
                
                rag_content = "\n\n".join(formatted_chunks)
            
            context_str += f"\n====== ANALYSIS CONTEXT FOR {ticker} ======\n{rag_content}\n===========================================\n"
            
        return {"context": context_str}

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
            
        # 🟢 STRICT TEMPLATE
        citation_instruction = """
        ---------------------------------------------------
        CRITICAL: ANSWER FORMATTING REQUIRED
        
        You must structure your answer exactly like this:
        
        ## [Risk/Point 1]
        [Explanation of the point]
        **Source:** [Filename.pdf, Page X]
        
        ## [Risk/Point 2]
        [Explanation of the point]
        **Source:** [Filename.pdf, Page X]
        
        Do not write general paragraphs without sources. 
        If a specific fact comes from 'Unknown_File', label it [Source: Unknown].
        ---------------------------------------------------
        """
        
        full_prompt = f"""{base_prompt}
        
        {citation_instruction}

        ====== DOCUMENT CONTEXT ======
        {context}
        ==============================
        
        USER QUESTION: {query}
        """

        new_messages = [SystemMessage(content=full_prompt)] + messages
        response = self.llm.invoke(new_messages)
        return {"messages": [response]}

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

    def ingest_data(self):
        print(f"📂 Scanning {self.data_path}...")
        if not os.path.exists(self.data_path): os.makedirs(self.data_path); return
        for folder in [f.path for f in os.scandir(self.data_path) if f.is_dir()]:
            ticker = os.path.basename(folder).upper()
            print(f"Processing {ticker}...")
            all_docs = []
            try: all_docs.extend(DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader).load())
            except: pass
            if not all_docs: continue
            splits = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200).split_documents(all_docs)
            self._get_vectorstore(f"docs_{ticker}").add_documents(splits)
            print(f"   ✅ Indexed {len(splits)} chunks.")

    def analyze(self, query: str):
        print(f"🤖 User Query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = self.app.invoke(inputs)
        return result["messages"][-1].content

if __name__ == "__main__":
    agent = BusinessAnalystGraphAgent()
