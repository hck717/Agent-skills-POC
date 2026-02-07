import os
import operator
from typing import Annotated, TypedDict, Union, List

# LangChain & LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Document Processing
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Tools Definition ---
@tool
def calculate_growth(current: float, previous: float) -> str:
    """
    Calculates Year-Over-Year (YoY) growth percentage.
    Args:
        current: The value for the current period (must be a real number).
        previous: The value for the previous period.
    """
    try:
        current = float(current)
        previous = float(previous)
        if previous == 0: return "Error: Previous value is zero."
        growth = ((current - previous) / previous) * 100
        return f"{growth:.2f}%"
    except:
        return "Error: Invalid inputs."

@tool
def calculate_margin(metric: float, revenue: float) -> str:
    """
    Calculates margin percentage.
    Args:
        metric: The numerator (e.g., Net Income, EBITDA).
        revenue: The total revenue.
    """
    try:
        metric = float(metric)
        revenue = float(revenue)
        if revenue == 0: return "Error: Revenue is zero."
        margin = (metric / revenue) * 100
        return f"{margin:.2f}%"
    except:
        return "Error: Invalid inputs."

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
    context: str

class BusinessAnalystGraphAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        # Models
        self.chat_model_name = "qwen2.5:7b"
        self.embed_model_name = "nomic-embed-text"
        
        print(f"🔧 Initializing LangGraph Agent v3.0 ({self.chat_model_name})...")
        
        # 1. LLM & Tools
        self.tools = [calculate_growth, calculate_margin]
        self.llm = ChatOllama(model=self.chat_model_name, temperature=0) # Zero temp
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 2. Embeddings & Vector Store
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        self.vectorstores = {}

        # 3. Build the Graph
        self.app = self._build_graph()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
        return self.vectorstores[collection_name]

    # --- Node Functions ---

    def retrieve_node(self, state: AgentState):
        """Node: Identifies Ticker and Retrieves Context"""
        last_msg = state["messages"][-1]
        query = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)
        
        ticker = self._identify_ticker(query)
        
        if ticker == "UNKNOWN":
            return {"context": "Error: Ticker not found", "ticker": "UNKNOWN"}
            
        print(f"🔍 [Graph] Retrieving for {ticker} (k=20)...")
        collection_name = f"docs_{ticker}"
        vs = self._get_vectorstore(collection_name)
        
        # Increased k=20 for better recall
        retriever = vs.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        if not docs:
            return {"context": "", "ticker": ticker}
        
        context_text = "\n".join([
            f"--- Doc: {d.metadata.get('filename')} (Page {d.metadata.get('page')}) ---\n{d.page_content}\n" 
            for d in docs
        ])
        
        return {"context": context_text, "ticker": ticker}

    def analyst_node(self, state: AgentState):
        """Node: Main Analyst Logic (Reasoning)"""
        context = state.get("context", "")
        ticker = state.get("ticker", "Unknown")
        
        if ticker == "UNKNOWN":
            return {"messages": [AIMessage(content="❌ Please specify a valid company name (e.g., Apple, Microsoft).")]}
            
        if not context:
            return {"messages": [AIMessage(content=f"⚠️ No documents found for {ticker}. Please ensure data is ingested.")]}

        # --- v3.0 OPTIMIZED PROMPT ---
        system_prompt = f"""You are a Senior Equity Research Analyst covering {ticker}.
        
        CONTEXT FROM 10-K FILINGS:
        {context}
        
        YOUR MISSION:
        1. Answer the user's question using ONLY the Context above.
        
        DECISION PROTOCOL:
        - **QUALITATIVE QUESTIONS** (e.g. "What are the risks?", "Who are competitors?"): 
          -> Answer directly using text from Context. 
          -> DO NOT use tools.
        
        - **QUANTITATIVE QUESTIONS** (e.g. "Calculate Margin", "Growth Rate"): 
          -> First, FIND the exact raw numbers in Context.
          -> Then, CALL the 'calculate_margin' or 'calculate_growth' tool.
          -> DO NOT calculate mentally.
        
        - **IMPOSSIBLE QUESTIONS** (e.g. "Future Stock Price", "Next Year's Revenue"):
          -> State "I cannot predict future outcomes based on historical filings."
          -> DO NOT guess or offer to calculate hypotheticals.
          
        MANDATORY: Cite your sources as [Page X].
        """
        
        # Filter out old system messages to keep context clean
        history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        
        messages = [SystemMessage(content=system_prompt)] + history
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyst")
        
        # Conditional Edge
        workflow.add_conditional_edges(
            "analyst",
            self._should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "analyst") # Loop back after tool execution
        
        return workflow.compile()

    def _should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            print(f"🛠️  [Graph] Calling Tool: {last_message.tool_calls[0]['name']}")
            return "tools"
        return END

    def _identify_ticker(self, query: str) -> str:
        q = query.upper()
        if "AAPL" in q or "APPLE" in q: return "AAPL"
        if "MSFT" in q or "MICROSOFT" in q: return "MSFT"
        if "TSLA" in q or "TESLA" in q: return "TSLA"
        if "NVDA" in q or "NVIDIA" in q: return "NVDA"
        if "GOOG" in q or "GOOGLE" in q: return "GOOG"
        return "UNKNOWN"

    # --- Ingestion Logic ---
    def ingest_data(self):
        print(f"📂 Scanning company folders in {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return

        subfolders = [f.path for f in os.scandir(self.data_path) if f.is_dir()]
        
        if not subfolders:
             print("❌ No company folders found inside data/.")
             return

        for folder in subfolders:
            ticker = os.path.basename(folder).upper()
            print(f"\n🏭 Processing Company: {ticker}...")
            
            # 1. Load PDFs (Case Insensitive)
            loader_pdf_lower = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
            loader_pdf_upper = DirectoryLoader(folder, glob="*.PDF", loader_cls=PyPDFLoader)
            
            # 2. Load Word Docs (.docx)
            loader_docx = DirectoryLoader(folder, glob="*.docx", loader_cls=Docx2txtLoader)
            
            # Combine all
            docs = loader_pdf_lower.load() + loader_pdf_upper.load() + loader_docx.load()
            
            if not docs:
                print(f"   ⚠️ No documents found for {ticker} (Checked .pdf, .PDF, .docx).")
                continue
                
            # Add Rich Metadata
            for doc in docs:
                filename = os.path.basename(doc.metadata.get('source', ''))
                doc.metadata["ticker"] = ticker
                doc.metadata["filename"] = filename
                doc.metadata["page"] = doc.metadata.get('page', 1) 

            # Optimized Splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, 
                chunk_overlap=300,
                separators=["\n\n", "\n", "Table of Contents", "."]
            )
            splits = text_splitter.split_documents(docs)
            
            collection_name = f"docs_{ticker}"
            print(f"   💾 Vectorizing {len(splits)} chunks into '{collection_name}'...")
            
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            
        print("\n✅ Institutional Knowledge Base Ready!")

    def analyze(self, query: str) -> str:
        """Entry point for the graph"""
        print(f"🤖 [Graph] Starting workflow for: '{query}'")
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        final_answer = ""
        # Using invoke instead of stream for simplicity in main loop
        result = self.app.invoke(inputs)
        
        if "messages" in result:
            final_answer = result["messages"][-1].content
        else:
            final_answer = "Error in graph execution."
            
        return final_answer
