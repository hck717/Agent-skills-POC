import os
import re
from typing import List, Dict, Any, Optional

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Document Processing
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Models
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# --- Tools with Error Handling ---
@tool
def calculate_growth(current: float, previous: float) -> str:
    """
    Calculates Year-Over-Year (YoY) growth percentage.
    Args:
        current: The value for the current period (must be a real number from filings).
        previous: The value for the previous period.
    """
    try:
        current = float(current)
        previous = float(previous)
        if previous == 0: return "Error: Previous value is zero."
        growth = ((current - previous) / previous) * 100
        return f"{growth:.2f}%"
    except ValueError:
        return "Error: Invalid inputs for calculation."

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
    except ValueError:
        return "Error: Invalid inputs for calculation."

class BusinessAnalystAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        # Models (Using Qwen 2.5 for better reasoning)
        self.chat_model_name = "qwen2.5:7b" 
        self.embed_model_name = "nomic-embed-text"
        
        print(f"🔧 Initializing Institutional Agent ({self.chat_model_name})...")

        # 1. Initialize LLM with Strict Temperature
        self.llm = ChatOllama(model=self.chat_model_name, temperature=0.0) # Zero temp for max factuality
        self.tools = [calculate_growth, calculate_margin]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 2. Initialize Embeddings
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        
        # 3. Vector Stores Cache
        self.vectorstores = {} 
        
        # 4. Memory
        self.message_history = ChatMessageHistory()

    def _get_vectorstore(self, collection_name):
        if collection_name not in self.vectorstores:
            self.vectorstores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
        return self.vectorstores[collection_name]

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
                # PDF has 'page', Docx doesn't usually have reliable page numbers via this loader
                doc.metadata["page"] = doc.metadata.get('page', 1) 

            # Optimized Splitter for Tables & Financials
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, # Larger chunks to keep tables intact
                chunk_overlap=300,
                separators=["\n\n", "\n", "Table of Contents", "."]
            )
            splits = text_splitter.split_documents(docs)
            
            collection_name = f"docs_{ticker}"
            print(f"   💾 Vectorizing {len(splits)} chunks into '{collection_name}'...")
            
            vs = self._get_vectorstore(collection_name)
            vs.add_documents(splits)
            
        print("\n✅ Institutional Knowledge Base Ready!")

    def _identify_ticker(self, query: str) -> str:
        query_upper = query.upper()
        if "APPLE" in query_upper or "AAPL" in query_upper: return "AAPL"
        if "MICROSOFT" in query_upper or "MSFT" in query_upper: return "MSFT"
        if "TESLA" in query_upper or "TSLA" in query_upper: return "TSLA"
        if "NVIDIA" in query_upper or "NVDA" in query_upper: return "NVDA"
        if "GOOGLE" in query_upper or "GOOG" in query_upper: return "GOOG"
        return "UNKNOWN"

    def analyze(self, query: str) -> str:
        # 1. Routing
        ticker = self._identify_ticker(query)
        context_text = ""
        
        if ticker == "UNKNOWN":
            return "❌ Target company not identified. Please specify the company name (e.g., Apple, Microsoft)."
        
        print(f"🔍 [Router] Target: {ticker}")
        collection_name = f"docs_{ticker}"
        vs = self._get_vectorstore(collection_name)
        
        # 2. Retrieval (Increased k for better recall)
        retriever = vs.as_retriever(search_kwargs={"k": 25}) 
        docs = retriever.invoke(query)
        
        if not docs:
            return f"⚠️ No documents found for {ticker}. Did you ingest data?"
            
        # Format Context with Citations
        context_text = "\n".join([
            f"--- Document: {d.metadata.get('filename')} (Page {d.metadata.get('page')}) ---\n{d.page_content}\n" 
            for d in docs
        ])
        
        # 3. Institutional Prompt
        system_prompt = f"""You are a Senior Equity Research Analyst covering {ticker}.
        
        ### MISSION
        Answer the User Query based STRICLY on the provided Context.
        
        ### CRITICAL RULES (VIOLATION = FIRED)
        1. **NO HYPOTHETICALS**: Never use "hypothetical values", "example numbers", or "assumed growth". If the specific number isn't in the text, state "Data not found in provided context".
        2. **CITATION MANDATORY**: Every financial claim must be backed by a source. Format: [Page X].
        3. **TOOL USAGE**: 
           - Only call 'calculate_growth' or 'calculate_margin' if you see the EXACT raw numbers in the Context.
           - Do not calculate if numbers are missing.
        4. **TONE**: Professional, objective, concise. No fluff.

        ### Context Data
        {context_text}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm_with_tools
        
        # 4. Execution with Memory
        self.message_history.add_user_message(query)
        
        # Thinking Indicator
        print(f"🤖 [Analyst] Reading {len(docs)} document chunks...")
        
        response = chain.invoke({
            "input": query,
            "history": self.message_history.messages
        })
        
        final_answer = ""
        
        # 5. Tool Loop
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🛠️  [Tool] Agent requesting: {response.tool_calls[0]['name']}")
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Execute
                tool_result = "Error"
                if tool_name == "calculate_growth":
                    tool_result = calculate_growth.invoke(tool_args)
                elif tool_name == "calculate_margin":
                    tool_result = calculate_margin.invoke(tool_args)
                
                print(f"   -> Result: {tool_result}")
                
                # Feed result back
                self.message_history.add_ai_message(response)
                self.message_history.add_message(HumanMessage(content=f"[SYSTEM] Tool '{tool_name}' output: {tool_result}. Now synthesize the final answer using this real data."))
                
                final_response = chain.invoke({
                    "input": "Synthesize final report.",
                    "history": self.message_history.messages
                })
                final_answer = final_response.content
        else:
            final_answer = response.content
            
        self.message_history.add_ai_message(final_answer)
        return final_answer

if __name__ == "__main__":
    pass
