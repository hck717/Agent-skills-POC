import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

class BusinessAnalystAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        # --- Model Configuration ---
        # Chat Model for Reasoning (Use a smart model)
        self.chat_model_name = "qwen2.5:7b" 
        # Embedding Model for Vectorization (Use a specialized small model)
        self.embed_model_name = "nomic-embed-text"
        
        print(f"🔧 Initializing Agent with Chat: {self.chat_model_name} | Embed: {self.embed_model_name}")

        # Initialize LLM
        self.llm = ChatOllama(model=self.chat_model_name, temperature=0.1) # Low temp for factual accuracy
        
        # Initialize Embeddings
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        
        # Vector Store Placeholder
        self.vectorstore = None
        self._load_vectorstore() # Try loading immediately if exists

    def _load_vectorstore(self):
        """Internal method to load existing DB without re-ingesting"""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            print("🔄 Loading existing Knowledge Base...")
            self.vectorstore = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
        else:
            print("⚠️ No Knowledge Base found. Please run ingest_data().")

    def ingest_data(self):
        """Process PDFs and build the Vector Database"""
        print(f"📂 Scanning documents in {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"⚠️ Created data directory. Please add PDF files to {self.data_path}")
            return

        # Load PDFs
        loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("❌ No PDF files found!")
            return

        print(f"📄 Loaded {len(docs)} pages. Splitting text...")
        
        # Optimized Splitting for Financial Docs
        # Larger chunks (2000 chars) to keep context of tables/paragraphs together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        print(f"💾 Vectorizing {len(splits)} chunks (this may take time)...")
        
        # Force Clean Start (Optional: remove old DB to prevent duplicates)
        # shutil.rmtree(self.db_path, ignore_errors=True) 
        
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ Knowledge Base Rebuilt Successfully!")

    def analyze(self, query: str) -> str:
        """Execute the RAG analysis pipeline"""
        if not self.vectorstore:
            return "❌ Error: Knowledge base is not loaded. Run ingest_data() first."

        # 1. Retrieval (Get Top 7 chunks for broader context)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})
        
        # 2. Senior Analyst Prompt Engineering
        template = """
        ### Role
        You are a Senior Equity Research Analyst (TMT Sector) at a top-tier investment bank. 
        Your goal is to provide deep, fact-based analysis based *strictly* on the provided 10-K filings.

        ### Instructions
        1. **Evidence-Based**: Only use the provided context. If the context is missing info, state "Insufficient data in filings".
        2. **Financial Precision**: Use specific numbers, margins (%), and dates whenever available.
        3. **Structured Output**: Format your response in Markdown with clear headers.
        4. **Critical Thinking**: Don't just summarize; analyze *implications* (e.g., "Inventories rose 20%, suggesting potential demand softening").

        ### Context (from 10-K Filings)
        {context}

        ### User Query
        {question}

        ### Analyst Report
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        # 3. RAG Chain Execution
        print(f"🔍 Analyzing: '{query}'...")
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(query)

    def format_docs(self, docs: List[Document]) -> str:
        """Format retrieved docs with source attribution"""
        formatted_text = ""
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 0)
            content = doc.page_content.replace("\n", " ")
            formatted_text += f"[Ref {i+1} | {source} p.{page}]: {content}\n\n"
        return formatted_text

# Self-test
if __name__ == "__main__":
    agent = BusinessAnalystAgent(data_path="../../data", db_path="../../storage/chroma_db")
    # agent.ingest_data()
    print(agent.analyze("What are the key risk factors mentioned regarding AI competition?"))
