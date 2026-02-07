import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class BusinessAnalystAgent:
    def __init__(self, data_path="./data", db_path="./storage/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        self.model_name = "llama3"  # Local Ollama Model

        # Initialize LLM
        self.llm = ChatOllama(model=self.model_name, temperature=0.2)

        # Initialize Embeddings (running locally)
        self.embeddings = OllamaEmbeddings(model=self.model_name)

        # Vector Store
        self.vectorstore = None

    def ingest_data(self):
        """讀取 PDF 並建立向量數據庫"""
        print(f"📂 Loading documents from {self.data_path}...")

        # Check if data path exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"⚠️ Created data directory at {self.data_path}. Please add PDF files.")
            return

        loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        if not docs:
            print("❌ No PDF files found in data folder!")
            return

        print(f"📄 Loaded {len(docs)} pages. Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        print("💾 Creating Vector Database (this may take a while)...")
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ Knowledge Base Ready!")

    def analyze(self, query):
        """執行分析任務"""
        if not self.vectorstore:
            # 如果數據庫已存在，嘗試加載
            if os.path.exists(self.db_path) and os.listdir(self.db_path):
                 print("🔄 Loading existing Vector Database...")
                 self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            else:
                print("⚠️ Knowledge base not found. Please run ingest_data() first!")
                return "No knowledge base found. Please ingest data."

        # RAG Retrieval
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Define the Business Analyst Persona
        template = """
        You are a Senior Business Analyst at a top-tier investment bank.
        Use the following pieces of retrieved context from the 10-K filings to answer the user's question.

        Focus on:
        1. Corporate Operating Model
        2. Competitive Advantage (Moat)
        3. Industry Landscape

        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer professional, concise, and structured.

        Context: {context}

        Question: {question}

        Analysis:
        """

        prompt = ChatPromptTemplate.from_template(template)

        # RAG Chain
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print(f"🤖 Business Analyst is thinking about: '{query}'...")
        return rag_chain.invoke(query)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

# For testing this module directly
if __name__ == "__main__":
    agent = BusinessAnalystAgent(data_path="../../data", db_path="../../storage/chroma_db")
    # agent.ingest_data() 
    response = agent.analyze("What is the primary business model?")
    print(response)
