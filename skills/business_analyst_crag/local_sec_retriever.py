import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_sec_pdf_chunks(data_path: str, ticker: str) -> List[Dict[str, Any]]:
    folder = os.path.join(data_path, ticker.upper())
    if not os.path.isdir(folder):
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks: List[Dict[str, Any]] = []

    for fn in os.listdir(folder):
        if not fn.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(folder, fn)
        pages = PyPDFLoader(full_path).load()  # each Document corresponds to a page
        for d in pages:
            page = d.metadata.get("page", "N/A")
            for c in splitter.split_text(d.page_content):
                chunks.append({"text": c, "filename": fn, "page": page})

    return chunks

def naive_rank(query: str, chunks: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    q = set(query.lower().split())
    scored = []
    for ch in chunks:
        t = ch["text"].lower()
        score = sum(1 for w in q if w in t)
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]

def retrieve_local_sec(data_path: str, ticker: str, query: str, k: int = 6) -> List[str]:
    chunks = load_sec_pdf_chunks(data_path, ticker)
    top = naive_rank(query, chunks, k=k)
    out = []
    for ch in top:
        out.append(f"--- SOURCE: {ch['filename']} (Page {ch['page']}) ---\\n{ch['text']}")
    return out
