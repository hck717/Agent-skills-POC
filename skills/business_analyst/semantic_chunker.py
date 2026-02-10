"""\nSemantic Chunking Module\n\nUses embedding-based similarity to create semantically coherent chunks\nthat respect natural document boundaries (paragraphs, topics, sections).\n\nBetter for financial documents where logical structure matters.\n"""

import numpy as np
from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Splits documents based on semantic similarity rather than fixed size.
    
    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Calculate similarity between adjacent sentences
    4. Create chunk boundaries where similarity drops below threshold
    """
    
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 80,
        min_chunk_size: int = 500,
        max_chunk_size: int = 4000
    ):
        """
        Args:
            embeddings: Embedding model for semantic similarity
            breakpoint_threshold_type: 'percentile' or 'standard_deviation'
            breakpoint_threshold_amount: 80 = 80th percentile for breaks
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk (hard limit)
        """
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Fallback to recursive splitter for oversized chunks
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic similarity.
        """
        all_splits = []
        
        for doc in documents:
            splits = self._split_single_document(doc)
            all_splits.extend(splits)
        
        return all_splits
    
    def _split_single_document(self, document: Document) -> List[Document]:
        """
        Split a single document semantically.
        """
        text = document.page_content
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            # Too short, return as-is
            return [document]
        
        # Step 2: Embed sentences
        print(f"   🧮 Embedding {len(sentences)} sentences for semantic chunking...")
        embeddings_list = []
        
        for sentence in sentences:
            if sentence.strip():
                emb = self.embeddings.embed_query(sentence)
                embeddings_list.append(emb)
            else:
                # Empty sentence, use zero vector
                embeddings_list.append([0.0] * 768)
        
        # Step 3: Calculate similarity distances between adjacent sentences
        distances = []
        for i in range(len(embeddings_list) - 1):
            similarity = self._cosine_similarity(
                embeddings_list[i],
                embeddings_list[i + 1]
            )
            # Convert similarity to distance (0 = same, 1 = opposite)
            distance = 1 - similarity
            distances.append(distance)
        
        # Step 4: Determine breakpoints
        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = np.mean(distances) + (self.breakpoint_threshold_amount * np.std(distances))
        else:
            threshold = 0.5  # Default fallback
        
        # Step 5: Create chunks at breakpoints
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, distance in enumerate(distances):
            if distance > threshold:
                # High distance = semantic break
                chunk_text = " ".join(current_chunk)
                
                # Respect min/max size
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                    current_chunk = [sentences[i + 1]]
                else:
                    # Too small, keep accumulating
                    current_chunk.append(sentences[i + 1])
            else:
                # Low distance = same topic, continue
                current_chunk.append(sentences[i + 1])
            
            # Check max size
            if len(" ".join(current_chunk)) > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Step 6: Handle oversized chunks with fallback
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Use recursive splitter as fallback
                oversized_doc = Document(page_content=chunk, metadata=document.metadata)
                sub_chunks = self.fallback_splitter.split_documents([oversized_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(Document(
                    page_content=chunk,
                    metadata=document.metadata
                ))
        
        print(f"   ✅ Created {len(final_chunks)} semantic chunks (from {len(sentences)} sentences)")
        return final_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.
        """
        # Split on period + space, question mark, exclamation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
