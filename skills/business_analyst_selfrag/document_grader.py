"""\nDocument Grading Module\n\nLLM-based relevance checker that filters retrieved documents\nbefore passing to generation stage.\n\nPrevents hallucination by ensuring only relevant context is used.\n"""

import re
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class DocumentGrader:
    """
    Grades documents for relevance to the user query.
    
    Uses LLM to evaluate each document:
    - "yes" = relevant, keep
    - "no" = irrelevant, discard
    """
    
    def __init__(self, model_name: str = "deepseek-r1:8b", temperature: float = 0.0):
        """
        Args:
            model_name: LLM for grading
            temperature: 0.0 for deterministic yes/no decisions
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
    
    def grade_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.3
    ) -> Tuple[List[Document], Dict]:
        """
        Grade documents for relevance.
        
        Args:
            query: User question
            documents: Retrieved documents to grade
            threshold: Minimum percentage of docs that must pass (0.3 = 30%)
        
        Returns:
            (filtered_documents, metadata)
            metadata = {
                'passed': int,
                'failed': int,
                'pass_rate': float,
                'meets_threshold': bool
            }
        """
        if not documents:
            return [], {
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
                'meets_threshold': False
            }
        
        print(f"\n📋 [Grade] Evaluating {len(documents)} documents for relevance...")
        
        passed_docs = []
        failed_count = 0
        
        for i, doc in enumerate(documents):
            # Create grading prompt
            grade_prompt = f"""You are a document relevance grader.

User Question: {query}

Document Content (first 600 chars):
{doc.page_content[:600]}...

Task: Is this document relevant to answering the user's question?

Respond with ONLY:
- "yes" if relevant
- "no" if irrelevant

Response:"""
            
            response = self.llm.invoke([HumanMessage(content=grade_prompt)])
            decision = response.content.strip().lower()
            
            # Parse decision
            if 'yes' in decision:
                passed_docs.append(doc)
                status = "✅ PASS"
            else:
                failed_count += 1
                status = "❌ FAIL"
            
            # Show progress every 5 docs
            if (i + 1) % 5 == 0 or (i + 1) == len(documents):
                print(f"   Progress: {i + 1}/{len(documents)} | Last: {status}")
        
        # Calculate statistics
        passed_count = len(passed_docs)
        pass_rate = passed_count / len(documents)
        meets_threshold = pass_rate >= threshold
        
        metadata = {
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': pass_rate,
            'meets_threshold': meets_threshold
        }
        
        print(f"\n   📊 Results: {passed_count} passed, {failed_count} failed")
        print(f"   📈 Pass rate: {pass_rate:.1%} (threshold: {threshold:.1%})")
        
        if meets_threshold:
            print(f"   ✅ Threshold met - proceeding with filtered documents")
        else:
            print(f"   ⚠️ Threshold NOT met - may need web search fallback")
        
        return passed_docs, metadata
    
    def extract_context_from_documents(
        self,
        documents: List[Document],
        include_metadata: bool = True
    ) -> str:
        """
        Convert graded documents to context string.
        
        Args:
            documents: Filtered documents
            include_metadata: Include source citations
        
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for doc in documents:
            if include_metadata:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"--- SOURCE: {source} (Page {page}) ---\n{doc.page_content}"
                )
            else:
                context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)
