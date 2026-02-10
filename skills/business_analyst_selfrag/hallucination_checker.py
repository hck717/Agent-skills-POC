"""\nHallucination Checking Module\n\nVerifies that generated answers are grounded in source documents.\nPrevents LLM from making unsupported claims.\n"""

import re
from typing import Tuple, List, Dict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class HallucinationChecker:
    """
    Checks if generated analysis is grounded in source documents.
    
    Uses LLM to verify each claim has document support.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:8b", temperature: float = 0.0):
        """
        Args:
            model_name: LLM for verification
            temperature: 0.0 for strict grounding checks
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
    
    def check_hallucination(
        self,
        analysis: str,
        context: str,
        max_context_length: int = 4000
    ) -> Tuple[bool, str, Dict]:
        """
        Check if analysis is grounded in context.
        
        Args:
            analysis: Generated answer to verify
            context: Source documents used for generation
            max_context_length: Truncate context to fit in prompt
        
        Returns:
            (is_grounded, feedback, metadata)
            - is_grounded: True if no hallucinations
            - feedback: Explanation of issues (if any)
            - metadata: Statistics about the check
        """
        print(f"\n🔍 [Hallucination Check] Verifying answer grounding...")
        
        # Truncate context if too long
        context_truncated = context[:max_context_length]
        if len(context) > max_context_length:
            context_truncated += "\n... [context truncated for brevity]"
        
        # Create verification prompt
        check_prompt = f"""You are a fact-checking assistant.

Your task: Verify that EVERY factual claim in the Analysis is supported by the Source Documents.

=== SOURCE DOCUMENTS ===
{context_truncated}

=== GENERATED ANALYSIS ===
{analysis}

=== INSTRUCTIONS ===
1. Read each sentence in the Analysis
2. Check if it's supported by the Source Documents
3. If ANY claim is unsupported, list it

Respond in this format:

GROUNDED: yes/no
REASON: [If 'no', explain which claims are unsupported]

Example responses:

GROUNDED: yes\nREASON: All claims are directly supported by source documents.

GROUNDED: no\nREASON: The claim \"Apple has 50% market share\" is not found in any source document. Sources only mention 23.4% market share.

Your response:"""
        
        response = self.llm.invoke([HumanMessage(content=check_prompt)])
        result = response.content.strip()
        
        # Parse response
        is_grounded = self._parse_grounded(result)
        reason = self._extract_reason(result)
        
        # Extract statistics
        citation_count = len(re.findall(r'\[\d+\]', analysis))
        analysis_length = len(analysis.split())
        
        metadata = {
            'is_grounded': is_grounded,
            'citation_count': citation_count,
            'analysis_word_count': analysis_length,
            'llm_response': result
        }
        
        if is_grounded:
            print(f"   ✅ PASS - All claims are grounded in source documents")
            print(f"   📊 Citations: {citation_count} | Words: {analysis_length}")
        else:
            print(f"   ❌ FAIL - Hallucinations detected")
            print(f"   ⚠️ Issue: {reason[:200]}...")
        
        return is_grounded, reason, metadata
    
    def _parse_grounded(self, response: str) -> bool:
        """
        Parse GROUNDED field from LLM response.
        """
        response_lower = response.lower()
        
        # Look for GROUNDED: yes/no
        if 'grounded: yes' in response_lower:
            return True
        if 'grounded: no' in response_lower:
            return False
        
        # Fallback: check for yes/no in first 100 chars
        first_line = response_lower[:100]
        if 'yes' in first_line and 'no' not in first_line:
            return True
        
        # Default to False (strict grounding)
        return False
    
    def _extract_reason(self, response: str) -> str:
        """
        Extract REASON field from LLM response.
        """
        # Try to extract after REASON:
        match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: return full response
        return response
    
    def suggest_improvements(self, analysis: str, context: str) -> str:
        """
        Suggest how to improve analysis to be more grounded.
        
        Args:
            analysis: Current analysis with issues
            context: Source documents
        
        Returns:
            Improvement suggestions
        """
        prompt = f"""You are an editor improving fact accuracy.

Source Documents:
{context[:2000]}...

Current Analysis (with hallucinations):
{analysis}

Task: Suggest 3 specific improvements to make the analysis fully grounded in sources.

Format:
1. [Specific suggestion]
2. [Specific suggestion]
3. [Specific suggestion]

Suggestions:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
