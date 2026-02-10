"""
Financial Data Extractor Module

Simulates XBRL data extraction using high-precision regex patterns for US GAAP terms.
Extracts key financial metrics (Revenue, Net Income, EPS, Assets) from text/tables.
"""

import re
from typing import Dict, List, Optional

class FinancialExtractor:
    """
    Extracts structured financial data from 10-K text/tables.
    Focuses on Consolidated Statement of Operations and Balance Sheet data.
    """
    
    def __init__(self):
        # Key US GAAP terms to look for
        self.metrics_map = {
            "Total Revenue": [
                r"Total revenue", r"Total net sales", r"Revenue", r"Net sales"
            ],
            "Net Income": [
                r"Net income", r"Net earnings", r"Net loss"
            ],
            "Operating Income": [
                r"Operating income", r"Operating profit", r"Income from operations"
            ],
            "EPS (Basic)": [
                r"Basic earnings per share", r"Basic net income per share", r"Earnings per share.*basic"
            ],
            "Total Assets": [
                r"Total assets"
            ],
            "Total Liabilities": [
                r"Total liabilities"
            ],
            "Cash & Equivalents": [
                r"Cash and cash equivalents"
            ]
        }
    
    def extract_metrics(self, text: str) -> Dict[str, str]:
        """
        Scan text for key financial metrics and their most likely values.
        Returns a dictionary of found metrics.
        """
        results = {}
        
        # Pre-process: normalize spaces
        text_norm = re.sub(r'\s+', ' ', text)
        
        for metric, patterns in self.metrics_map.items():
            for pattern in patterns:
                # Regex logic:
                # 1. Match the term (case insensitive)
                # 2. Followed by optional dots/spaces
                # 3. Look for currency symbol or numbers (allow parantheses for negatives)
                # 4. Capture the number (including millions/billions indicators if present, though rare in tables)
                
                # Pattern: Term ... $ 123,456
                regex = f"({pattern})" + r"[\s\.]*?\$?\s*?\(?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\)?"
                
                match = re.search(regex, text_norm, re.IGNORECASE)
                if match:
                    # Found a match!
                    # match.group(2) is the number
                    val = match.group(2)
                    
                    # Store if not already found (prioritize first/primary patterns)
                    if metric not in results:
                        results[metric] = val
                        break # Found best match for this metric
        
        return results

    def format_financials_for_context(self, financials: Dict[str, str]) -> str:
        """
        Format extracted data for LLM context.
        """
        if not financials:
            return ""
            
        lines = ["====== 📊 EXTRACTED FINANCIAL METRICS (US GAAP) ======"]
        for k, v in financials.items():
            lines.append(f"{k}: {v}")
        lines.append("====================================================\n")
        return "\n".join(lines)
