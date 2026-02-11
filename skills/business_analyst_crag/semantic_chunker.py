"""Proposition-based semantic chunking utilities.

This is a pragmatic implementation for the POC:
- If OPENAI_API_KEY is available, use a small LLM to split text into atomic propositions.
- Otherwise, fall back to a deterministic sentence splitter that still avoids fixed token windows.

The output is a list of short, standalone statements suitable for embedding.
"""

from __future__ import annotations

import os
import re
from typing import List


def _fallback_sentence_split(text: str, max_len: int = 220) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    # Rough sentence boundaries. This is not perfect but avoids fixed-size token windows.
    sents = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # If sentence is too long, split on semicolons/commas.
        if len(s) > max_len:
            parts = re.split(r"[;:]\s+|,\s+", s)
            for p in parts:
                p = p.strip()
                if 20 <= len(p) <= 500:
                    out.append(p)
        else:
            if 20 <= len(s) <= 500:
                out.append(s)

    # Dedupe while preserving order
    seen = set()
    dedup = []
    for x in out:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(x)
    return dedup


def proposition_chunk(text: str, max_props: int = 50) -> List[str]:
    """Split input text into standalone propositions."""
    text = (text or "").strip()
    if not text:
        return []

    # Optional LLM-based splitter (kept lightweight)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Split the following text into atomic, standalone propositions. "
                "Each proposition must be a single sentence that can stand alone without context. "
                "Return ONLY a JSON array of strings.\n\nTEXT:\n" + text[:8000]
            )
            resp = client.chat.completions.create(
                model=os.getenv("PROPOSITION_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            # Extract JSON array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return _fallback_sentence_split(text)[:max_props]
            import json

            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                props = [p.strip() for p in arr if isinstance(p, str) and 10 < len(p.strip()) < 500]
                return props[:max_props] if props else _fallback_sentence_split(text)[:max_props]
        except Exception:
            return _fallback_sentence_split(text)[:max_props]

    # Deterministic fallback
    return _fallback_sentence_split(text)[:max_props]
