# 🧠 Prompt Engineering for Financial Agents

This guide explains the persona system used in `graph_agent.py`. The agent automatically selects one of the following prompts based on keywords in the user's query.

## 1. The "Forensic Analyst" (Qualitative)
**Trigger Keywords:** `risk`, `flag`, `audit`, `forensic`, `warning`.
**Use Case:** Deep dives into 10-K text, finding risks, governance checks.
**Goal:** Translate corporate euphemisms into investment reality.

## 2. The "Quant Analyst" (Quantitative)
**Trigger Keywords:** `calculate`, `compare`, `margin`, `table`, `growth`.
**Use Case:** Extracting tables, calculating ratios, comparing peers.
**Goal:** Strict data accuracy and tabular presentation.

## 3. The "General Analyst" (Default)
**Trigger Keywords:** Anything else.
**Use Case:** General summaries, business overviews, "what does this company do?".
**Goal:** Clear, helpful, and cited answers.

---
**How to Add a New Persona:**
1. Create a new `.md` file in the `prompts/` folder (e.g., `prompts/trader.md`).
2. Update the `analyst_node` function in `graph_agent.py` to add a new `elif` condition for your trigger keywords.
