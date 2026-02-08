# Agent-skills-POC

# 1. 激活虛擬環境
/opt/homebrew/bin/python3.11 -m venv .venv

source .venv/bin/activate

# 2. 安裝依賴 (如果你修改過 requirements.txt)
pip install -r requirements.txt


# import EODHD key
export EODHD_API_KEY="6957671cac2858.27368157"


# 3. 確保 Ollama 在後台運行 (如果沒開)
# (建議開一個新 Terminal 窗口運行這個)
ollama serve 

# Chat 模型 (負責思考)
ollama pull qwen2.5:7b

# Embedding 模型 (負責讀文件)
ollama pull nomic-embed-text

# 清空舊數據庫 (因為結構變了):
rm -rf storage/chroma_db

python main.py





依家個新架構（Architecture）已經由舊時單向嘅「直線流程」升級做一個真正識思考嘅 **"ReAct Loop"（邊諗邊做循環）**。

簡單嚟講，以前係「搵資料 $\rightarrow$ 答問題」，而家係模擬一個真人分析師嘅工作模式：「思考 $\rightarrow$ 搵工具 $\rightarrow$ 睇結果 $\rightarrow$ 再思考 $\rightarrow$ 寫報告」。

以下係成個 Flow 嘅廣東話拆解：

### 1. 核心大腦 (The Brain: Analyst Node)
*   **角色**：呢個係 **Qwen 2.5** 模型。
*   **做啲咩**：當你問佢嘢（例如：「比較 Apple 同 Microsoft 嘅 Margin」），佢唔會即刻答你。佢會先**「諗計」**（Planning）。
*   **思考過程**：
    > 「老細問兩間公司，我手頭上無資料。首先我要查 Apple，之後查 Microsoft，最後用 Python 計個數出嚟。」

### 2. 工具箱 (The Tools)
大腦可以隨時伸手攞以下三樣法寶：

1.  **`retrieve_filings` (內部知識庫 + BERT Rerank)**
    *   **以前**：只係簡單 Keyword search，搵到垃圾都照俾你。
    *   **而家**：搵完之後，會用 **BERT Reranker** 幫你再過濾一次，確保搵出嚟嘅係真係有關嘅「財務數據」或者「風險披露」，精準度高好多。

2.  **`get_market_data` (即時報價)**
    *   直接連去 **Yahoo Finance**。如果你問「依家股價幾多」、「PE Ratio 幾多」，佢識得出去上網攞最新數字，唔會再話「我數據只去到 2023 年」。

3.  **`financial_analysis_sandbox` (Python 計數機)**
    *   **最重要嘅升級**。LLM 心算好差，經常計錯數。
    *   而家個 Agent 識得寫 **Python (Pandas) Code** 去計複雜嘅數（例如 CAGR, Margin Growth），保證計出嚟嘅數係 100% 準確。

### 3. 實戰流程舉例 (The Loop)
假設你問：**「幫我計下 Apple 嘅 Net Margin，同埋睇下依家股價幾多？」**

1.  **第一轉 (Turn 1)**：
    *   **Agent 諗**：「我要搵 Net Income 同 Revenue 先計到 Margin。」
    *   **行動**：Call `retrieve_filings(ticker="AAPL", query="Net Income and Revenue 2024")`。
    *   **結果**：系統由 PDF 搵到相關數字俾佢。

2.  **第二轉 (Turn 2)**：
    *   **Agent 諗**：「有數喇，但我唔會心算。我要寫 Code 計。」
    *   **行動**：Call `financial_analysis_sandbox(code="margin = 96995 / 383285 * 100...")`。
    *   **結果**：Python 算返個準確百份比 (e.g., 25.31%) 俾佢。

3.  **第三轉 (Turn 3)**：
    *   **Agent 諗**：「搞掂 Margin，而家要搵股價。」
    *   **行動**：Call `get_market_data(ticker="AAPL")`。
    *   **結果**：Yahoo Finance 俾個現價 (e.g., $225.50) 佢。

4.  **結局 (Final Answer)**：
    *   **Agent 諗**：「齊晒料，可以交貨。」
    *   **輸出**：寫一段完整報告俾你，話你知 Margin 係幾多（連埋 Citation 來源），同埋依家股價幾錢。

### 總結升級重點
*   **主動性**：以前搵唔到資料就投降；而家佢識換個 keyword 再搵，或者分幾步做。
*   **準確性**：用 Python 代替心算，用 BERT 代替普通 Search。
*   **實時性**：識得睇 Live 股價，唔再係「書呆子」。

呢個就係所謂嘅 **Agentic RAG**，佢唔止係一個 Search Engine，而係一個識用工具嘅 **AI 員工**。