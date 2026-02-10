# Business Analyst - Standard RAG (標準版)

## 📋 簡介

用 **Hybrid Search + RRF + BERT Reranking** 分析 10-K 文件嘅系統。

---

## 🏗️ 架構

```
用戶問題
   ↓
1. identify_node()    → 搵公司 (AAPL, MSFT...)
   ↓
2. research_node()    → 搵相關文件
   ├─ Vector Search (語義)
   ├─ BM25 Search (關鍵字)
   ├─ RRF 融合
   └─ BERT 重排 (top 8)
   ↓
3. analyst_node()     → LLM 生成分析
   └─ 自動加引用
```

---

## 📁 檔案結構

### **graph_agent.py** (主控)
核心 RAG 流程，包含 3 個 nodes：
- `identify_node()` - 提取公司名/股票代碼
- `research_node()` - 混合搜尋 + 重排
  - `_hybrid_search()` - Vector + BM25
  - `_reciprocal_rank_fusion()` - RRF 融合
  - `reranker.predict()` - BERT 評分
- `analyst_node()` - 生成分析
  - `_load_prompt()` - 載入 persona
  - `_inject_citations()` - 補返引用

### **agent.py** (Legacy)
舊版，包含 `calculate_growth()` 等 tools，保留作參考。

---

## 🔧 運作原理

### 1️⃣ **Hybrid Search（混合搜尋）**

```python
# Vector Search (語義相似)
Query: "供應鏈風險"
→ Embedding: [0.23, -0.45, ...]
→ 搵最似嘅 25 份文件

# BM25 Search (關鍵字)
Query: "供應鏈風險"
→ 關鍵字 match: "供應鏈", "風險"
→ 搵最多關鍵字嘅 25 份文件

# RRF Fusion
兩邊都 rank 高 → 最終分數高
```

### 2️⃣ **RRF 算法**

```
RRF score = 1/(60 + vector_rank) + 1/(60 + bm25_rank)

例子：
Doc A: Vector rank 1, BM25 rank 5
  → 1/61 + 1/65 = 0.0318

Doc B: Vector rank 2, BM25 rank 1  ← 兩邊都好
  → 1/62 + 1/61 = 0.0325 (最高)
```

### 3️⃣ **BERT Reranking**

```python
# RRF 之後仲有 25 份文件
# BERT 精確評分每份同 query 嘅相關度

reranker.predict([
  [query, doc1.content],
  [query, doc2.content],
  ...
])
→ [0.92, 0.88, 0.85, ..., 0.12, 0.08]
    ^^^^  ^^^^  ^^^^        ^^^^  ^^^^ 唔相關
    相關嘅

# 揀 top 8
```

---

## 🚀 使用

```python
from skills.business_analyst_standard.graph_agent import BusinessAnalystGraphAgent

# 初始化
agent = BusinessAnalystGraphAgent(
    data_path="./data",
    db_path="./storage/chroma_db"
)

# 載入文件
agent.ingest_data()

# 分析
result = agent.analyze("Apple 有咩供應鏈風險？")
```

---

## 📊 效能

| 指標 | 數值 |
|------|------|
| 延遲 | 75-110秒 |
| 準確度 | 88-93% |
| Chunk size | 4000字 |
| Top K | 8份文件 |

---

## 🔑 關鍵特點

✅ **三層搜尋** - Vector + BM25 + BERT  
✅ **自動引用** - 保留 SOURCE markers  
✅ **Persona 系統** - 根據問題揀角色  
✅ **穩定可靠** - 適合生產環境
