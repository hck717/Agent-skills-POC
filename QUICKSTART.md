# Quick Start Guide

## 🚀 Complete Setup in 5 Minutes

### 1. Pull Latest Changes

```bash
cd Agent-skills-POC
git pull origin main
```

### 2. Activate Virtual Environment

```bash
# If not created yet
python3.11 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Ollama (for Business Analyst)

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 5. Test Your Setup

```bash
# Quick API test
python test_api.py YOUR_PERPLEXITY_API_KEY

# Full system test (recommended)
export PERPLEXITY_API_KEY="your-key"
python test_full_system.py
```

**Expected output:**
```
🎉 ALL TESTS PASSED!
✅ Your system is fully configured and working!
```

### 6. Run Streamlit

```bash
streamlit run app.py
```

---

## 🎯 What's New (v2.1)

### Fixed Issues

✅ **Perplexity API Connection**
- Updated to valid Sonar models (`sonar`, `sonar-pro`)
- Better error handling (400, 401, 429)
- Connection test utility

✅ **ReAct Loop Improvements**
- Now encourages full 5-iteration loops
- Better prompting to call specialist agents
- Won't finish early unless comprehensive

✅ **Enhanced Synthesis**
- Proper integration of specialist outputs
- Fallback handling when no specialists called
- Uses `sonar-pro` for better quality

✅ **UI Enhancements**
- API key input in sidebar
- Better error messages
- Real-time status indicators

### New Features

✨ **Test Scripts**
- `test_api.py` - Quick API connection test
- `test_full_system.py` - Comprehensive 5-test suite

✨ **Better Documentation**
- `docs/TROUBLESHOOTING.md` - Complete debug guide
- `QUICKSTART.md` - This guide!

✨ **Vector DB Integration**
- Business Analyst properly connects to ChromaDB
- RAG retrieval from 10-K documents
- BERT reranking for better relevance

---

## 📊 How It Works Now

### The ReAct Loop (5 Iterations)

```
🔁 ITERATION 1/5
├─ 🧠 THOUGHT: "Need to analyze 10-K filing for risks"
├─ ⚡ ACTION: call_specialist(business_analyst)
└─ 👁️ OBSERVATION: [Risk analysis from 10-K]

🔁 ITERATION 2/5  
├─ 🧠 THOUGHT: "Need quantitative metrics"
├─ ⚡ ACTION: call_specialist(quantitative_analyst)
└─ 👁️ OBSERVATION: [Financial ratios]

🔁 ITERATION 3/5
├─ 🧠 THOUGHT: "Need market context"
├─ ⚡ ACTION: call_specialist(market_analyst)
└─ 👁️ OBSERVATION: [Market sentiment]

🔁 ITERATION 4/5
├─ 🧠 THOUGHT: "Have comprehensive data"
├─ ⚡ ACTION: finish
└─ 🎯 Complete

📝 SYNTHESIS
└─ Combining 3 specialist outputs into report...
```

### Specialist Agent: Business Analyst

```
Query → Business Analyst
         │
         ├─ 📂 Load Documents from ./data/
         │
         ├─ 💾 Query ChromaDB (Vector Search)
         │
         ├─ 🎯 BERT Reranking (Top-K)
         │
         ├─ 🤖 LangGraph Processing
         │
         └─ 📝 Structured Analysis
```

---

## 📋 Example Queries

### Simple (May Finish Early)

```
❌ "What are Apple's main products?"
   → Iteration 1: Search → Finish
   → No specialist agents needed
```

### Complex (Full 5 Iterations)

```
✅ "Analyze Apple's competitive risks from their latest 10-K filing and compare their business model to key competitors"
   → Iteration 1: Business Analyst (10-K risk analysis)
   → Iteration 2: Business Analyst (business model)
   → Iteration 3: Industry Analyst (competitor analysis)
   → Iteration 4: Quantitative Analyst (metrics comparison)
   → Iteration 5: Finish → Synthesis
```

### Optimal Query Structure

**Good queries include:**
- Multiple aspects (risks + business model + market position)
- Specific document references ("from 10-K", "in MD&A section")
- Comparative elements ("compare to...", "versus...")
- Analytical depth ("evaluate", "analyze", "assess")

**Example:**
```
Based on Tesla's latest 10-K filing:
1. What are the key risk factors in the Risk Factors section?
2. How has their business model evolved according to MD&A?
3. What are their competitive advantages mentioned?
4. Provide specific quotes and page references.
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
export PERPLEXITY_API_KEY="pplx-xxxxxxxxxxxxxxxx"

# Optional
export EODHD_API_KEY="your-key"  # For market data
export HF_TOKEN="your-token"     # For faster HuggingFace downloads
```

### Folder Structure

```
Agent-skills-POC/
├── data/                    # Add your 10-K PDFs here
│   ├── AAPL/
│   │   └── apple_10k_2024.pdf
│   ├── TSLA/
│   │   └── tesla_10k_2024.pdf
│   └── .gitkeep
│
├── storage/
│   └── chroma_db/         # Auto-created vector DB
│
├── skills/
│   └── business_analyst/  # RAG-powered agent
│
└── orchestrator_react.py  # ReAct orchestrator
```

### Max Iterations Setting

In Streamlit sidebar:
- **Min: 1** - Single specialist call
- **Default: 5** - Comprehensive multi-agent analysis
- **Max: 10** - Deep exhaustive research

**Recommendation:** Start with 5, adjust based on query complexity.

---

## 🔍 Monitoring ReAct Loop

### In Terminal

```bash
# Watch detailed logs
python main_orchestrated.py

# You'll see:
🔁 REACT-BASED EQUITY RESEARCH ORCHESTRATOR
📥 Query: [your query]
🔄 Max Iterations: 5
📊 Registered Agents: business_analyst

----------------------------------------------------------------------
ITERATION 1/5
----------------------------------------------------------------------
🧠 [THOUGHT 1] Reasoning about next action...
   💭 [Reasoning output]
   ⚡ Action: call_specialist
      Agent: business_analyst
      Task: [Task description]

⚙️ [ACTION] Executing call_specialist...
   🤖 Calling business_analyst...
   ✅ business_analyst completed (5432 chars)

👁️ [OBSERVATION] [Results preview]...
```

### In Streamlit UI

1. Enable **"Auto-show ReAct Trace"** in sidebar
2. Run query
3. See trace automatically displayed below report
4. Check metrics:
   - Iterations: How many loops executed
   - Duration: Total time
   - Agents Called: Which specialists were used

---

## ✅ Verification Checklist

### Before Using

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list | grep streamlit`)
- [ ] Ollama running (`ollama list`)
- [ ] Models downloaded (qwen2.5:7b, nomic-embed-text)
- [ ] Perplexity API key set
- [ ] At least one PDF in `./data/`

### Test Commands

```bash
# 1. Quick test
python test_api.py

# 2. Full system test
python test_full_system.py

# 3. Manual orchestrator test
python main_orchestrated.py
# Ask: "Analyze risks from Apple 10-K"

# 4. Streamlit
streamlit run app.py
```

---

## 🐛 Common Issues

### "No specialist outputs to synthesize"

**Cause:** ReAct loop finished without calling agents

**Fix:**
- Ask more complex questions
- Ensure documents exist in `./data/`
- Check Business Analyst is registered
- See improved prompting in v2.1

### "Business Analyst initialization error"

**Cause:** Ollama not running or models missing

**Fix:**
```bash
# Check Ollama
ollama list  # Should show models

# If not running
ollama serve

# Pull models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### "Invalid model error"

**Cause:** Using old model names

**Fix:** Updated to Sonar models in v2.1
- ✅ Valid: `sonar`, `sonar-pro`, `sonar-reasoning-pro`
- ❌ Invalid: `llama-3.1-sonar-*`, `pplx-7b-chat`

### "Rate limit exceeded"

**Cause:** Too many API calls

**Fix:**
- Wait 60 seconds
- Reduce max iterations
- Check your API plan limits

---

## 📚 Resources

### Documentation

- [README.md](README.md) - Project overview
- [docs/REACT_FRAMEWORK.md](docs/REACT_FRAMEWORK.md) - ReAct architecture
- [docs/SPECIALIST_AGENTS.md](docs/SPECIALIST_AGENTS.md) - Agent details
- [docs/UI_GUIDE.md](docs/UI_GUIDE.md) - Streamlit usage
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Debug guide

### Test Scripts

- `test_api.py` - API connection test
- `test_full_system.py` - Comprehensive test suite

### Getting Help

1. Check `docs/TROUBLESHOOTING.md`
2. Run `python test_full_system.py`
3. Open GitHub issue with:
   - Error message
   - Test results
   - System info (OS, Python version)

---

## 🎓 Learning Path

### Beginner

1. Run `test_full_system.py` → Verify everything works
2. Try simple query in Streamlit
3. Read ReAct trace to understand loop
4. Add your own 10-K PDF to `./data/`

### Intermediate

1. Try complex multi-aspect queries
2. Experiment with max iterations (3-10)
3. Review Business Analyst code
4. Understand ChromaDB vector search

### Advanced

1. Implement new specialist agent
2. Customize synthesis prompts
3. Add real-time data sources
4. Integrate with your workflows

---

**Ready to start? Run:**

```bash
python test_full_system.py && streamlit run app.py
```

🚀 **Happy researching!**
