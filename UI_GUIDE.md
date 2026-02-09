# Streamlit UI Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit
# Or install all requirements
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export PERPLEXITY_API_KEY="your-perplexity-key"
export EODHD_API_KEY="your-eodhd-key"  # Optional
```

### 3. Start Ollama (for Business Analyst)

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

## 🖥️ UI Overview

### Main Interface

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                │  Main Content Area                │
│                         │                                   │
│  🔬 System Status       │  💬 Query Input                   │
│  ✅ Environment OK      │  [Text area for question]         │
│                         │  [🔍 Analyze] [🗑️ Clear] [💾 Save] │
│  🚀 Initialize System   │                                   │
│                         │  📊 Results                       │
│  🤖 Specialist Agents   │  ┌─────────────────────────────┐ │
│  ✅ business_analyst    │  │ Latest Result │ History     │ │
│  ⏳ quantitative_analyst│  └─────────────────────────────┘ │
│  ⏳ market_analyst      │                                   │
│  ...                    │  📄 Research Report               │
│                         │  [Report content]                 │
│  ⚙️ Settings            │                                   │
│  Max Iterations: [5]    │  🔍 Show ReAct Trace [toggle]    │
│  ☐ Auto-show Trace      │  [Trace content if shown]         │
│                         │                                   │
│  🔄 Reset System        │                                   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

### 1. **System Initialization**

**Location:** Sidebar

- Click **"🚀 Initialize System"** to start the ReAct orchestrator
- System checks environment variables
- Registers available specialist agents
- Shows agent status (✅ Active / ⏳ Planned)

**Status Indicators:**
- ✅ Green = Agent implemented and ready
- ⏳ Gray = Agent planned but not yet implemented
- ❌ Red = Error with agent initialization

### 2. **Query Interface**

**Location:** Main content area

**Example Queries:**

Click "📌 Example Queries" to see:

**Company Analysis:**
- "What are Apple's key competitive risks?"
- "Analyze Tesla's market position"
- "Compare Microsoft and Google"

**Financial Metrics:**
- "What is Apple's profit margin?"
- "Analyze Amazon's growth trajectory"
- "Compare Netflix and Disney's valuations"

**How to Use:**
1. Type your question in the text area
2. Click **"🔍 Analyze"** to start ReAct loop
3. Watch the spinner while agents work
4. View results automatically when complete

### 3. **Results Display**

#### **Metrics Bar**

Shows key performance indicators:

```
┌────────────┬────────────┬──────────────┬─────────────┐
│ Iterations │  Duration  │ Agents Called│ Time/Iter   │
│     3      │   45.2s    │      3       │    15.1s    │
└────────────┴────────────┴──────────────┴─────────────┘
```

- **Iterations:** Number of ReAct loop cycles
- **Duration:** Total time from start to finish
- **Agents Called:** How many specialist agents were invoked
- **Time/Iter:** Average time per iteration

#### **Research Report**

Formatted markdown output including:
- Executive Summary
- Detailed Analysis
- Risk Factors
- Conclusion
- Citations (if from Business Analyst)

#### **ReAct Trace**

Toggle to view the complete reasoning process:

```
=== Iteration 1 ===
Thought: Need qualitative risk data from 10-K filings
Action: call_specialist → business_analyst
Task: Extract competitive risk factors from Apple's 10-K
Observation: Extracted 5 competitive risks with citations...

=== Iteration 2 ===
Thought: Have risks, now need quantitative margin calculations
Action: call_specialist → quantitative_analyst
Task: Calculate Apple's net profit margin for last 3 years
Observation: [QUANTITATIVE ANALYST - PLACEHOLDER]...

=== Iteration 3 ===
Thought: Sufficient information gathered from available agents
Action: finish
Observation: Orchestration complete - proceeding to synthesis
```

### 4. **History Tab**

**Location:** Results area → History tab

- Stores all queries from current session
- Expandable cards for each past query
- Shows timestamp, duration, iterations
- Can toggle ReAct trace for each historical query
- Data persists until session reset or page refresh

### 5. **Settings (Sidebar)**

#### **Max Iterations Slider**

```
Max Iterations: [====●====] 5
                1        10
```

- **1-3:** Quick queries (simple questions)
- **4-5:** Balanced (recommended)
- **6-10:** Deep research (complex analysis)

**Impact:**
- Higher = More thorough but slower
- Lower = Faster but may miss details

#### **Auto-show ReAct Trace**

☐ Auto-show ReAct Trace

- **Unchecked (default):** Trace hidden, manual toggle
- **Checked:** Trace automatically displayed after each query

### 6. **Action Buttons**

#### **🔍 Analyze** (Primary button)
- Starts ReAct loop with your query
- Disabled if query is empty
- Shows spinner during execution

#### **🗑️ Clear History**
- Removes all queries from history tab
- Does not reset system or agents
- Confirms with page refresh

#### **💾 Download**
- Available when history exists
- Downloads all queries and reports as markdown
- Filename: `research_report_YYYYMMDD_HHMMSS.md`

#### **🔄 Reset System**
- Completely resets orchestrator
- Clears history
- Requires re-initialization
- Use if system becomes unresponsive

## 🎨 UI Components

### Status Boxes

**Success (Green):**
```
✅ System initialized successfully!
```

**Warning (Yellow):**
```
⚠️ EODHD_API_KEY not set (optional)
```

**Error (Red):**
```
❌ PERPLEXITY_API_KEY not set
```

**Info (Blue):**
```
ℹ️ No history yet. Run another query to see history.
```

### Progress Indicators

During query execution:

```
🧠 ReAct loop running...
   ⏳ [Spinner animation]
```

After completion:

```
✅ Analysis complete in 45.2s (3 iterations)
```

## 🔧 Troubleshooting

### Issue: "PERPLEXITY_API_KEY not set"

**Solution:**
```bash
export PERPLEXITY_API_KEY="your-key"
streamlit run app.py
```

### Issue: Business Analyst shows error

**Possible Causes:**
1. Ollama not running
2. Models not pulled
3. No documents in `/data` folder

**Solution:**
```bash
# Check Ollama
ollama list

# Pull models if missing
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Check data folder
ls -la ./data
```

### Issue: Slow response times

**Optimization:**
1. Reduce max iterations (try 3)
2. Use simpler queries
3. Check Ollama model performance
4. Ensure good internet connection (for Perplexity API)

### Issue: "Connection refused" error

**Solution:**
```bash
# Restart Ollama
pkill ollama
ollama serve
```

### Issue: UI becomes unresponsive

**Solution:**
1. Click **"🔄 Reset System"** in sidebar
2. Refresh browser page (Ctrl+R / Cmd+R)
3. Restart Streamlit:
   ```bash
   # Stop: Ctrl+C
   streamlit run app.py
   ```

## 💡 Tips & Best Practices

### 1. **Query Formulation**

**Good Queries:**
- "What are Apple's main competitive risks?" ✅
- "Calculate Tesla's profit margins over 3 years" ✅
- "Compare Amazon and Alibaba's market positioning" ✅

**Avoid:**
- "Tell me everything about Apple" ❌ (too broad)
- "??" ❌ (unclear intent)
- "Yes" ❌ (not a question)

### 2. **Monitoring ReAct Reasoning**

For learning/debugging:
1. Enable "Auto-show ReAct Trace"
2. Watch how the orchestrator thinks
3. See which agents it chooses and why
4. Understand iteration patterns

### 3. **Managing History**

- Download reports before clearing history
- History is session-based (lost on page refresh)
- Use Download button for important analyses

### 4. **Performance Optimization**

**For Quick Queries:**
- Set Max Iterations: 2-3
- Use specific, focused questions

**For Deep Analysis:**
- Set Max Iterations: 6-8
- Ask multi-faceted questions
- Expect longer wait times

### 5. **Understanding Agent Status**

Check sidebar before complex queries:
- If only Business Analyst is ✅, expect 10-K analysis only
- Quantitative queries will get placeholder responses
- Plan queries based on available agents

## 📊 Example Workflows

### Workflow 1: Simple Risk Analysis

1. **Initialize System** (sidebar)
2. **Enter Query:** "What are Apple's competitive risks?"
3. **Click Analyze**
4. **Wait ~30s** (typically 2-3 iterations)
5. **Review Report** with citations
6. **Toggle ReAct Trace** to see reasoning

**Expected Result:**
- Business Analyst extracts risks from 10-K
- Report with 4-5 risk factors
- Page citations included
- 2-3 iterations

### Workflow 2: Comparative Analysis

1. **Enter Query:** "Compare Microsoft and Google's competitive positioning"
2. **Set Max Iterations:** 5
3. **Click Analyze**
4. **Wait ~60s** (4-5 iterations)
5. **Review comparative report**
6. **Download** for future reference

**Expected Result:**
- Multiple agent calls (Business Analyst for both companies)
- Synthesis combining insights
- 4-5 iterations

### Workflow 3: Session Management

1. Run multiple queries (3-5)
2. Switch to **History tab**
3. Review past analyses
4. Click **Download** to save all reports
5. **Clear History** when done
6. **Reset System** for fresh start

## 🌐 Network Configuration

### Default Port

```
http://localhost:8501
```

### Custom Port

```bash
streamlit run app.py --server.port 8080
```

### External Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

**Access from network:**
```
http://YOUR_IP:8501
```

## 📱 Mobile Access

Streamlit is responsive and works on mobile browsers:

1. Find your computer's IP address
2. Ensure firewall allows port 8501
3. Run with `--server.address 0.0.0.0`
4. Access from mobile: `http://YOUR_IP:8501`

**Note:** Mobile experience is optimized but desktop is recommended for best UX.

## 🔒 Security Notes

- API keys are read from environment variables (not stored in UI)
- Session state is browser-local only
- No data is persisted to disk by default
- Download feature creates local files only
- Do not expose to public internet without authentication

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [ReAct Framework Guide](REACT_FRAMEWORK.md)
- [Agent Specifications](SPECIALIST_AGENTS.md)
- [Project README](README.md)

---

**Built with:** Streamlit + ReAct + Multi-Agent Orchestration  
**Last Updated:** February 9, 2026
