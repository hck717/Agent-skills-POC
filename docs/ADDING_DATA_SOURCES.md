# Adding More Data Sources

## 🎯 Purpose

This guide shows you how to add more company data (10-K filings, financial reports) to enable:
- **Comparative analysis** (e.g., "Compare Apple vs Google")
- **Multi-company research** (e.g., "Analyze big tech competitive landscape")
- **Cross-company citations** (References from multiple sources)

---

## 📁 Current Data Structure

```
Agent-skills-POC/
└── data/
    └── AAPL/                    ← Currently only Apple
        └── APPL 10-k Filings.pdf
```

---

## 🚀 How to Add More Companies

### **Step 1: Create Company Folders**

```bash
cd Agent-skills-POC/data

# Create folders for each company you want to add
mkdir GOOGL  # Google/Alphabet
mkdir MSFT   # Microsoft
mkdir NVDA   # Nvidia
mkdir TSLA   # Tesla
mkdir META   # Meta/Facebook
mkdir AMZN   # Amazon
```

### **Step 2: Download 10-K Filings**

#### **Option A: SEC EDGAR (Official, Free)**

1. Go to [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html)
2. Search for company name (e.g., "Alphabet Inc")
3. Click on the company
4. Filter by "10-K" form type
5. Click on the latest 10-K filing
6. Click "Open document" → Save as PDF
7. Place in appropriate folder:
   ```bash
   mv ~/Downloads/googl-10k-2024.pdf data/GOOGL/
   ```

#### **Option B: Company Investor Relations (Often Easier)**

| Company | Investor Relations URL |
|---------|------------------------|
| Google/Alphabet | [abc.xyz/investor](https://abc.xyz/investor/) |
| Microsoft | [microsoft.com/investor](https://www.microsoft.com/en-us/Investor) |
| Nvidia | [investor.nvidia.com](https://investor.nvidia.com/) |
| Tesla | [ir.tesla.com](https://ir.tesla.com/) |
| Meta | [investor.fb.com](https://investor.fb.com/) |
| Amazon | [ir.aboutamazon.com](https://ir.aboutamazon.com/) |

**Steps:**
1. Go to company's investor relations page
2. Navigate to "SEC Filings" or "Annual Reports"
3. Download latest 10-K (usually a PDF)
4. Place in company folder

---

### **Step 3: Supported File Formats**

The Business Analyst supports multiple formats:

```bash
data/GOOGL/
  ✓ GOOGL-10K-2024.pdf          # PDF (primary)
  ✓ earnings-report-Q4.docx     # Word documents
  ✓ risk-assessment.txt         # Text files
  ✓ strategic-plan.md           # Markdown files
```

**File naming doesn't matter** - the system identifies companies by **folder name** (ticker symbol).

---

### **Step 4: Ingest the Data**

#### **Via Streamlit UI (Recommended):**

1. Start Streamlit: `streamlit run app.py`
2. In sidebar, expand **"🔧 Data Management"**
3. Click **"🔄 Reingest All Data"**
4. Wait for processing (may take 1-2 minutes per company)
5. Click **"📊 Check Database Stats"** to verify

**Expected output:**
```
📈 Database Statistics:
AAPL: 156 chunks
GOOGL: 203 chunks
MSFT: 178 chunks
NVDA: 145 chunks
---
Total Chunks: 682
```

#### **Via Python (Alternative):**

```python
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent

agent = BusinessAnalystGraphAgent(
    data_path="./data",
    db_path="./storage/chroma_db"
)

# Ingest all documents from data/ folder
agent.ingest_data()

# Check what was ingested
stats = agent.get_database_stats()
print(stats)
```

---

## ✅ Verification

### **Check Console Output:**

```
📂 Scanning ./data...

📊 Processing AAPL...
   Folder: ./data/AAPL
   ✅ Loaded 1 PDF documents
   🔪 Splitting documents into chunks...
   🧮 Embedding 156 chunks...
   ✅ Indexed 156 chunks from 1 PDFs

📊 Processing GOOGL...
   Folder: ./data/GOOGL
   ✅ Loaded 1 PDF documents
   🔪 Splitting documents into chunks...
   🧮 Embedding 203 chunks...
   ✅ Indexed 203 chunks from 1 PDFs

============================================================
✅ INGESTION COMPLETE
   Total documents: 2
   Total chunks: 359
   Database: ./storage/chroma_db
============================================================
```

### **Test Queries:**

Once data is ingested, try these queries:

```
# Single company
Analyze Apple's competitive positioning and risk factors

# Comparative (requires both AAPL and GOOGL)
Compare Apple and Google's business models and risk profiles

# Multi-company (requires AAPL, MSFT, GOOGL)
Analyze big tech competitive landscape focusing on AI capabilities
```

---

## 📚 Expected Citation Improvements

### **Before (Only AAPL):**
```markdown
## References
[1] APPL 10-k Filings.pdf - Page 23
[2] APPL 10-k Filings.pdf - Page 45
[3] APPL 10-k Filings.pdf - Page 67
```

### **After (AAPL + GOOGL + MSFT):**
```markdown
## Competitive Analysis

Apple's iPhone ecosystem generates 52% of revenue [1], while Google's 
cloud services grew 34% YoY [2]. Microsoft's enterprise focus provides 
stable recurring revenue through Azure and Office 365 [3].

Apple faces supply chain concentration in China [4], Google navigates 
regulatory scrutiny over ad monopoly [5], and Microsoft manages 
cybersecurity risks in cloud infrastructure [6].

## References
[1] APPL 10-k Filings.pdf - Page 23
[2] GOOGL 10-K 2024.pdf - Page 45
[3] MSFT Annual Report 2024.pdf - Page 67
[4] APPL 10-k Filings.pdf - Page 89
[5] GOOGL 10-K 2024.pdf - Page 123
[6] MSFT Annual Report 2024.pdf - Page 145
```

✅ **Multi-source citations enable true comparative analysis!**

---

## 🛠️ Troubleshooting

### **Problem: "No documents found for GOOGL"**

**Cause:** Folder exists but is empty

**Fix:**
```bash
ls -la data/GOOGL/  # Check if files exist
# If empty, download 10-K and place in folder
```

---

### **Problem: "Found 0 chunks for GOOGL"**

**Cause:** File format not supported or corrupted

**Fix:**
```bash
# Check file type
file data/GOOGL/*

# Should show:
# GOOGL-10K.pdf: PDF document, version 1.4

# If shows "HTML" or "ASCII text", re-download as proper PDF
```

---

### **Problem: Ingestion takes forever**

**Cause:** Large PDFs (500+ pages) or many files

**Expected Times:**
- 100-page PDF: ~30 seconds
- 200-page PDF: ~1 minute
- 500-page PDF: ~3 minutes

**Tip:** Watch console for progress:
```
🔪 Splitting documents into chunks...  ← May take 10-30s for large PDFs
🧮 Embedding 203 chunks...            ← May take 20-60s
```

---

### **Problem: Citations still only from one company**

**Cause:** Query doesn't mention other companies

**Fix:** Be explicit in your query:
```
# Bad (will only search AAPL):
"Analyze competitive risks"

# Good (will search both AAPL and GOOGL):
"Compare Apple and Google's competitive risks"

# Good (will search AAPL, GOOGL, MSFT):
"Analyze Apple, Google, and Microsoft's AI strategies"
```

The system uses **ticker identification** from your query to determine which companies to search.

---

## 💡 Pro Tips

### **1. Organize by Year (Optional)**

```bash
data/
  └── AAPL/
      ├── AAPL-10K-2024.pdf
      ├── AAPL-10K-2023.pdf
      └── AAPL-10Q-Q4-2024.pdf
```

The system will index **all files** in each folder.

### **2. Mix Document Types**

```bash
data/GOOGL/
  ├── GOOGL-10K-2024.pdf          # Official SEC filing
  ├── earnings-call-Q4.docx      # Earnings transcript
  ├── strategic-initiatives.md   # Internal notes
  └── risk-assessment.txt        # Analysis notes
```

All formats contribute to the knowledge base!

### **3. Reset if Something Goes Wrong**

If ingestion fails or produces bad results:

1. In Streamlit sidebar: **"🗑️ Reset Database"**
2. Check **"I understand this will delete all data"**
3. Click **"🗑️ Reset Database"**
4. Re-run **"🔄 Reingest All Data"**

This clears the vector database and starts fresh.

---

## 🎯 Quick Start: Add Google Data

```bash
# 1. Create folder
mkdir -p data/GOOGL

# 2. Download Google's 10-K
# Visit: https://abc.xyz/investor/
# Download latest 10-K to data/GOOGL/

# 3. Restart Streamlit
streamlit run app.py

# 4. In UI: Click "Reingest All Data"

# 5. Test query:
"Compare Apple and Google's business models and competitive advantages"
```

**Expected Result:**
```markdown
## Comparative Analysis

Apple's ecosystem integration [1] contrasts with Google's
platform-based approach [2]...

## References
[1] APPL 10-k Filings.pdf - Page 23
[2] GOOGL-10K-2024.pdf - Page 45  ← NEW!
```

---

## 🚀 Recommended Data Sources

| Priority | Companies | Rationale |
|----------|-----------|----------|
| **High** | GOOGL, MSFT | Enable "Big Tech" comparative analysis |
| **Medium** | NVDA, META | GPU/AI trends, social media landscape |
| **Low** | TSLA, AMZN | EV/logistics, e-commerce benchmarks |

**Suggested First Addition:** **GOOGL** (Google/Alphabet)
- Direct competitor to Apple
- Enables iOS vs Android comparisons
- Strong AI positioning for comparative analysis

---

## 📊 Impact on Report Quality

| Metric | Before (AAPL only) | After (AAPL + GOOGL + MSFT) |
|--------|-------------------|-----------------------------|
| **Citation Sources** | 1 company | 3 companies |
| **Comparative Depth** | Single perspective | Multi-company benchmarks |
| **Reference Diversity** | All from 1 doc | 3-6 different docs |
| **Query Capabilities** | Single-company only | Comparative analysis ✅ |

---

## ❓ FAQ

**Q: Can I add non-tech companies?**  
A: Yes! Add any ticker. Just create a folder (e.g., `data/WMT` for Walmart) and add documents.

**Q: Do I need to modify code?**  
A: No code changes needed. The system auto-detects all folders in `data/`.

**Q: What if I have 100+ page PDFs?**  
A: Works fine. The chunking system handles large documents automatically.

**Q: Can I add quarterly reports (10-Q)?**  
A: Yes! Add 10-Q, 8-K, earnings transcripts - any text document works.

**Q: How much disk space needed?**  
A: Roughly 2-3x the PDF size for vector embeddings. A 5MB 10-K uses ~10-15MB total.

---

## ✅ Summary

1. **Create folders** in `data/` using ticker symbols (GOOGL, MSFT, etc.)
2. **Download 10-K PDFs** from SEC EDGAR or company investor relations
3. **Reingest data** via Streamlit UI
4. **Test queries** mentioning multiple companies
5. **Enjoy comparative analysis** with multi-source citations!

---

**Need help?** Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) or open an issue on GitHub.
