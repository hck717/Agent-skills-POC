# Troubleshooting Guide

## API Configuration Issues

### Fixed: Perplexity API 400 Error

**Symptoms:**
```
Error calling Perplexity API: 400 Client Error: Bad Request
```

**Root Causes:**
1. Invalid or missing API key
2. Incorrect model name
3. Malformed API request

**Solutions Applied:**

#### 1. API Key Input in UI

You can now enter your API key directly in the Streamlit interface:

1. Open `streamlit run app.py`
2. In sidebar, click **"⚙️ Configure API Keys"**
3. Enter your Perplexity API key
4. Click **"🚀 Initialize System"**

#### 2. Changed Perplexity Model

**Before (Caused 400 error):**
```python
model = "llama-3.1-sonar-large-128k-online"
```

**After (Fixed):**
```python
model = "llama-3.1-sonar-small-128k-online"
```

The `sonar-small` model is more stable and commonly available.

#### 3. Better Error Handling

The system now provides helpful error messages:
- **401** → "Invalid API key"
- **400** → "Check API key and model name"
- **429** → "Rate limit exceeded"

---

## Getting Your API Keys

### Perplexity API (Required)

1. Go to [Perplexity AI Settings](https://www.perplexity.ai/settings/api)
2. Sign in or create account
3. Generate API key
4. Copy the key (starts with `pplx-`)
5. Enter in UI or set environment variable:
   ```bash
   export PERPLEXITY_API_KEY="pplx-your-key-here"
   ```

### EODHD API (Optional)

1. Go to [EODHD Registration](https://eodhd.com/register)
2. Sign up for free tier
3. Get API key from dashboard
4. Enter in UI or set environment variable:
   ```bash
   export EODHD_API_KEY="your-key-here"
   ```

---

## Common Error Messages

### "PERPLEXITY_API_KEY not found"

**Solution:**
- Enter API key in UI sidebar
- Or set environment variable before running:
  ```bash
  export PERPLEXITY_API_KEY="your-key"
  streamlit run app.py
  ```

### "Invalid API key"

**Checks:**
1. Key starts with `pplx-`
2. No extra spaces
3. Key is active (check Perplexity dashboard)
4. Not using old/revoked key

### "Rate limit exceeded"

**Solutions:**
- Wait 60 seconds before retry
- Check your API plan limits
- Consider upgrading plan if needed

### "Business Analyst initialization error"

**Possible causes:**
1. Ollama not running
2. Models not downloaded
3. No documents in `/data` folder

**Solutions:**
```bash
# Check Ollama is running
ollama list

# If not running, start it
ollama serve

# Download required models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Check data folder
ls ./data/
# Should contain PDF files organized by ticker
```

---

## Environment Setup Checklist

### Before Running Streamlit

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Perplexity API key obtained
- [ ] Ollama installed and running
- [ ] Models downloaded (qwen2.5:7b, nomic-embed-text)
- [ ] At least one PDF in `./data/` folder (optional)

### Verify Setup

```bash
# Check Python version
python --version
# Should show 3.11 or higher

# Check virtual environment
which python
# Should point to .venv

# Check Ollama
ollama list
# Should show qwen2.5:7b and nomic-embed-text

# Test imports
python -c "from orchestrator_react import ReActOrchestrator; print('✅ Imports work')"
```

---

## Debugging Steps

### 1. Enable Debug Mode

In Streamlit, errors are caught in expandable "Debug Information" sections.

### 2. Check API Call Logs

Look for these patterns in terminal:
```
🧠 [THOUGHT 1] Reasoning about next action...
   ⚠️ Error calling Perplexity API: ...
```

### 3. Test API Key Manually

```bash
curl -X POST https://api.perplexity.ai/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-sonar-small-128k-online",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

If this fails, the API key is invalid.

### 4. Check File Permissions

Ensure the app can read/write to necessary directories:
```bash
ls -la ./data/
ls -la ./storage/chroma_db/
```

---

## Performance Issues

### Slow Response Times

**Causes:**
- Large documents being processed
- Ollama running on CPU instead of GPU
- Too many iterations

**Solutions:**
1. Reduce max iterations (sidebar slider)
2. Use GPU-accelerated Ollama if available
3. Optimize document chunking

### High Memory Usage

**Solutions:**
1. Close other applications
2. Reduce batch size in Business Analyst
3. Use smaller Ollama models
4. Clear ChromaDB cache periodically

---

## Still Having Issues?

### Check Documentation

- [README.md](../README.md) - Project overview
- [docs/UI_GUIDE.md](UI_GUIDE.md) - UI walkthrough
- [docs/REACT_FRAMEWORK.md](REACT_FRAMEWORK.md) - ReAct architecture

### GitHub Issues

Open an issue on GitHub with:
1. Error message (full traceback)
2. Your setup (OS, Python version, etc.)
3. Steps to reproduce
4. What you've already tried

### Community Support

Check existing issues for similar problems:
[GitHub Issues](https://github.com/hck717/Agent-skills-POC/issues)

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|----------|
| API key error | Enter key in UI sidebar |
| 400 Bad Request | Check API key format (starts with `pplx-`) |
| Rate limit | Wait 60s, reduce iterations |
| Ollama error | `ollama serve` in separate terminal |
| Import error | `pip install -r requirements.txt` |
| No documents | Add PDFs to `./data/TICKER/` |
| Slow performance | Reduce max iterations to 3 |
| Division by zero | Update to latest version (fixed) |

---

**Last Updated:** February 9, 2026
