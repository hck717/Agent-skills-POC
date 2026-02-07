# Agent-skills-POC

# 1. 激活虛擬環境
/opt/homebrew/bin/python3.11 -m venv .venv

source .venv/bin/activate

# 2. 安裝依賴 (如果你修改過 requirements.txt)
pip install -r requirements.txt

# 3. 確保 Ollama 在後台運行 (如果沒開)
# (建議開一個新 Terminal 窗口運行這個)
ollama serve 

# Chat 模型 (負責思考)
ollama pull qwen2.5:7b

# Embedding 模型 (負責讀文件)
ollama pull nomic-embed-text

python main.py
