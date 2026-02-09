# Changelog

All notable changes to this project will be documented in this file.

---

## [2.0.0] - 2026-02-09

### 🎉 Major Restructure

**Repository completely reorganized for clarity and maintainability.**

#### 📁 Structure Changes

**Created Folders:**
- `docs/` - All documentation files
- `orchestrator/` - Orchestration logic modules

**Moved Files:**
- `REACT_FRAMEWORK.md` → `docs/REACT_FRAMEWORK.md`
- `SPECIALIST_AGENTS.md` → `docs/SPECIALIST_AGENTS.md`
- `UI_GUIDE.md` → `docs/UI_GUIDE.md`
- `ORCHESTRATOR_README.md` → `docs/ORCHESTRATOR.md`

**Created Import Structure:**
- `orchestrator/__init__.py` - Module initialization
- `orchestrator/react.py` - ReAct framework imports
- `orchestrator/legacy.py` - Legacy planner imports

**Root Now Contains Only:**
- Essential entry points (`app.py`, `main.py`, `main_orchestrated.py`)
- Core implementations (`orchestrator_react.py`, `orchestrator.py`)
- Configuration (`requirements.txt`, `.gitignore`)
- Main documentation (`README.md`, `CHANGELOG.md`)

#### 📝 Documentation Updates

**README.md** - Complete rewrite:
- Clear 3-option quick start (Streamlit/CLI ReAct/CLI Single)
- Visual architecture diagram
- Comprehensive project structure tree
- ReAct vs Traditional comparison table
- Complete tech stack listing
- Performance metrics table
- FAQ section
- Development guide
- Badge indicators

**New .gitignore:**
- Python artifacts
- Virtual environments
- IDE files
- ChromaDB storage
- Data files
- API keys
- OS-specific files

#### ✨ New Features

- [x] **Streamlit UI** (`app.py`)
  - Point-and-click interface
  - Real-time metrics dashboard
  - ReAct trace visualization
  - Session history management
  - Report download functionality
  - Adjustable settings

- [x] **ReAct Framework** (`orchestrator_react.py`)
  - Iterative Think → Act → Observe loop
  - Dynamic agent selection
  - Self-correction capabilities
  - Early stopping optimization
  - Complete reasoning trace

- [x] **Documentation Folder** (`docs/`)
  - Centralized documentation
  - Easy navigation
  - Clear separation from code

#### 🔄 Breaking Changes

**Import Paths:**
```python
# Old (still works)
from orchestrator_react import ReActOrchestrator

# New (recommended)
from orchestrator.react import ReActOrchestrator
```

**Documentation Locations:**
```
# Old
REACT_FRAMEWORK.md (root)
SPECIALIST_AGENTS.md (root)

# New
docs/REACT_FRAMEWORK.md
docs/SPECIALIST_AGENTS.md
```

---

## [1.0.0] - 2026-02-08

### Initial Implementation

- [x] Business Analyst agent with LangGraph
- [x] RAG with ChromaDB + BERT reranking
- [x] Multi-agent orchestration framework
- [x] CLI interface
- [x] Persona-based analysis
- [x] Document ingestion pipeline

---

## Roadmap

### v2.1.0 (Planned)
- [ ] Quantitative Analyst implementation
- [ ] Market Analyst with real-time data
- [ ] Parallel agent execution

### v2.2.0 (Planned)
- [ ] Industry Analyst with web search
- [ ] ESG Analyst
- [ ] Macro Analyst

### v3.0.0 (Future)
- [ ] Multi-turn memory system
- [ ] Cost tracking and optimization
- [ ] Agent performance analytics
- [ ] Chart visualization in UI
- [ ] Authentication and multi-user support

---

**Navigation:**
- [README](README.md) - Main documentation
- [docs/](docs/) - Detailed guides
- [GitHub Releases](https://github.com/hck717/Agent-skills-POC/releases)
