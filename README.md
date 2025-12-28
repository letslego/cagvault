# CagVault

A **Cache-Augmented Generation (CAG)** application for private, local document chat using large language models with intelligent document parsing, LanceDB-backed persistent storage, and credit agreement analysis capabilities.

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install prerequisites
brew install ollama
brew services start ollama

# 2. Clone and setup
git clone https://github.com/letslego/cagvault.git
cd cagvault
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .

# 3. Download LLM model (choose one based on your RAM)
# Default (16GB RAM): Qwen3-14B
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL
# Or for best quality (64GB+ RAM): DeepSeek V3
# ollama pull deepseek-ai/DeepSeek-V3
# Or lightweight (8GB RAM): Llama 3.1 8B
# ollama pull llama3.1:8b

# 4. Start the app
streamlit run app.py
# Open http://localhost:8501 in your browser

# 5. Upload a PDF and start chatting!
```

**First-Time Tips:**
- Upload a credit agreement PDF to see section analysis in action
- Try "💡 Suggested Questions" after parsing completes
- Explore the "Sections" tab to see hierarchical structure
- Use "Agentic Search" for intelligent query understanding

## 🎯 What's New (December 2025)

**🤖 Agentic RAG System:**
- 🧠 **Multi-Step Reasoning**: Agent understands intent, selects strategy, validates answers
- 🎯 **5 Retrieval Strategies**: Semantic, Keyword, Hybrid, Agentic, Entity-based (auto-selected)
- ✅ **Self-Reflection**: Optional answer validation with confidence scoring
- 📊 **Full Transparency**: Complete reasoning traces showing agent's thought process
- 🎓 **Smart Strategy Selection**: Automatically chooses best approach based on query type
- 🔧 **Claude Agent SDK Integration**: 6 specialized MCP tools built with Agent SDK:
  - 🌐 **Web Search**: Fetch current data from external sources (@tool decorator)
  - 🏷️ **Entity Extraction**: Extract dates, amounts, names, organizations (NER-based)
  - 📊 **Section Ranking**: Prioritize important sections using credit analyst criteria
  - 🔗 **Cross-Document Relationships**: Find references, amendments, guarantees
  - 🔍 **Fact Verification**: Validate claims against web sources
  - 💡 **Follow-Up Suggestions**: Intelligent next-question recommendations

**Storage Architecture Upgrade:**
- 🗄️ **LanceDB Embedded Database**: Replaced Redis with LanceDB for all persistent storage
- ⚡ **In-Process Caching**: 3-second TTL DataFrame cache for sub-millisecond reads
- 🔍 **Full-Text Search**: Built-in FTS indexes on content, titles, and questions
- 📦 **Zero External Dependencies**: No separate database server required - all data in `./lancedb`
- 🔄 **Redis Migration Tool**: One-time utility to import existing Redis data
- 🔒 **ACID Compliance**: Reliable transactions with automatic cache invalidation

**Enhanced PDF Intelligence:**
- 🔬 **LLM-Powered Section Analysis**: Parallel processing with credit analyst classification and importance scoring
- 📊 **Smart Section Extraction**: Hierarchical document structure with page-accurate tracking
- 🔍 **Multi-Modal Search**: Keyword, semantic, and agentic (Claude-powered) search within documents
- 🏷️ **Named Entity Recognition**: Extract and index parties, dates, amounts, and legal terms
- 📌 **Referenced Section Display**: Automatically expand cited sections in chat responses

**Intelligent Caching System:**
- 💾 **Q&A Cache**: LanceDB-backed answer caching per document with persistent storage
- 📚 **Question Library**: Track popular questions by category with autocomplete suggestions
- ⚡ **KV-Cache Optimization**: 10-40x faster multi-turn conversations
- 📈 **Cache Analytics**: Real-time statistics and per-document cache management

**Credit Agreement Features:**
- 📋 **Document Classification**: Automatic detection of covenants, defaults, and key provisions
- 🎯 **Section Importance Scoring**: AI-driven relevance analysis for credit analysts
- 🔗 **Cross-Reference Detection**: Track dependencies between sections
- 📄 **Page-Accurate Citations**: Precise page ranges for every section

## What is Cache-Augmented Generation (CAG)?

Based on the paper [*"Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks"*](https://arxiv.org/abs/2412.15605v1) (WWW '25), CAG is an alternative paradigm to traditional Retrieval-Augmented Generation (RAG) that leverages the extended context capabilities of modern LLMs.

### CAG vs RAG: Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TRADITIONAL RAG WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Query                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌──────────────────┐         ┌──────────────────┐                         │
│  │  Retriever       │◄────────┤  Search Index    │  ⏱️  LATENCY             │
│  │  (BM25/Dense)    │         │  (Large DB)      │                         │
│  └────────┬─────────┘         └──────────────────┘                         │
│           │                                                                  │
│           │ Retrieved Documents                                             │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ Generator (LLM)  │  ⚠️  Risk of:                                         │
│  │                  │      • Missing relevant docs                          │
│  │  (Generate Ans)  │      • Ranking errors                                │
│  └────────┬─────────┘      • Search failures                               │
│           │                                                                  │
│           ▼                                                                  │
│      Answer                                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│               CACHE-AUGMENTED GENERATION (CAG) WORKFLOW                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─── SETUP PHASE (One-time) ─────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  All Documents                                                     │   │
│  │      │                                                             │   │
│  │      ▼                                                             │   │
│  │  ┌──────────────────┐                                             │   │
│  │  │  LLM Processor   │  Populate LanceDB Cache                     │   │
│  │  │  (Batch Process) │  (Sections + Q&A store)                     │   │
│  │  └────────┬─────────┘                                             │   │
│  │           │                                                        │   │
│  │           ▼                                                        │   │
│  │  ┌──────────────────────┐                                         │   │
│  │  │  Cached LanceDB Store│  💾  Embedded on Disk                  │   │
│  │  │  (Ready to use)      │                                         │   │
│  │  └──────────┬───────────┘                                         │   │
│  │             │                                                     │   │
│  └─────────────┼─────────────────────────────────────────────────────┘   │
│                │                                                           │
│  ┌─── INFERENCE PHASE (Fast) ────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  User Query        LanceDB Cache                                 │   │
│  │      │                  │                                        │   │
│  │      └──────────┬───────┘                                        │   │
│  │                 ▼                                                │   │
│  │        ┌──────────────────────┐                                 │   │
│  │        │  LLM + LanceDB Hits  │  ✨ LOCAL RETRIEVAL!            │   │
│  │        │  (Context + cache)   │  ✨ LOW LATENCY!               │   │
│  │        │                      │  ✨ GUARANTEED CONTEXT!        │   │
│  │        └──────────┬───────────┘                                 │   │
│  │                   │                                              │   │
│  │                   ▼                                              │   │
│  │              Answer (Instant)                                    │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─── MULTI-TURN OPTIMIZATION ───────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  For next query: Simply truncate and reuse cached knowledge     │   │
│  │  (No need to reprocess documents)                              │   │
│  │                                                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Workflow Phases

**1. Preload Phase** (One-time setup)
- All relevant documents are loaded into the LLM's extended context window
- The model processes the entire knowledge base at once

**2. Cache Phase** (Offline computation)
- The model's key-value (KV) cache is precomputed and stored
- This cache encapsulates the inference state of the LLM with all knowledge
- No additional computation needed for each query

**3. Inference Phase** (Fast queries)
- User queries are appended to the preloaded context
- The model uses the cached parameters to generate responses directly
- **No retrieval step needed** → Instant answers

**4. Reset Phase** (Multi-turn optimization)
- For new queries, the cache is efficiently truncated and reused
- The preloaded knowledge remains available without reprocessing

### Advantages

- ✅ **Zero Retrieval Latency**: No real-time document search
- ✅ **Unified Context**: Holistic understanding of all documents
- ✅ **Simplified Architecture**: Single model, no retriever integration
- ✅ **Eliminates Retrieval Errors**: All relevant information is guaranteed to be available
- ✅ **Perfect for Constrained Knowledge Bases**: Ideal when all documents fit in context window

## 🏗️ Architecture Overview

CagVault now runs as a local agentic stack that combines Streamlit UI, Claude Agent SDK tools, and LanceDB-backed storage.

```
┌────────────────────────────────────────────────────────────────────┐
│                              Browser                               │
│                   Streamlit UI (app.py)                            │
│  - Chat with reasoning trace and skill tags                        │
│  - Upload/parse PDFs and manage caches                             │
│  - Question library + sections/entities explorer                   │
└───────────────┬────────────────────────────────────────────────────┘
                │ questions, uploads, actions
┌───────────────▼────────────────────────────────────────────────────┐
│                           Agent Brain                              │
│  Router: question classifier + skill inference                     │
│  Planner: chooses cached answer, retrieval, or tool use            │
│  Reasoner: Claude/Ollama models with reflection                    │
│  Tools (Claude Agent SDK via MCP):                                 │
│    • web_search • entity_extractor • section_ranker                │
│    • cross_doc_links • fact_verifier • followup_suggester          │
│  Skills: PDF parser, TOC/NER search, credit analyst prompts,       │
│          knowledge-base skill registry                             │
│  Caches: Q&A cache (LanceDB), question library (LanceDB),          │
│          in-memory DataFrame cache                                 │
└───────────────┬────────────────────────────────────────────────────┘
                │ retrieval + storage calls
┌───────────────▼────────────────────────────────────────────────────┐
│                      Storage and Engines                           │
│  LanceDB (embedded): doc_sections, qa_cache, question_library      │
│  Search: full-text, semantic, agentic rerank, entity filters       │
│  Runtimes: Ollama models, CAG MCP server hosting the tools         │
└────────────────────────────────────────────────────────────────────┘
```

**Key Flows:**
- **Upload/Parse → LanceDB**: PDFs run through Docling + LLM section analysis, saved to `doc_sections` with entities and TOC metadata.
- **Ask → Router → Cache** (default mode): Questions first check LanceDB Q&A cache/question library before invoking the LLM.
- **Retrieval/Tools**: When needed, the agent retrieves sections from LanceDB or calls MCP tools (web, entity, ranking, cross-doc, verification, follow-ups).
- **Answering**: Responses stream with reasoning trace, cited sections, and the skills/tools used for transparency.
- **Persistence**: All storage is local (LanceDB + optional caches); no cloud services are required.

**Execution Modes:**
- **Default (LanceDB Chat)**: Uses LanceDB retrieval plus Q&A cache and question library for fast local answers. No MCP tools or multi-step agent planning are invoked.
- **Agentic RAG Mode (toggle in UI)**: Adds planning, strategy selection, and MCP tools (web search, entities, ranking, cross-doc, fact check, follow-ups). This path currently bypasses the LanceDB Q&A cache for answers.

**Knowledge Base Skills:**
- Skills live locally in `knowledge-base/` and are inferred by lightweight keyword heuristics. They are rendered with each answer for transparency and kept private on-disk (see `.gitignore`).

## Core Features

### 🔒 Privacy & Security
- **Fully Local & Private**: No API keys, cloud services, or internet required (except Redis optional)
- **Document Control**: All processing happens on your machine
- **Optional Redis**: Can run fully in-memory without Redis for maximum privacy

### 📄 Intelligent Document Processing
- **Enhanced PDF Parsing**: Using Docling with LLM-powered section analysis
- **Multi-Format Support**: PDF, TXT, MD files and web URLs
- **Hierarchical Structure**: Automatic detection of sections, subsections, and tables
- **Named Entity Recognition**: Extract parties, dates, monetary amounts, and legal terms
- **Page-Accurate Tracking**: Precise page ranges for every section

### 🔍 Advanced Search Capabilities
- **Keyword Search**: Fast full-text search across all sections
- **Semantic Search**: AI-powered similarity matching
- **Agentic Search**: Claude-driven intelligent query understanding with reasoning
- **Entity Filtering**: Search by PARTY, DATE, MONEY, AGREEMENT, or PERCENTAGE

### 💾 Intelligent Caching System
- **Q&A Cache**: Redis-backed answer caching with automatic deduplication
- **Question Library**: Track popular questions organized by 15+ categories
- **KV-Cache Optimization**: 10-40x faster multi-turn conversations
- **Cache Analytics**: Real-time statistics and granular cache management
- **Document-Specific Caching**: Per-document cache with TTL management

### 💬 Enhanced Chat Experience
- **Streaming Responses**: Real-time generation with thinking process visibility
- **Referenced Sections**: Auto-expand cited sections in answers
- **Suggested Questions**: Category-based question recommendations
- **Autocomplete Search**: Type-ahead suggestions from question library
- **Multi-Document Context**: Chat across multiple documents simultaneously

### 🎯 Credit Agreement Analysis
- **Section Classification**: Automatic identification of COVENANTS, DEFAULTS, DEFINITIONS, etc.
- **Importance Scoring**: AI-driven relevance analysis for credit analysts
- **Cross-Reference Tracking**: Detect dependencies between sections
- **Covenant Analysis**: Specialized understanding of debt agreements and financial covenants

### 🧠 Extended Context & Performance
- **Large Context Windows**: Leverages Qwen3-14B's 8K+ token capacity
- **Concurrent Request Handling**: 4 parallel LLM workers for simultaneous requests
- **Parallel Processing**: Concurrent LLM calls for faster document analysis (4 workers)
- **Smart Page Estimation**: Word-based calculation for instant section mapping
- **Memory Management**: In-memory section store with LanceDB persistence
- **Connection Pooling**: Optimized Ollama connections with timeout management

## Prerequisites

### Required
- macOS (or Linux/Windows with appropriate package managers)
- Python 3.12.x or 3.14.x
- Homebrew (for macOS)
- At least 10GB free disk space (for the LLM model)
- 16GB RAM recommended (8GB minimum for 7B models)

### Data Storage
- **LanceDB** (included with dependencies)
  - Embedded vector database for persistent storage
  - No separate installation or server required
  - Automatically stores: Q&A cache, question library, parsed document sections
  - Database location: `./lancedb` directory
  - Migration tool available for existing Redis users (see below)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/letslego/cagvault.git
cd cagvault
```

### 2. Set Up Python Environment

Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### 3. Install Dependencies

Install all required Python packages:

```bash
pip install -e .
```

This will install:
- `streamlit` - Web UI framework
- `langchain-core`, `langchain-ollama`, `langchain-groq`, `langchain-community` - LLM orchestration
- `docling` - Document conversion library
- Other dependencies (see `pyproject.toml`)

### 4. Install and Start Ollama

Ollama is a local LLM inference server that runs models entirely on your machine.

#### macOS Installation:

```bash
brew install ollama
brew services start ollama
```

Verify Ollama is running:

```bash
ollama list
```

#### Linux Installation:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

#### Windows Installation:

Download and run the installer from [ollama.com/download](https://ollama.com/download)

### 5. Download LLM Models

CagVault supports multiple high-performance models optimized for RAG and document understanding. Choose based on your hardware and performance needs.

#### Quick Start: Download Essential Models (Recommended)

Download 3 essential models covering all use cases (~30GB):

```bash
./download_essential_models.sh
```

This installs:
- **Qwen3-14B** (Default) - Best balance of quality and speed
- **Llama 3.1 8B** (Lightweight) - Fast responses, low memory
- **Phi-4** (Efficient) - Microsoft's optimized model

#### Download All Models

To download all 10 supported models (~200GB):

```bash
./download_models.sh
```

⚠️ **Warning**: This downloads 200GB+ and takes 2-4 hours

#### Manual Model Selection

Alternatively, download individual models:

#### Recommended Models for RAG/Document Analysis

**DeepSeek V3 (Recommended for Best Quality)** - 685B parameters, state-of-the-art reasoning:
```bash
ollama pull deepseek-ai/DeepSeek-V3
```
*Requires: 64GB+ RAM, Apple Silicon M3 Max or similar*

**Qwen3-14B (Default)** - Excellent balance of quality and speed:
```bash
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL
```
*Requires: 16GB+ RAM*

**DeepSeek R1** - Advanced reasoning for complex credit agreement queries:
```bash
ollama pull deepseek-ai/DeepSeek-R1
```
*Requires: 32GB+ RAM*

**Mistral Large** - Excellent long-context performance:
```bash
ollama pull mistral-large-latest
```
*Requires: 32GB+ RAM*

**Command R+** - Cohere's RAG-optimized model:
```bash
ollama pull command-r-plus:latest
```
*Requires: 32GB+ RAM*

#### Lightweight Models (8GB-16GB RAM)

**Phi-4** - Microsoft's efficient 14B model:
```bash
ollama pull phi4:latest
```

**Llama 3.1 8B** - Fast and lightweight:
```bash
ollama pull llama3.1:8b
```

**Mistral Small** - Quick responses for simpler queries:
```bash
ollama pull mistral-small-latest
```

#### High-End Models (32GB+ RAM)

**Llama 3.3 70B** - Strong reasoning:
```bash
ollama pull llama3.3:70b
```

**Gemma 2 27B** - Google's reasoning model:
```bash
ollama pull gemma2:27b
```

#### Switching Models

**Option 1: Use the UI (Recommended)**

1. Start the app: `streamlit run app.py`
2. Open the sidebar
3. Expand "🤖 Model Settings"
4. Select your preferred model from the dropdown
5. Click "🔄 Restart App" to apply

The UI shows RAM requirements and speed for each model to help you choose.

**Option 2: Edit Config File**

Edit `config.py` directly:

```python
class Config:
    MODEL = DEEPSEEK_V3  # Change from QWEN_3 to any model above
    OLLAMA_CONTEXT_WINDOW = 8192
```

Available model constants:
- `QWEN_3` (default) - Qwen3-14B
- `DEEPSEEK_V3` - DeepSeek V3 (best quality)
- `DEEPSEEK_R1` - DeepSeek R1 (advanced reasoning)
- `MISTRAL_LARGE` - Mistral Large
- `MISTRAL_SMALL` - Mistral Small
- `LLAMA_3_3_70B` - Llama 3.3 70B
- `LLAMA_3_1_8B` - Llama 3.1 8B
- `PHI_4` - Phi-4
- `GEMMA_2_27B` - Gemma 2 27B
- `COMMAND_R_PLUS` - Command R+

#### Browsing Available Models

```bash
# List installed models
ollama list

# Search for models on Ollama library
ollama search deepseek
ollama search mistral
ollama search llama3

# Pull any model
ollama pull <model-name>
```

### 6. (Optional) Migrate from Redis

If you have existing data in Redis, you can migrate it to LanceDB:

```python
# In Python console or script
from lancedb_cache import get_lancedb_store
import redis

# Connect to your Redis instance
redis_client = redis.from_url("redis://localhost:6379/0")

# Migrate all data (documents, Q&A cache, question library)
store = get_lancedb_store()
store.migrate_from_redis(redis_client)

print("Migration complete! Redis data imported to LanceDB.")
```

**Note**: After migration, you can optionally remove Redis. LanceDB is now the default persistent storage and requires no separate server.

### 7. Verify Installation

Check that everything is installed correctly:

```bash
# Python environment
python --version  # Should show 3.12.x or 3.14.x

# Ollama service
ollama list  # Should show your downloaded models

# Python packages
pip list | grep -E "(streamlit|langchain|docling|lancedb)"
```

## Running the Application

### Start the Streamlit App

With your virtual environment activated:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8504`

### Using the Application

#### 1. Upload Documents

**Via File Upload:**
- Click the file uploader in the sidebar
- Select PDF, TXT, or MD files
- Watch the enhanced parsing process with section extraction
- View parsing statistics: pages, sections, entities found

**Via URL:**
- Paste a web URL in the text input
- Click "Add Web Page" to scrape and convert to text

**From LanceDB:**
- Click "🗄️ Documents in LanceDB" expander
- Select a previously parsed document
- Click "Load for chat" to restore from persistent storage

#### 2. Explore Document Structure

**Sections Tab:**
- Browse hierarchical document structure
- View page ranges, word counts, and table indicators
- Click to expand section content
- See coverage statistics (page distribution)

**Search Tab:**
- **Agentic Search**: Claude-powered intelligent search with reasoning
- **Keyword Search**: Fast full-text search with match counts
- **Semantic Search**: AI similarity matching with relevance scores

**Entities Tab:**
- Filter by type: MONEY, DATE, PARTY, AGREEMENT, PERCENTAGE
- Click entities to see source sections
- Track key document information

#### 3. Ask Questions

**Direct Input:**
- Type your question in the chat input at the bottom
- Press Enter to submit

**Suggested Questions:**
- Click "💡 Suggested Questions" to see popular queries
- Click any suggestion to instantly ask it
- Questions are categorized: Definitions, Parties, Financial, etc.

**Browse by Category:**
- Click "📚 Browse by Category"
- Explore questions organized by 15+ categories
- View document-specific or global questions

#### 4. Review Responses

**Chat Messages:**
- **Thinking Process**: Expand "CAG's thoughts" to see reasoning
- **Streaming Answers**: Watch responses generate in real-time
- **Cache Indicator**: "💾 Using cached response" shows when answers are cached

**Referenced Sections:**
- Automatically expands sections cited in the answer
- Click section expanders to view full content
- Includes page ranges and section metadata

**Cache Status:**
- Green "💾 Using cached response" = instant retrieval from LanceDB
- No indicator = fresh LLM generation + automatic caching to LanceDB

#### 5. Manage Caches

**Cache Stats (Sidebar):**
- View total contexts, tokens, and cache hits
- Clear all context cache with "🧹 Clear Cache"

**Q&A Cache Management:**
- View cached Q&A pairs per document
- Browse questions with thinking and responses
- Clear per-document cache or all Q&A cache
- Persistent storage in LanceDB (no memory limits)

**Question Library:**
- Search library with autocomplete
- View usage counts and categories
- Delete individual questions
- Clear entire library

## Project Structure

```
cagvault/
├── app.py                          # Streamlit UI with enhanced features
├── config.py                       # Model configuration and settings
├── models.py                       # LLM factory (Ollama/Groq)
├── knowledge.py                    # Document loading and conversion
├── chatbot.py                      # Chat logic with streaming and prompts
├── kvcache.py                      # KV-Cache manager for context caching
├── lancedb_cache.py                # LanceDB storage layer with in-process cache
├── qa_cache.py                     # LanceDB-backed Q&A caching system
├── question_library.py             # Question library with categorization
├── simple_cag.py                   # Simplified CAG implementation
├── pyproject.toml                  # Python dependencies
├── lancedb/                        # LanceDB embedded database directory
│   ├── doc_sections.lance/         # Document sections table
│   ├── qa_cache.lance/             # Q&A cache table
│   └── question_library.lance/    # Question library table
├── skills/
│   └── pdf_parser/
│       ├── pdf_parser.py           # Core PDF parsing (Docling)
│       ├── enhanced_parser.py      # LLM-powered section analysis
│       ├── ner_search.py           # NER and search engines
│       ├── credit_analyst_prompt.py # Credit analyst classification
│       └── llm_section_evaluator.py # Section importance scoring
├── .cache/
│   ├── documents/                  # Parsed document cache
│   ├── kvcache/                    # KV-cache storage
│   └── toc_sections/               # TOC-based section extraction
└── README.md                       # This file
```

## Configuration

### Model Selection

By default, CagVault uses **Qwen3-14B locally via Ollama**. To change models, edit `config.py`:

```python
class Config:
    MODEL = DEEPSEEK_V3  # Change to any model constant
    OLLAMA_CONTEXT_WINDOW = 8192  # Adjust context size
```

### Supported Providers

Currently, CagVault supports:

- **Ollama** (default): Local inference, completely private, no API key needed
- **Groq** (optional): Cloud inference, requires `GROQ_API_KEY` environment variable

### Model Comparison for RAG

| Model | Size | RAM Required | Context Window | Best For | Speed |
|-------|------|--------------|----------------|----------|-------|
| **DeepSeek V3** | 685B | 64GB+ | 64K | Best overall quality, complex reasoning | Slow |
| **DeepSeek R1** | ~70B | 32GB+ | 32K | Advanced reasoning, credit analysis | Medium |
| **Command R+** | ~104B | 32GB+ | 128K | RAG-optimized, long documents | Medium |
| **Mistral Large** | ~123B | 32GB+ | 128K | Long-context tasks | Medium |
| **Llama 3.3 70B** | 70B | 32GB+ | 128K | Strong reasoning, instruction following | Medium |
| **Gemma 2 27B** | 27B | 16GB+ | 8K | Balanced reasoning | Fast |
| **Qwen3-14B** ⭐ | 14B | 16GB | 8K | Default, excellent balance | Fast |
| **Phi-4** | 14B | 16GB | 16K | Efficient, Microsoft-optimized | Fast |
| **Llama 3.1 8B** | 8B | 8GB | 128K | Lightweight, fast responses | Very Fast |
| **Mistral Small** | 7B | 8GB | 32K | Simple queries, minimal resources | Very Fast |

⭐ = Default model

### Adding Custom Models

To add a new model not in the config:

```python
# In config.py, add your model
MY_CUSTOM_MODEL = ModelConfig(
    "model-name-from-ollama",
    temperature=0.0,
    provider=ModelProvider.OLLAMA
)

# Then set it as default
class Config:
    MODEL = MY_CUSTOM_MODEL
```

For a full list of available models, visit [ollama.com/library](https://ollama.com/library)

## Troubleshooting

### Ollama Connection Error

**Error**: `httpx.ConnectError: [Errno 61] Connection refused`

**Solution**: Start the Ollama service:
```bash
brew services start ollama  # macOS
# or
ollama serve &  # Linux
```

### Model Not Found

**Error**: `ollama.ResponseError: model 'xyz' not found`

**Solution**: Pull the model first:
```bash
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL
```

Or use a different available model:
```bash
ollama pull llama2:latest
```

### Python Version Issues

**Error**: Pydantic warnings or import errors

**Solution**: Ensure you're using Python 3.12:
```bash
python --version
# If not 3.12, recreate the virtual environment with Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .
```

### Out of Memory

If the model runs out of memory during inference:

- Use a smaller model (e.g., `llama2:latest` instead of `qwen3:14b`)
- Reduce `OLLAMA_CONTEXT_WINDOW` in `config.py`
- **Reduce `OLLAMA_NUM_PARALLEL`** in `config.py` (try 2 instead of 4)
- Close other applications
- Increase system swap space

### Slow or Hanging Requests

If requests are timing out or hanging:

**Check concurrent load**:
```bash
# Monitor Ollama connections
lsof -i :11434 | wc -l  # Count active connections
```

**Solutions**:
- Increase `REQUEST_TIMEOUT` in `config.py` for complex queries
- Reduce `OLLAMA_NUM_PARALLEL` if system is overloaded
- Check Ollama logs: `ollama logs` or check system console
- Restart Ollama: `brew services restart ollama`

**Optimal settings by RAM**:
- 8-16GB RAM: `OLLAMA_NUM_PARALLEL = 2`
- 16-32GB RAM: `OLLAMA_NUM_PARALLEL = 4` (default)
- 32GB+ RAM: `OLLAMA_NUM_PARALLEL = 6-8`

See [CONCURRENT_REQUESTS.md](CONCURRENT_REQUESTS.md) for detailed tuning guide.

### LanceDB Storage Issues

**Error**: Database connection or table access issues

**Solution**: 
```bash
# Check LanceDB directory permissions
ls -la ./lancedb

# If corrupted, remove and restart (will lose cached data)
rm -rf ./lancedb
streamlit run app.py  # Tables will be recreated

# To inspect LanceDB contents
python -c "import lancedb; db = lancedb.connect('./lancedb'); print(db.list_tables().tables)"
```

**Note**: LanceDB is embedded and requires no separate server. All data is stored locally in the `./lancedb` directory.

### KV-Cache Issues

If cache seems corrupted or causes issues:

```bash
# Clear the KV cache
rm -rf .cache/kvcache/

# Or clear via the UI
# Click "🧹 Clear Cache" in the sidebar
```

### Q&A Cache Issues

If cached answers seem outdated or incorrect:

```bash
# Clear Q&A cache via UI:
# 1. Expand "💾 Q&A Cache Management" in sidebar
# 2. Click "🗑️ Clear All Q&A Cache"

# Or clear LanceDB cache programmatically:
python -c "from qa_cache import get_qa_cache; get_qa_cache().clear_all_cache()"

# Or remove the entire QA table:
python -c "import lancedb; db = lancedb.connect('./lancedb'); db.drop_table('qa_cache')"
```

### Duplicate Sections / Looping

If you see repeated sections in the UI or logs:

**Cause**: Document loaded multiple times without clearing memory

**Solution**: This should be automatically prevented by the deduplication guards. If it still occurs:
```bash
# Restart the app (clears in-memory state)
pkill -f streamlit
streamlit run app.py

# Or clear LanceDB document cache
python -c "import lancedb; db = lancedb.connect('./lancedb'); db.drop_table('doc_sections')"
```

### Section References Not Appearing

If cited sections don't auto-expand in chat:

**Check**:
1. LLM is citing sections by number (e.g., "Section 5.12.2") or title
2. Document has been parsed with enhanced parser (not URL-only)
3. Section titles match citation format

**Debug**: Check the logs for "Referenced sections" or "No section titles detected"

## Performance Considerations

Based on the CAG paper's experiments:

- **Small contexts** (3-16 docs, ~21k tokens): CAG provides **10x+ speedup** over dynamic context loading
- **Medium contexts** (4-32 docs, ~32-43k tokens): CAG offers **17x+ speedup**
- **Large contexts** (7-64 docs, ~50-85k tokens): CAG achieves **40x+ speedup**

The precomputed KV cache eliminates the need to reprocess documents for each query, making multi-turn conversations dramatically faster.

## Technical Details

### How CAG Works in This Application

1. **Document Upload**: User uploads files or provides URLs
2. **Conversion**: Docling converts documents to plain text
3. **Context Preloading**: Documents are concatenated and passed to the LLM
4. **KV Cache**: Ollama automatically caches the model's inference state (handled internally)
5. **Query Processing**: User questions are appended to the cached context
6. **Streaming Response**: The model generates answers using the preloaded knowledge

## Technical Details

### How CAG Works in This Application

1. **Document Upload**: User uploads files or provides URLs
2. **Conversion**: Docling converts documents to plain text
3. **Context Preloading**: Documents are concatenated and passed to the LLM
4. **KV-Cache Creation**: The model's inference state is precomputed and stored
5. **Efficient Queries**: User questions are processed using the cached context
6. **Streaming Response**: The model generates answers using preloaded knowledge

### Current Architecture (December 2025)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAGVAULT ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  DOCUMENT INGESTION PIPELINE                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Documents (PDF/TXT/MD/URL)                                           │
│           │                                                                 │
│           ▼                                                                 │
│  ┌───────────────────┐                                                     │
│  │  Docling Parser   │  ← Converts PDFs with layout preservation          │
│  │  (skills/pdf_*)   │  ← OCR support (optional)                          │
│  └────────┬──────────┘  ← Table detection                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌───────────────────────────────────────────────┐                        │
│  │  Enhanced Parser (LLM-Powered Analysis)       │                        │
│  │                                                │                        │
│  │  • Hierarchical section extraction             │                        │
│  │  • Parallel LLM importance scoring (4 workers) │                        │
│  │  • Credit analyst classification               │                        │
│  │  • Page-accurate tracking (word-based)         │                        │
│  │  • Named Entity Recognition (NER)              │                        │
│  │  • Cross-reference detection                   │                        │
│  └────────┬──────────────────────────────────────┘                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌───────────────────────────────────────────────┐                        │
│  │  SectionMemoryStore (In-Memory)               │                        │
│  │                                                │                        │
│  │  • Hierarchical document structure             │                        │
│  │  • Section → Subsection relationships          │                        │
│  │  • Metadata indexing (pages, importance, type) │                        │
│  │  • Deduplication prevention                    │                        │
│  └────────┬──────────────────────────────────────┘                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌───────────────────────────────────────────────┐                        │
│  │  LanceDB Persistent Storage (Embedded)        │                        │
│  │                                                │                        │
│  │  Table: doc_sections                          │                        │
│  │  • Hierarchical sections (parent_id, order)   │                        │
│  │  • Full-text search indexes (content, title)  │                        │
│  │  • Pre-computed keywords & entities           │                        │
│  │  • Document metadata (pages, type, size)      │                        │
│  │                                                │                        │
│  │  In-Process Cache: 3s TTL DataFrame           │                        │
│  │  • Sub-millisecond reads for frequent access  │                        │
│  │  • Thread-safe with automatic invalidation    │                        │
│  └────────────────────────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SEARCH & RETRIEVAL LAYER                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌───────────────────┐  ┌────────────────────┐    │
│  │ Keyword Search   │  │ Semantic Search   │  │ Agentic Search     │    │
│  │ (FullTextSearch) │  │ (Embedding-based) │  │ (Claude-powered)   │    │
│  └────────┬─────────┘  └────────┬──────────┘  └─────────┬──────────┘    │
│           │                     │                        │                │
│           └─────────────────────┴────────────────────────┘                │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌────────────────────────┐                             │
│                    │  Search Results        │                             │
│                    │  + Relevance Scores    │                             │
│                    │  + Reasoning (Agentic) │                             │
│                    └────────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CHAT & Q&A LAYER                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Question                                                              │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────┐                              │
│  │  Question Library (LanceDB)             │                              │
│  │                                          │                              │
│  │  Table: question_library                │                              │
│  │  • 15+ categories (Definitions, etc.)   │                              │
│  │  • Usage tracking & popularity          │                              │
│  │  • Autocomplete suggestions (FTS)       │                              │
│  │  • Per-document & global questions      │                              │
│  │  • In-process cache (3s TTL)            │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  Q&A Cache (LanceDB)                    │                              │
│  │                                          │                              │
│  │  Table: qa_cache                        │                              │
│  │  Key: sha256(question + doc_ids)        │                              │
│  │  Value: {response, thinking, metadata}  │                              │
│  │  FTS Index: question field              │                              │
│  │                                          │                              │
│  │  Cache Hit? → Return cached response ✓  │                              │
│  │  Cache Miss? → Continue to LLM ↓        │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  Context Builder                        │                              │
│  │                                          │                              │
│  │  • Load full document content           │                              │
│  │  • Build hierarchical context           │                              │
│  │  • Include section metadata             │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  KV-Cache Manager                       │                              │
│  │                                          │                              │
│  │  • Precompute context state             │                              │
│  │  • Track token counts                   │                              │
│  │  • Deduplicate sources                  │                              │
│  │  • Persistent disk storage              │                              │
│  │                                          │                              │
│  │  10-40x speedup for multi-turn chat!    │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  Ollama LLM Server (4 Parallel Workers)│                              │
│  │                                          │                              │
│  │  Model: Qwen3-14B (Q4_K_XL quantized)   │                              │
│  │  Context: 8K+ tokens                    │                              │
│  │  Temperature: 0.0 (deterministic)       │                              │
│  │                                          │                              │
│  │  ┌────────────────────────────┐         │                              │
│  │  │ System Prompt              │         │                              │
│  │  │ • Credit analyst expertise │         │                              │
│  │  │ • Cross-reference checking │         │                              │
│  │  │ • Citation requirements    │         │                              │
│  │  └────────────────────────────┘         │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  Response Stream                        │                              │
│  │                                          │                              │
│  │  <think>...</think> → Reasoning         │                              │
│  │  Answer → Final response                │                              │
│  │                                          │                              │
│  │  • Auto-cache to LanceDB                │                              │
│  │  • Extract section references           │                              │
│  │  • Track to question library            │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                              │
│  │  Referenced Section Matcher             │                              │
│  │                                          │                              │
│  │  • Regex-based title matching           │                              │
│  │  • Numeric prefix detection (5.12.2)    │                              │
│  │  • Section/§ prefix variants            │                              │
│  │  • Case-insensitive matching            │                              │
│  │  • Subsection inclusion                 │                              │
│  └────────┬────────────────────────────────┘                              │
│           │                                                                 │
│           ▼                                                                 │
│  Streamlit UI Display:                                                     │
│  • Chat messages                                                           │
│  • Expandable thinking blocks                                              │
│  • Referenced section expanders with full content                          │
│  • Cache status indicators                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA FLOW SUMMARY                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. UPLOAD: PDF → Docling → Enhanced Parser → Section Analysis (parallel)  │
│  2. STORE:  Sections → Memory + LanceDB persistence                        │
│  3. INDEX:  Keywords + Entities + Semantic embeddings                      │
│  4. QUERY:  Question → Library + Q&A Cache check                           │
│  5. SEARCH: Keyword/Semantic/Agentic → Relevant sections                   │
│  6. BUILD:  Context from sections → KV-Cache                               │
│  7. INFER:  LLM with cached context → Streamed response                    │
│  8. MATCH:  Extract section refs → Auto-expand in UI                       │
│  9. CACHE:  Store Q&A + Update library + Track usage                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### Core Infrastructure
- **Ollama**: Local LLM inference server (Qwen3-14B)
- **LanceDB**: Embedded vector database for persistent storage (Q&A cache, sections, questions)
- **Streamlit**: Interactive web UI with real-time updates
- **LangChain**: LLM orchestration and streaming

#### Document Processing
- **Docling** (`skills/pdf_parser/pdf_parser.py`): PDF/HTML/TXT/MD conversion with layout preservation
- **EnhancedPDFParserSkill** (`skills/pdf_parser/enhanced_parser.py`): 
  - LLM-powered section extraction
  - Parallel importance scoring (ThreadPoolExecutor)
  - Hierarchical structure with page tracking
  - LanceDB persistence with deduplication guards
- **SectionMemoryStore**: In-memory hierarchical document structure
- **NamedEntityRecognizer** (`skills/pdf_parser/ner_search.py`): Extract and index entities

#### Search & Retrieval
- **FullTextSearchEngine**: Fast keyword search with tokenization
- **Semantic Search**: Embedding-based similarity matching
- **Agentic Search**: Claude-powered intelligent query understanding

#### Caching System
- **KVCacheManager** (`kvcache.py`): Context state caching with disk persistence
- **QACacheManager** (`qa_cache.py`): LanceDB-backed Q&A caching with persistent storage
- **QuestionLibraryManager** (`question_library.py`): Question tracking with categorization and usage analytics
- **LanceDBStore** (`lancedb_cache.py`): Unified storage layer with in-process DataFrame cache (3s TTL)

#### Credit Analysis
- **CreditAnalystPrompt** (`skills/pdf_parser/credit_analyst_prompt.py`): Section classification and importance
- **LLMSectionEvaluator** (`skills/pdf_parser/llm_section_evaluator.py`): Batch analysis with parallel processing

#### LanceDB Storage Architecture

**Unified Storage Layer** (`lancedb_cache.py`):
- **Embedded Vector Database**: No external server required, all data in `./lancedb` directory
- **Three Main Tables**:
  1. **doc_sections**: Hierarchical document sections with full-text search
  2. **qa_cache**: Question-answer pairs with thinking and metadata
  3. **question_library**: Popular questions with usage tracking and categorization

**Schema Design**:
```python
# doc_sections table
document_id: string          # Unique document identifier
document_name: string        # Human-readable name
section_id: string          # Section unique ID
parent_id: string           # Parent section for hierarchy
level: int32                # Nesting level (1, 2, 3...)
order_idx: int32            # Preservation of document order
title: string               # Section title
content: string             # Section text content
keywords: list<string>      # Pre-computed search tokens
entities_json: string       # NER results (JSON)
metadata_json: string       # Section metadata
total_pages: int32          # Document page count
extraction_method: string   # Parser version/method
source: string              # Origin (upload, URL, etc.)
stored_at: string           # Timestamp (ISO 8601)

# qa_cache table
cache_key: string           # SHA256 hash of question + doc_ids
question: string            # Original question
response: string            # LLM answer
thinking: string            # Reasoning process
doc_ids: list<string>       # Associated documents
timestamp: string           # Cache creation time
metadata_json: string       # Model, source count, etc.

# question_library table
question: string            # Unique question text (normalized)
doc_ids: list<string>       # Related documents
category: string            # Question category
usage_count: int64          # Popularity metric
is_default: bool            # Pre-seeded question
created_at: string          # Creation timestamp
metadata_json: string       # Additional metadata
```

**Performance Optimizations**:
1. **Full-Text Search (FTS) Indexes**:
   - doc_sections: `content`, `title`, `document_name`
   - qa_cache: `question`
   - question_library: `question`

2. **In-Process DataFrame Cache** (3-second TTL):
   - Caches table contents as pandas DataFrames in memory
   - Sub-millisecond reads for frequent queries
   - Thread-safe with locks
   - Automatic invalidation on writes
   - Warmed on startup for instant first access

3. **Write Strategy**:
   - Immediate writes to LanceDB (ACID-compliant)
   - Cache invalidation triggered after successful write
   - No blocking - operations complete quickly

**Data Flow**:
```
┌──────────────────────────────────────────────────────────┐
│ Application Request (Read)                               │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ Check In-Process Cache (3s TTL)                            │
│ • Thread-safe lock acquisition                             │
│ • Check timestamp validity                                 │
└────┬───────────────────────────────────────────┬───────────┘
     │ Hit                                       │ Miss
     ▼                                           ▼
┌─────────────────┐                   ┌──────────────────────┐
│ Return DataFrame│                   │ Query LanceDB Table  │
│ (sub-ms)        │                   │ • Convert to pandas  │
└─────────────────┘                   │ • Store in cache     │
                                      │ • Return DataFrame   │
                                      └──────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Application Request (Write)                              │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ Write to LanceDB                                           │
│ • ACID transaction                                         │
│ • Immediate persistence                                    │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│ Invalidate In-Process Cache                                │
│ • Remove cached DataFrame                                  │
│ • Next read will refresh from disk                         │
└────────────────────────────────────────────────────────────┘
```

**Migration from Redis**:
- Optional one-time migration utility: `lancedb_cache.migrate_from_redis(redis_client)`
- Imports documents, Q&A cache, and question library
- Preserves all metadata and relationships
- No data loss during transition

### Performance Optimizations

#### Multi-Layer Caching Strategy

**1. KV-Cache (Context State)**
- **No document reprocessing**: Once cached, documents aren't re-tokenized
- **Multi-turn speedup**: 10-40x faster for subsequent queries (from CAG paper)
- **Memory efficient**: Tracks token counts and cache size
- **Automatic deduplication**: Same documents aren't cached twice
- **Persistent storage**: Caches stored on disk for reuse across sessions

**2. Q&A Cache (Response Level)**
- **Instant retrieval**: Identical questions return cached answers immediately
- **Document-aware**: Cache keys include document IDs for precise matching
- **Persistent storage**: No expiration, manually managed via UI or API
- **Thinking included**: Caches both reasoning and final response
- **Per-document management**: Clear cache for specific documents

**3. Document Section Cache (LanceDB)**
- **Parse once**: Parsed sections persisted to LanceDB with FTS indexes
- **Fast reload**: Load document structure without re-parsing (in-process cache)
- **Hierarchical storage**: Maintains parent-child relationships via order_idx
- **Search index**: Pre-computed keywords and entities with full-text search
- **Deduplication guards**: Prevents repeated section additions
- **In-process cache**: 3-second TTL DataFrame cache for frequent reads

#### Parallel Processing & Concurrent Requests

**Concurrent Request Handling**
- **4 parallel LLM workers** handle simultaneous requests
- Non-blocking chat responses during document processing
- Multiple users can interact concurrently
- Configurable via `Config.OLLAMA_NUM_PARALLEL`
- 5-minute request timeout prevents hanging operations
- See [CONCURRENT_REQUESTS.md](CONCURRENT_REQUESTS.md) for detailed configuration

**Section Analysis (4 workers)**
- Concurrent LLM calls for importance scoring
- Classification of section types (COVENANT, DEFAULT, etc.)
- Batch processing of subsections
- Progress logging every 10 sections

**Word-Based Page Estimation**
- ~250 words per page heuristic
- Instant calculation vs. slow LLM page range calls
- Accurate enough for UI display and citations

#### Memory Management

**In-Memory Section Store**
- Fast lookups by section ID
- Hierarchical traversal for subsections
- Automatic memory clearing before fresh loads
- Prevents duplicate section accumulation

## Best Practices

### For Credit Agreement Analysis

1. **Upload Full Agreement**: Include all sections, schedules, and amendments
2. **Let Parsing Complete**: Wait for parallel LLM analysis to finish (progress shown)
3. **Use Agentic Search**: For complex queries, agentic search provides reasoning
4. **Check Referenced Sections**: Always expand cited sections to verify context
5. **Review Cache**: Use Q&A cache management to track analysis history

### For Optimal Performance

1. **Enable Redis**: Install and run Redis for best caching performance
2. **Batch Upload**: Upload all related documents before starting Q&A
3. **Use Suggested Questions**: Build question library for faster team collaboration
4. **Monitor Cache Stats**: Clear old caches periodically to free memory
5. **Parallel Processing**: Parser uses 4 workers by default; increase for faster analysis

### For Question Library

1. **Categorize Thoughtfully**: Questions are auto-categorized but review for accuracy
2. **Track Usage**: Popular questions surface to the top automatically
3. **Search Before Asking**: Use autocomplete to find existing answers
4. **Document-Specific**: Filter questions by document for focused analysis
5. **Clear Periodically**: Remove outdated questions to keep library relevant

### For Multi-Document Context

1. **Related Documents**: Upload contracts and amendments together
2. **Clear Context Cache**: When switching document sets, clear cache
3. **Check Message Source IDs**: Verify which documents are in context
4. **Redis Loading**: For frequently used documents, load from Redis cache

## Limitations

### Context Window Constraints
- **Qwen3-14B**: ~8K tokens (~3-4 medium PDFs or 1 large credit agreement)
- **Token Estimation**: ~750 tokens per page for dense legal documents
- **Workaround**: Focus on specific sections or use search to find relevant parts

### Memory Requirements
- **Minimum**: 8GB RAM for 7B models
- **Recommended**: 16GB RAM for 14B models
- **With Redis**: Additional ~100MB-1GB depending on document count
- **Section Analysis**: Uses 4 parallel workers (can adjust in code)

### Redis Dependency
- **Optional**: App works without Redis but with limited features
- **Q&A Cache**: Requires Redis for persistence
- **Question Library**: Requires Redis for cross-session storage
- **Document Sections**: Can use memory-only but won't persist

### Not Ideal For
- **Constantly Updating Knowledge**: Traditional RAG better for dynamic data
- **Very Large Corpora**: 100+ documents may exceed context limits
- **Real-Time Collaboration**: Single-user app, not designed for teams
- **Production Deployments**: This is a research/analysis tool, not a production service

## Recent Changes (December 2025)

### Enhanced PDF Intelligence
- ✅ **Parallel LLM Section Analysis**: 4 concurrent workers for faster parsing
- ✅ **Credit Analyst Classification**: Automatic detection of COVENANTS, DEFAULTS, etc.
- ✅ **Importance Scoring**: AI-driven relevance analysis (0-1 scale)
- ✅ **Page-Accurate Tracking**: Word-based estimation for instant page mapping
- ✅ **Hierarchical Sections**: Full parent-child relationships preserved

### Search & Discovery
- ✅ **Multi-Modal Search**: Keyword, semantic, and agentic (Claude-powered)
- ✅ **Named Entity Recognition**: Extract PARTY, DATE, MONEY, AGREEMENT entities
- ✅ **Entity Filtering**: Browse by entity type across all sections
- ✅ **Section References**: Auto-expand cited sections in chat responses

### Caching System
- ✅ **Q&A Cache**: LanceDB-backed with persistent storage
- ✅ **Question Library**: 15+ categories with autocomplete
- ✅ **Suggested Questions**: Popular queries by document or global
- ✅ **Cache Analytics**: Real-time stats and management UI
- ✅ **Deduplication Guards**: Prevent repeated section additions

### UI/UX Improvements
- ✅ **Document Tabs**: Sections, Search, Entities in organized tabs
- ✅ **Cache Indicators**: Visual feedback for cache hits
- ✅ **Referenced Section Expanders**: Click to view full cited sections
- ✅ **Browse by Category**: Explore questions by type
- ✅ **LanceDB Document Picker**: Load previously parsed documents

### Performance
- ✅ **Concurrent Request Handling**: 4 parallel LLM workers for simultaneous requests
- ✅ **Memory Management**: Automatic clearing before fresh loads
- ✅ **Parallel Processing**: ThreadPoolExecutor for section analysis
- ✅ **LanceDB Persistence**: Store parsed sections with FTS indexes for instant reload
- ✅ **Word-Based Estimation**: Fast page calculation without LLM calls
- ✅ **Connection Pooling**: Optimized Ollama connections with timeout management

### Technical
- ✅ **Python 3.14 Support**: Compatible with latest Python
- ✅ **Embedded Storage**: No external database server required
- ✅ **Enhanced Error Handling**: Better logging and fallbacks
- ✅ **Document Deduplication**: Prevent duplicate button keys

## Citation

If you use this project or the CAG methodology, please cite the original paper:

```bibtex
@inproceedings{chan2025cag,
  title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author={Chan, Brian J and Chen, Chao-Ting and Cheng, Jui-Hung and Huang, Hen-Hsen},
  booktitle={Proceedings of the ACM Web Conference 2025},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Author

Created by [Amitabha Karmakar](https://www.linkedin.com/in/amitabha-karmakar/)

## Support

### Getting Help

**For Issues or Questions:**
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the [Best Practices](#best-practices) for optimal usage
3. Check logs in the terminal where you ran `streamlit run app.py`
4. Open a GitHub issue with:
   - Error message and full traceback
   - Python version (`python --version`)
   - Ollama status (`ollama list`)
   - LanceDB tables (`python -c "import lancedb; print(lancedb.connect('./lancedb').list_tables().tables)"`)
   - Steps to reproduce

**Documentation:**
- CAG Paper: https://arxiv.org/abs/2412.15605v1
- Implementation Details:
  - `documentation/AGENTIC_RAG_GUIDE.md` - **NEW!** Multi-step reasoning RAG system
  - `documentation/AGENT_SDK_INTEGRATION.md` - **NEW!** Claude Agent SDK MCP tools
  - `documentation/MCP_TOOLS_GUIDE.md` - MCP tools user guide
  - `documentation/QA_CACHE_IMPLEMENTATION.md` - Q&A caching system
  - `documentation/QUESTION_LIBRARY_IMPLEMENTATION.md` - Question library design
  - `documentation/PDF_PARSER_SKILL_SUMMARY.md` - Enhanced PDF parsing
  - `documentation/CLAUDE_SKILLS_GUIDE.md` - Claude skills integration
  - `skills/pdf_parser/ENHANCED_PARSER_GUIDE.md` - Advanced document parsing

**Logs & Debugging:**
```bash
# Check terminal output for detailed logs
# Logs include:
# - Section extraction progress
# - LLM analysis status
# - Cache hits/misses
# - LanceDB storage status
# - Entity extraction results

# Enable more verbose logging (if needed):
export LOG_LEVEL=DEBUG
streamlit run app.py
```

---

Built with ❤️ using Qwen3, Ollama, LangChain, Docling, and Streamlit
