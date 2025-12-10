# CagVault

A **Cache-Augmented Generation (CAG)** application for private, local document chat using large language models with extended context windows.

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
│  │  │  LLM Processor   │  Precompute KV-Cache                        │   │
│  │  │  (Batch Process) │  (Encodes all knowledge)                    │   │
│  │  └────────┬─────────┘                                             │   │
│  │           │                                                        │   │
│  │           ▼                                                        │   │
│  │  ┌──────────────────────┐                                         │   │
│  │  │  Cached KV-State     │  💾  Stored on Disk/Memory             │   │
│  │  │  (Ready to use)      │                                         │   │
│  │  └──────────┬───────────┘                                         │   │
│  │             │                                                     │   │
│  └─────────────┼─────────────────────────────────────────────────────┘   │
│                │                                                           │
│  ┌─── INFERENCE PHASE (Fast) ────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  User Query        Cached KV-State                               │   │
│  │      │                  │                                        │   │
│  │      └──────────┬───────┘                                        │   │
│  │                 ▼                                                │   │
│  │        ┌──────────────────────┐                                 │   │
│  │        │  LLM with Preloaded  │  ✨ NO RETRIEVAL!               │   │
│  │        │  Context + KV-Cache  │  ✨ NO LATENCY!                │   │
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

## Features

- 🔒 **Fully Local & Private**: No API keys, cloud services, or internet required
- 📄 **Multi-Format Support**: PDF, TXT, MD files and web URLs
- 💬 **Streaming Chat**: Real-time response generation with thinking process visibility
- 🧠 **Extended Context**: Leverages Qwen3-14B's 8K+ context window
- ⚡ **KV-Cache Optimization**: Precomputed context caching for 10-40x faster multi-turn queries
- 🎨 **Modern UI**: Clean Streamlit interface with cache statistics

## Prerequisites

- macOS (or Linux/Windows with appropriate package managers)
- Python 3.12.x
- Homebrew (for macOS)
- At least 10GB free disk space (for the LLM model)

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

### 5. Download the Qwen3 Model

Pull the Qwen3-14B quantized model (~9.2GB download):

```bash
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL
```

This will download the model to your local machine. The download may take 10-20 minutes depending on your internet speed.

**Alternative Models:**

If you want to use a different model, you can browse available models:

```bash
ollama list
ollama pull <model-name>
```

Popular alternatives:
- `llama3.3:latest` - Llama 3.3 (default 8B)
- `mistral:latest` - Mistral 7B
- `qwen2.5:latest` - Qwen 2.5

To change the model in the app, edit `config.py`:

```python
QWEN_3 = ModelConfig(
    provider=ModelProvider.OLLAMA,
    name="your-model-name-here"  # Change this
)
```

### 6. Verify Installation

Check that everything is installed correctly:

```bash
# Python environment
python --version  # Should show 3.12.x

# Ollama service
ollama list  # Should show your downloaded models

# Python packages
pip list | grep -E "(streamlit|langchain|docling)"
```

## Running the Application

### Start the Streamlit App

With your virtual environment activated:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8504`

### Using the Application

1. **Upload Documents** (optional):
   - Use the sidebar file uploader to add PDF, TXT, or MD files
   - Or enter a URL to scrape web content

2. **Start Chatting**:
   - Type your question in the chat input at the bottom
   - The model will process your documents (if uploaded) and generate a response
   - Watch the thinking process unfold in real-time

3. **View Responses**:
   - Thinking blocks show the model's reasoning process
   - Final answers appear as assistant messages

## Project Structure

```
cagvault/
├── app.py           # Streamlit UI and main application logic
├── config.py        # Model configuration and settings
├── models.py        # LLM factory (creates Ollama/Groq instances)
├── knowledge.py     # Document loading and conversion
├── chatbot.py       # Chat logic with streaming and prompts
├── simple_cag.py    # Simplified CAG implementation
├── pyproject.toml   # Python dependencies
└── README.md        # This file
```

## Configuration

### Model Selection

By default, CagVault uses **Qwen3-14B locally via Ollama**. To change models, edit `config.py`:

```python
class Config:
    MODEL = QWEN_3  # Change to LLAMA_3_3 or add your own
    OLLAMA_CONTEXT_WINDOW = 8192  # Adjust context size
```

### Supported Providers

Currently, CagVault supports:

- **Ollama** (default): Local inference, completely private, no API key needed
- **Groq** (optional): Cloud inference, requires `GROQ_API_KEY` environment variable

### Available Models

**Local Models (via Ollama):**
- `hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL` (default) - Qwen3 14B, 8K context
- `llama2:latest` - Llama 2 7B
- `mistral:latest` - Mistral 7B
- See [ollama.com/library](https://ollama.com/library) for more

**Cloud Models (via Groq):**
- `meta-llama/llama-3.1-8b-instant`
- `mixtral-8x7b-32768`

To use a different local model, pull it first:
```bash
ollama pull <model-name>
```

Then update `config.py`:
```python
QWEN_3 = ModelConfig(
    "<model-name>",
    temperature=0.0,
    provider=ModelProvider.OLLAMA
)
```

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
- Close other applications
- Increase system swap space

### KV-Cache Issues

If cache seems corrupted or causes issues:

```bash
# Clear the cache
rm -rf .cache/kvcache/

# Or clear via the UI
# Click "🧹 Clear Cache" in the sidebar
```

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

### Architecture

```
User Documents
    ↓
Docling (Format Conversion)
    ↓
KV-Cache Manager (Precompute & Store)
    ↓
Ollama (Local LLM Inference)
    ↓
Streamlit UI (Chat Interface)
```

### Key Components

- **kvcache.py**: Manages KV-state caching with in-memory and disk storage
- **Docling**: Converts PDF/HTML/TXT/MD → plain text with layout preservation
- **LangChain**: Orchestrates LLM interactions and streaming
- **Ollama**: Local inference server with automatic KV caching
- **Streamlit**: Renders the chat UI with real-time updates

### Performance Optimizations

The KV-Cache implementation provides:

- **No document reprocessing**: Once cached, documents aren't re-tokenized
- **Multi-turn speedup**: 10-40x faster for subsequent queries (from paper)
- **Memory efficient**: Tracks token counts and cache size
- **Automatic deduplication**: Same documents aren't cached twice
- **Persistent storage**: Caches are stored on disk for reuse across sessions

## Limitations

- **Context Window**: Currently limited to ~8k tokens (Qwen3) or ~128k tokens (Llama 3.1)
- **Document Size**: Works best with constrained knowledge bases that fit in context
- **Memory Usage**: Large models require significant RAM (8GB+ for 7B models, 16GB+ for 14B models)
- **Not for Unbounded Knowledge**: For very large or constantly updating knowledge bases, traditional RAG may be more appropriate

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

## Support

For issues or questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the CAG paper: https://arxiv.org/abs/2412.15605v1

---

Built with ❤️ using Qwen3, Ollama, LangChain, Docling, and Streamlit
