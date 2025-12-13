from dataclasses import dataclass
from enum import Enum

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
 
@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider
 
# === HIGH-PERFORMANCE RAG MODELS ===

# Qwen3 - Excellent for document understanding and credit analysis
QWEN_3 = ModelConfig(
    "hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL", temperature=0.0, provider=ModelProvider.OLLAMA
)

# DeepSeek V3 - State-of-the-art reasoning and document comprehension
DEEPSEEK_V3 = ModelConfig(
    "deepseek-ai/DeepSeek-V3", temperature=0.0, provider=ModelProvider.OLLAMA
)

# DeepSeek R1 - Advanced reasoning capabilities for complex queries
DEEPSEEK_R1 = ModelConfig(
    "deepseek-ai/DeepSeek-R1", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Mistral Large - Excellent for long-context RAG tasks
MISTRAL_LARGE = ModelConfig(
    "mistral-large-latest", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Mistral Small - Fast and efficient for simpler queries
MISTRAL_SMALL = ModelConfig(
    "mistral-small-latest", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Llama 3.3 70B - Strong reasoning and instruction following
LLAMA_3_3_70B = ModelConfig(
    "llama3.3:70b", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Llama 3.1 8B - Lightweight option for faster responses
LLAMA_3_1_8B = ModelConfig(
    "llama3.1:8b", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Phi-4 - Microsoft's efficient small model (14B) for document QA
PHI_4 = ModelConfig(
    "phi4:latest", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Gemma 2 27B - Google's strong reasoning model
GEMMA_2_27B = ModelConfig(
    "gemma2:27b", temperature=0.0, provider=ModelProvider.OLLAMA
)

# Command R+ - Cohere's RAG-optimized model
COMMAND_R_PLUS = ModelConfig(
    "command-r-plus:latest", temperature=0.0, provider=ModelProvider.OLLAMA
)

# === CLOUD MODELS (Groq) ===

LLAMA_3_3 = ModelConfig(
    "meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, provider=ModelProvider.GROQ
)
 
class Config:
    SEED = 42
    MODEL = QWEN_3
    OLLAMA_CONTEXT_WINDOW = 8192
    
    # Concurrent request handling
    OLLAMA_NUM_PARALLEL = 4  # Number of concurrent requests Ollama can handle
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # Connection pooling for better concurrency
    REQUEST_TIMEOUT = 300  # 5 minutes for long-running requests
    MAX_RETRIES = 3
 
    class UI:
        ALLOWED_FILE_TYPES = ["pdf", "txt", "md"]
