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
 
QWEN_3 = ModelConfig(
    "hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL", temperature=0.0, provider=ModelProvider.OLLAMA
)
LLAMA_3_3 = ModelConfig(
    "meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, provider=ModelProvider.GROQ
)
 
class Config:
    SEED = 42
    MODEL = QWEN_3
    OLLAMA_CONTEXT_WINDOW = 8192
 
    class UI:
        ALLOWED_FILE_TYPES = ["pdf", "txt", "md"]