from config import Config, ModelConfig, ModelProvider
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

def create_llm(model_config: ModelConfig) -> BaseChatModel:
    if model_config.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            verbose=False,
            keep_alive=-1,
        )
    elif model_config.provider == ModelProvider.GROQ:
        return ChatGroq(model=model_config.name, temperature=model_config.temperature)
    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")
