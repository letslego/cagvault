#!/bin/bash
# Download all CAGVault RAG-optimized models
# Run this script to pull all supported models

echo "🚀 Downloading CAGVault RAG Models"
echo "===================================="
echo ""
echo "⚠️  This will download ~200GB+ of models"
echo "⚠️  Estimated time: 2-4 hours depending on your connection"
echo "⚠️  Press Ctrl+C to cancel"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "📦 Downloading lightweight models (8GB RAM)..."
echo "----------------------------------------------"
ollama pull llama3.1:8b
ollama pull mistral-small-latest

echo ""
echo "📦 Downloading medium models (16GB RAM)..."
echo "-------------------------------------------"
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL
ollama pull phi4:latest

echo ""
echo "📦 Downloading large models (32GB+ RAM)..."
echo "-------------------------------------------"
ollama pull gemma2:27b
ollama pull llama3.3:70b
ollama pull mistral-large-latest
ollama pull command-r-plus:latest

echo ""
echo "📦 Downloading state-of-the-art models (64GB+ RAM)..."
echo "-------------------------------------------------------"
ollama pull deepseek-ai/DeepSeek-V3
ollama pull deepseek-ai/DeepSeek-R1

echo ""
echo "✅ All models downloaded successfully!"
echo ""
echo "To use a model, either:"
echo "  1. Select it from the UI sidebar (Model Settings)"
echo "  2. Edit config.py and change Config.MODEL"
echo ""
echo "Installed models:"
ollama list
