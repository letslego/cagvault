#!/bin/bash
# Quick download script for essential models

echo "🚀 Downloading Essential CAGVault Models"
echo "========================================="
echo ""
echo "This will download 3 essential models (~30GB):"
echo "  1. Qwen3-14B (Default) - 16GB RAM"
echo "  2. Llama 3.1 8B (Lightweight) - 8GB RAM"  
echo "  3. Phi-4 (Efficient) - 16GB RAM"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "📦 Downloading Qwen3-14B (Default)..."
ollama pull hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_XL

echo ""
echo "📦 Downloading Llama 3.1 8B (Lightweight)..."
ollama pull llama3.1:8b

echo ""
echo "📦 Downloading Phi-4 (Efficient)..."
ollama pull phi4:latest

echo ""
echo "✅ Essential models downloaded!"
echo ""
echo "To download more models, run: ./download_models.sh"
echo "Or download individually with: ollama pull <model-name>"
echo ""
echo "Installed models:"
ollama list
