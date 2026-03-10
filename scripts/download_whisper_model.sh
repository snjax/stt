#!/bin/bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
MODEL_FILE="$MODEL_DIR/ggml-large-v3-turbo.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
    exit 0
fi

mkdir -p "$MODEL_DIR"

echo "Downloading ggml-large-v3-turbo.bin (~1.5 GB)..."
curl -L -o "$MODEL_FILE" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"

echo "Done: $MODEL_FILE"
