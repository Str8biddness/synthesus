#!/bin/bash
# download_models.sh - Download required model files for Synthesus 2.0
# Usage: bash download_models.sh

set -e

MODELS_DIR="models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Synthesus 2.0 Model Downloader${NC}"
echo "================================"
echo ""

mkdir -p "$MODELS_DIR"

if command -v wget &> /dev/null; then
    DOWNLOADER="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl -L -o"
else
    echo -e "${RED}Error: Neither wget nor curl found.${NC}"
    exit 1
fi

echo -e "${YELLOW}Downloading TinyLlama GGUF (638 MB)...${NC}"
TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_FILE="$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ -f "$TINYLLAMA_FILE" ]; then
    echo -e "${GREEN}[SKIP] TinyLlama already exists${NC}"
else
    $DOWNLOADER "$TINYLLAMA_FILE" "$TINYLLAMA_URL"
    echo -e "${GREEN}[DONE] TinyLlama downloaded${NC}"
fi

echo ""
echo -e "${YELLOW}Downloading Piper TTS voice (~116 MB)...${NC}"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
PIPER_ONNX="$MODELS_DIR/en_US-ryan-high.onnx"
PIPER_JSON="$MODELS_DIR/en_US-ryan-high.onnx.json"

if [ -f "$PIPER_ONNX" ] && [ -f "$PIPER_JSON" ]; then
    echo -e "${GREEN}[SKIP] Piper voice files already exist${NC}"
else
    $DOWNLOADER "$PIPER_ONNX" "$PIPER_BASE/en_US-ryan-high.onnx"
    $DOWNLOADER "$PIPER_JSON" "$PIPER_BASE/en_US-ryan-high.onnx.json"
    echo -e "${GREEN}[DONE] Piper voice files downloaded${NC}"
fi

echo ""
echo -e "${GREEN}All models downloaded!${NC}"
echo "  TinyLlama: $TINYLLAMA_FILE"
echo "  Piper ONNX: $PIPER_ONNX"
echo ""
echo -e "${YELLOW}Note: Whisper tiny.en (~75MB) auto-downloads via pywhispercpp on first use.${NC}"
echo ""
echo -e "${GREEN}Ready! Run: bash build.sh --rebuild${NC}"
