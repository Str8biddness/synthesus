# models/

This directory holds large model files required for full Synthesus 2.0 capability.
These are NOT committed to the repo due to file size.

## Required model files

| File | Size | Purpose | Download |
|------|------|---------|----------|
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | ~638 MB | Right hemisphere SLM (llama-cpp) | [TheBloke/TinyLlama HuggingFace](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| `en_US-ryan-high.onnx` | ~116 MB | Piper TTS voice | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/high) |
| `en_US-ryan-high.onnx.json` | ~4 KB | Piper TTS config | Same as above |

## Quick download

```bash
# TinyLlama GGUF
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Piper TTS voice
wget -P models/ https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
wget -P models/ https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
```

## Running without models

Synthesus 2.0 runs in degraded mode without these files:
- **No TinyLlama**: Right hemisphere returns stub responses (Python fallback active)
- **No Piper TTS**: Speech output disabled
- All reasoning modules (SINN, PPBRS, Bayesian, etc.) still fully functional
