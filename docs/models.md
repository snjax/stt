# Model Guide

## Whisper Large V3 Turbo

**Source:** [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

### Architecture

Encoder-decoder Transformer:
- ~800M parameters (turbo variant, distilled from large-v3)
- 32 encoder layers, 4 decoder layers
- 1280-dimensional hidden state, 20 attention heads
- 128 mel bins, 30-second audio segments
- Multilingual: 100 languages including Russian and English

### GGML Format

The app uses whisper.cpp's GGML binary format. Model files contain:
- Hyperparameters (vocab size, layer counts, dimensions)
- Mel filterbank weights
- Tokenizer vocabulary (GPT-2 byte-level BPE)
- Model tensors (f16 for weights, f32 for biases and embeddings)

### Supported Models

The app searches for models in this order:
1. `ggml-large-v3-turbo.bin`
2. `ggml-large-v3-turbo-q5_0.bin`
3. `ggml-large-v3.bin`
4. `ggml-base.bin`

### Download

```bash
# Large V3 Turbo (~1.5 GB, recommended)
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

### Language Support

Whisper supports automatic language detection and handles both Russian
and English with high accuracy. Language is auto-detected per inference call.
