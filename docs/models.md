# Model Guide

## GigaAM-v3 (RNN-Transducer)

**Source:** [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) (branch `e2e_rnnt`)

### Architecture

RNN-Transducer (RNN-T) with a Conformer encoder:

- **Encoder**: Conformer blocks → 768-dimensional frame embeddings
- **Decoder**: LSTM (hidden size 320) predicts next token from previous label
- **Joint network**: Combines encoder and decoder outputs → vocabulary logits
- **Tokenizer**: SentencePiece (1024 tokens + 1 blank)

### Export to ONNX

```bash
python3 scripts/export_onnx.py --output-dir models/onnx
```

This produces 4 files in `models/onnx/`:

| File | Description | Size |
|------|-------------|------|
| `v3_e2e_rnnt_encoder.onnx` | Conformer encoder | ~30 MB |
| `v3_e2e_rnnt_encoder.onnx.data` | Encoder weights (external) | ~900 MB |
| `decoder.onnx` | LSTM decoder | ~2.5 MB |
| `joint.onnx` | Joint network | ~3 MB |
| `tokenizer.model` | SentencePiece vocabulary | ~500 KB |

### ONNX Session Inputs/Outputs

**Encoder:**
- Input: `audio_signal` [1, 64, T] (mel spectrogram)
- Input: `length` [1] (number of mel frames)
- Output: `encoded` [1, T', 768]
- Output: `encoded_len` [1]

**Decoder:**
- Input: `targets` [1, 1] (previous token ID, i64)
- Input: `target_length` [1] (always 1)
- Input: `states_0` [1, 1, 320] (LSTM hidden state h)
- Input: `states_1` [1, 1, 320] (LSTM cell state c)
- Output: `dec` [1, 1, 320]
- Output: `states_0_out` [1, 1, 320]
- Output: `states_1_out` [1, 1, 320]

**Joint:**
- Input: `enc` [1, 768, 1] (single encoder frame, transposed)
- Input: `dec` [1, 320, 1] (decoder output, transposed)
- Output: `logits` [1, 1, 1, 1025] (vocabulary + blank)

### Language Support

GigaAM-v3 is primarily trained on Russian speech data. It produces reasonable
results on English but is not optimized for it.

---

## Whisper Large V3 Turbo (MLX)

**Source:** [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-whisper
```

The model weights are automatically downloaded from HuggingFace on first use
and cached in `~/.cache/huggingface/hub/`.

### Architecture

Encoder-decoder Transformer (attention-based):
- ~800M parameters (turbo variant, distilled)
- 128 mel bins (internal, handled by mlx-whisper)
- 30-second audio segments
- Multilingual: 99 languages including Russian and English

### Integration

The Rust app spawns `scripts/whisper_server.py` as a subprocess
and communicates via JSON-RPC over stdin/stdout:

```
→ {"method": "init", "params": {"model": "mlx-community/whisper-large-v3-turbo"}}
← {"result": {"status": "ready"}}

→ {"method": "transcribe", "params": {"audio_path": "/tmp/audio.wav"}}
← {"result": {"text": "transcribed text", "language": "ru"}}

→ {"method": "shutdown", "params": {}}
← {"result": {"status": "bye"}}
```

### Language Support

Whisper supports automatic language detection and handles both Russian
and English with high accuracy. It generally outperforms GigaAM on English
and performs comparably on Russian.
