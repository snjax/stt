# STT — Speech-to-Text Desktop App

Cross-platform (macOS / Linux) speech-to-text application with a minimal always-on-top GUI.
Supports two inference backends:

| Backend | Model | Acceleration |
|---------|-------|-------------|
| **GigaAM-v3** | `ai-sage/GigaAM-v3` (RNN-Transducer, ~240M params) | CoreML (macOS) / CUDA (Linux) via ONNX Runtime |
| **Whisper** | `mlx-community/whisper-large-v3-turbo` | Apple MLX GPU via `mlx-whisper` Python subprocess |

GigaAM is optimized for Russian; Whisper handles both Russian and English well.

## Quick Start

### Prerequisites

- **Rust** (edition 2024)
- **ONNX Runtime** — installed via Homebrew (`brew install onnxruntime`) or system package
- **Python 3** with a virtual environment (for Whisper backend only)

### Build & Run

```bash
# Export GigaAM ONNX models (one-time)
python3 scripts/export_onnx.py

# Build
cargo build --release

# Run GUI (GigaAM, default)
./target/release/stt

# Run GUI (Whisper)
./target/release/stt --whisper

# Transcribe a file
./target/release/stt --file audio.wav
./target/release/stt --whisper --file audio.wav
```

### Whisper Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-whisper
```

The app automatically discovers `.venv/bin/python` relative to the project directory.
Override with `PYTHON=/path/to/python` if needed.

## GUI Controls

- **Record / Stop** — toggle audio capture
- **Copy** — copy transcription to clipboard
- **Pin / Unpin** — toggle always-on-top window mode
- **Auto-copy to clipboard** — checkbox; when enabled, transcription is automatically copied after recognition

## Architecture

See [docs/architecture.md](docs/architecture.md) for details.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.dylib` / `.so` |
| `STT_MODEL_DIR` | Directory containing ONNX models and tokenizer |
| `STT_WHISPER_SERVER` | Path to `whisper_server.py` |
| `PYTHON` | Python interpreter for Whisper backend |

## License

Private project.
