# STT — Speech-to-Text Desktop App

Cross-platform (macOS / Linux) speech-to-text application with a minimal always-on-top GUI.
Uses **whisper.cpp** (C FFI) with the Whisper Large V3 Turbo model for multilingual transcription.

Supports streaming: partial results appear while you speak.

## Quick Start

### Prerequisites

- **Rust** (edition 2024)
- **CMake** (for building whisper.cpp)
- whisper.cpp is included as a git submodule

### Build & Run

```bash
# Initialize submodule
git submodule update --init --recursive

# Build
cargo build --release

# Run GUI
./target/release/stt

# Transcribe a file
./target/release/stt --file audio.wav
```

### Model

Place a GGML whisper model at `models/ggml-large-v3-turbo.bin`.

You can convert from a locally cached MLX model or download one:

```bash
# Download official GGML model
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Or set `STT_WHISPER_MODEL=/path/to/model.bin` to use a custom path.

## GUI Controls

- **Record / Stop** — toggle audio capture
- **Copy** — copy transcription to clipboard
- **Pin / Unpin** — toggle always-on-top window mode
- **Auto-copy to clipboard** — when enabled, transcription is automatically copied after recognition

## Architecture

See [docs/architecture.md](docs/architecture.md) for details.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `STT_WHISPER_MODEL` | Path to GGML whisper model file |

## License

Private project.
