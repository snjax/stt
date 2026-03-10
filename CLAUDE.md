# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Build (CPU, default)
cargo build --release

# Build with Metal GPU (macOS)
cargo build --release --features metal

# Build with CUDA GPU (Linux)
cargo build --release --features cuda

# Run GUI (GigaAM backend, default)
./target/release/stt

# Run GUI (Whisper backend with streaming)
./target/release/stt --whisper

# Transcribe a file directly
./target/release/stt --file audio.wav
./target/release/stt --whisper --file audio.wav
```

No test suite or linter is configured. Use `cargo check` and `cargo clippy` for validation.

## One-time Setup

### Whisper (whisper.cpp)
Download a GGML model:
```bash
bash scripts/download_whisper_model.sh
# Or manually:
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

### GigaAM
GigaAM requires ONNX model export and ONNX Runtime library:
```bash
pip install -r scripts/requirements.txt
python3 scripts/export_onnx.py          # exports to models/onnx/
```

## Environment Variables

- `ORT_DYLIB_PATH` — Path to libonnxruntime shared library (auto-discovered on common paths)
- `STT_MODEL_DIR` — Custom ONNX model directory (default: auto-discover `models/onnx/`)
- `STT_WHISPER_MODEL` — Custom path to whisper GGML model file

## Architecture

Cross-platform (macOS/Linux) desktop STT app with two inference backends.

**Threading model:** Main thread runs egui GUI; a lazy-spawned worker thread handles inference. They communicate via `mpsc` channels. Whisper backend supports streaming (partial results during recording).

**Dual backends:**
- **GigaAM-v3** (default): Pure Rust inference pipeline. Audio → mel spectrogram (rustfft) → ONNX Runtime (encoder/decoder/joint) → greedy RNN-T decode → SentencePiece detokenize. Russian-optimized. Uses CoreML on macOS, CUDA on Linux.
- **Whisper v3 Turbo** (`--whisper`): Native whisper.cpp via C FFI. Model stays in memory. Supports streaming with sliding window (partial results every ~500ms during recording). Auto-detects language. GPU: Metal (macOS), CUDA (Linux).

**Module responsibilities:**
- `main.rs` — CLI arg parsing, routes to GUI or file transcription
- `app.rs` — `SpeechApp` struct implementing `eframe::App`; UI controls; worker thread; streaming command protocol
- `audio.rs` — cpal-based audio capture with resampling to 16kHz mono; WAV loading; `drain_new_samples()` for streaming
- `inference.rs` — GigaAM ONNX session management, model path discovery, RNN-T greedy decode loop
- `mel.rs` — Mel spectrogram (Hann window, 64 bins, FFT size 400, hop 160, log scale)
- `tokenizer.rs` — SentencePiece wrapper; blank_id = vocab size (1024)
- `whisper_cpp.rs` — FFI bindings to whisper.cpp C API; safe Rust wrapper for WhisperContext
- `streaming.rs` — StreamingEngine: sliding window inference, chunk management, finalization

**Streaming protocol (Whisper):**
- `PushSamples(Vec<f32>)` — new audio samples from mic, triggers partial inference every ~500ms
- `Finalize` — stop recording, run full inference on complete buffer
- `PartialResult(String)` — intermediate transcription shown in gray/italic
- `Finished(Result<String>)` — final transcription

**Key constants (GigaAM):** sample rate 16kHz, 64 mel bins, encoder dim 768, decoder LSTM hidden 320, blank token 1024, max 10 symbols per RNN-T step.

**whisper.cpp:** Built as static library from `third_party/whisper.cpp` submodule via cmake in `build.rs`. Linked statically with ggml.

**ONNX Runtime:** Loaded dynamically via `ort::init_from()`. The `ort` crate is pinned to `=2.0.0-rc.10`.
