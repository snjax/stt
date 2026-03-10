# Setup Guide

## macOS (Apple Silicon)

### 1. Install ONNX Runtime

```bash
brew install onnxruntime
```

This installs `libonnxruntime.dylib` to `/opt/homebrew/lib/`, which the app
discovers automatically. Override with `ORT_DYLIB_PATH` if needed.

### 2. Export GigaAM Models

```bash
# Create a Python venv for export scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

# Export ONNX models
python3 scripts/export_onnx.py
```

Models are saved to `models/onnx/` (~950 MB total).

### 3. (Optional) Whisper Backend

```bash
# In the same venv or a new one:
pip install mlx-whisper
```

The model weights (~3 GB) are downloaded automatically on first use.

### 4. Build and Run

```bash
cargo build --release
./target/release/stt              # GigaAM GUI
./target/release/stt --whisper    # Whisper GUI
```

## macOS (Intel)

Same as Apple Silicon, but:
- ORT path: `/usr/local/lib/libonnxruntime.dylib`
- MLX (Whisper) is not supported on Intel Macs — use GigaAM only

## Linux

### 1. Install ONNX Runtime

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
and place `libonnxruntime.so` in `/usr/lib/` or set `ORT_DYLIB_PATH`.

For CUDA support, use the GPU variant (`onnxruntime-gpu`).

### 2. Audio Dependencies

```bash
# Debian/Ubuntu
sudo apt install libasound2-dev

# Fedora
sudo dnf install alsa-lib-devel
```

### 3. Export and Build

Same as macOS:

```bash
python3 scripts/export_onnx.py
cargo build --release
./target/release/stt
```

Note: Whisper MLX backend is macOS-only. On Linux, use GigaAM.

## Download Test Audio

```bash
python3 scripts/download_test_audio.py
```

Downloads `test_ru.wav` and `test_en.wav` to `test_data/`.

## Verify Inference

```bash
python3 scripts/test_inference.py
```

Compares PyTorch and ONNX inference results on test audio files.

## Troubleshooting

### "mlx_whisper import failed: No module named 'mlx_whisper'"

The app couldn't find a Python with mlx-whisper installed. Solutions:
1. Create `.venv` in the project root: `python3 -m venv .venv && .venv/bin/pip install mlx-whisper`
2. Or set `PYTHON=/path/to/python-with-mlx-whisper`

### "unable to find libonnxruntime"

Set `ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib` (or `.so` on Linux).

### "unable to find models"

Set `STT_MODEL_DIR=/path/to/models/onnx` or run from the project root.

### "Protobuf parsing failed"

Ensure `ort` crate is pinned to `=2.0.0-rc.10` in Cargo.toml. Later versions
have a broken static library.
