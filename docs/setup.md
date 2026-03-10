# Setup Guide

## macOS (Apple Silicon)

### 1. Build

```bash
git submodule update --init --recursive
cargo build --release
```

CMake and a C++ compiler are required (Xcode Command Line Tools).
whisper.cpp is built automatically via `build.rs` with Metal GPU support.

### 2. Model

Place a GGML model in `models/`:

```bash
mkdir -p models
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Or set `STT_WHISPER_MODEL=/path/to/model.bin`.

### 3. Run

```bash
./target/release/stt              # GUI
./target/release/stt --file a.wav # File mode
```

## Linux

### 1. Audio Dependencies

```bash
# Debian/Ubuntu
sudo apt install libasound2-dev cmake

# Fedora
sudo dnf install alsa-lib-devel cmake
```

### 2. Build and Run

```bash
git submodule update --init --recursive
cargo build --release
./target/release/stt
```

For CUDA GPU acceleration, build with `cargo build --release --features cuda`.

## Download Test Audio

```bash
python3 scripts/download_test_audio.py
```

Downloads `test_ru.wav` and `test_en.wav` to `test_data/`.

## Troubleshooting

### "unable to find whisper GGML model"

Download the model or set `STT_WHISPER_MODEL=/path/to/model.bin`.

### Build fails with CMake errors

Ensure CMake is installed and whisper.cpp submodule is initialized:
```bash
git submodule update --init --recursive
```
