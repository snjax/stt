# Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────┐
│                    GUI (egui/eframe)                      │
│  ┌──────┐  ┌──────┐  ┌───────┐  ┌───────────────────┐   │
│  │Record│  │ Copy │  │Pin/Un │  │☑ Auto-clipboard   │   │
│  └──┬───┘  └──────┘  └───────┘  └───────────────────┘   │
│     │         ▲                                          │
│     │    transcription                                   │
│     ▼         │                                          │
│  ┌──────────────────┐    mpsc     ┌──────────────────┐   │
│  │  AudioRecorder   │───────────▶│  Worker Thread    │   │
│  │  (cpal, 16kHz)   │  samples   │  (lazy init)     │   │
│  └──────────────────┘            └────────┬─────────┘   │
│                                           │              │
└───────────────────────────────────────────┼──────────────┘
                                            │
                        ┌───────────────────┼───────────────┐
                        │                   │               │
                   ┌────▼────┐        ┌─────▼─────┐
                   │ GigaAM  │        │  Whisper   │
                   │  (ONNX) │        │(mlx-whisper│
                   │         │        │ subprocess)│
                   └─────────┘        └───────────┘
```

## Data Flow

### Recording Pipeline

1. **cpal** opens the default input device at the best available config
2. Audio samples accumulate in a thread-safe `Arc<Mutex<Vec<f32>>>` buffer
3. On stop, the buffer is drained, downmixed to mono, and resampled to 16 kHz
4. Samples are sent to the worker thread via `mpsc::channel`

### GigaAM Inference Pipeline

```
16 kHz f32 samples
    │
    ▼
┌─────────────────────┐
│ Mel Spectrogram     │  rustfft, 64 mel bins, hop=160, win=400
│ [1, 64, T]          │  center padding, Hann window, log scale
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Encoder (ONNX)      │  Conformer → [1, T', 768]
│ CoreML / CUDA EP    │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Greedy Decode       │  For each encoder frame:
│ Decoder + Joint     │    Decoder LSTM (prev token, h, c)
│ (ONNX)              │    Joint network → logits → argmax
│                     │    Emit token or advance frame
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ SentencePiece       │  token IDs → text
│ Tokenizer           │
└─────────────────────┘
```

**Key parameters:**

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz |
| Mel bins | 64 |
| FFT size | 400 |
| Hop length | 160 |
| Window | 400 (Hann) |
| Encoder dim | 768 |
| Decoder dim (LSTM hidden) | 320 |
| Blank token ID | 1024 (= vocab size) |
| Max symbols per step | 10 |

### Whisper Inference Pipeline

```
16 kHz f32 samples
    │
    ▼
┌──────────────────────┐
│ Write temp WAV       │  /tmp/stt-whisper-{pid}-{nanos}.wav
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Python subprocess    │  JSON-RPC over stdin/stdout
│ whisper_server.py    │
│                      │
│  mlx_whisper.transcribe(path, model)
│  → {"result": {"text": "...", "language": "en"}}
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Cleanup temp WAV     │
└──────────────────────┘
```

The Python subprocess stays alive for the lifetime of `WhisperTranscriber`,
keeping the model loaded in MLX GPU memory between transcriptions.

## Module Map

| Module | File | Purpose |
|--------|------|---------|
| `main` | `src/main.rs` | CLI entry point, argument parsing |
| `app` | `src/app.rs` | GUI, worker thread, state management |
| `audio` | `src/audio.rs` | Audio capture (cpal), WAV loading (hound), resampling |
| `inference` | `src/inference.rs` | GigaAM ONNX inference, ORT initialization, RNN-T decoding |
| `mel` | `src/mel.rs` | Mel spectrogram computation (rustfft) |
| `tokenizer` | `src/tokenizer.rs` | SentencePiece wrapper |
| `whisper` | `src/whisper.rs` | Whisper subprocess manager, JSON-RPC client |

## ONNX Runtime Loading

ORT is loaded dynamically at runtime (`load-dynamic` feature):

1. `$ORT_DYLIB_PATH` environment variable
2. `/opt/homebrew/lib/libonnxruntime.dylib` (Apple Silicon Homebrew)
3. `/usr/local/lib/libonnxruntime.dylib` (Intel Mac Homebrew)
4. `/usr/lib/libonnxruntime.so` and `/usr/lib/x86_64-linux-gnu/libonnxruntime.so` (Linux)

Execution providers:
- **macOS**: CoreML (GPU/ANE acceleration)
- **Linux**: CUDA (GPU acceleration)

## Threading Model

```
Main thread (UI)
    │
    ├── egui event loop (16ms repaint during recording/transcribing)
    │
    └── mpsc channel ──▶ Worker thread
                            │
                            ├── Lazy-initialized transcriber
                            │   (GigaAM sessions or Whisper subprocess)
                            │
                            └── Blocks on command_rx.recv()
                                Sends results via event_tx
```

The worker thread is spawned once at app startup. The transcriber is created
lazily on the first transcription request, so the app starts instantly.
