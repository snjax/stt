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
                                     ┌──────▼──────┐
                                     │ whisper.cpp  │
                                     │  (C FFI)     │
                                     │  Metal/CPU   │
                                     └─────────────┘
```

## Data Flow

### Recording Pipeline

1. **cpal** opens the default input device at the best available config
2. Audio samples accumulate in a thread-safe `Arc<Mutex<Vec<f32>>>` buffer
3. Every ~33ms, new samples are drained and sent to the worker thread via `mpsc::channel`
4. Worker feeds samples into `StreamingEngine` which accumulates audio and periodically runs partial inference

### Streaming Inference

During recording, `StreamingEngine` runs partial whisper inference on a sliding 10-second window
every 5 seconds, sending partial results back to the GUI (shown in gray italic).

### Finalization

On stop:
1. All pending audio samples are flushed to the worker
2. Worker drains any queued `PushSamples` commands (appending audio without inference)
3. A single final inference runs on the full audio buffer (up to 30s; longer audio is chunked with 2s overlap)
4. Result is sent back to the GUI

### whisper.cpp Integration

The app links whisper.cpp as a static library via CMake (build.rs).
Raw C FFI bindings call `whisper_full()` directly — no whisper-rs crate.

GPU acceleration:
- **macOS**: Metal (automatic on Apple Silicon)
- **Linux**: CUDA (with `cuda` feature)

## Module Map

| Module | File | Purpose |
|--------|------|---------|
| `main` | `src/main.rs` | CLI entry point, argument parsing |
| `app` | `src/app.rs` | GUI, worker thread, state management |
| `audio` | `src/audio.rs` | Audio capture (cpal), WAV loading (hound), resampling |
| `streaming` | `src/streaming.rs` | Sliding window streaming engine |
| `whisper_cpp` | `src/whisper_cpp.rs` | whisper.cpp FFI bindings, model loading |

## Threading Model

```
Main thread (UI)
    │
    ├── egui event loop (~30fps repaint during recording/transcribing)
    │
    └── mpsc channel ──▶ Worker thread
                            │
                            ├── Lazy-initialized StreamingEngine
                            │   (wraps WhisperCppTranscriber)
                            │
                            └── Blocks on command_rx.recv()
                                Sends results via event_tx
```

The worker thread is spawned once at app startup. The transcriber is created
lazily on the first transcription request, so the app starts instantly.
