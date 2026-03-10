# PRD: Streaming Whisper via whisper.cpp

## Цель

Перейти с текущего Whisper-бэкенда (mlx-whisper через Python-subprocess) на
**whisper.cpp** через Rust-биндинги (`whisper-rs`), получив:

1. **Честный стриминг** — текст появляется пока пользователь говорит
2. **Модель всегда в памяти** — загрузка один раз при старте, без повторных инициализаций
3. **Нативный Rust** — без Python-зависимостей для Whisper
4. **GPU-ускорение** — Metal (macOS) / CUDA (Linux) через feature flags whisper-rs

---

## Текущая архитектура (as-is)

```
AudioRecorder ──▶ stop() ──▶ Vec<f32> ──▶ worker thread
                                              │
                               ┌──────────────┴──────────────┐
                               │                             │
                          GigaAM (ONNX)              Whisper (mlx-whisper)
                          Rust-native                Python subprocess
                          модель в памяти            JSON-RPC stdin/stdout
                                                     загрузка при первом вызове
```

**Проблемы текущего Whisper-бэкенда:**
- Требует Python venv с mlx-whisper
- Только macOS (MLX не работает на Linux)
- Не стриминговый — ждёт окончания записи, потом распознаёт целиком
- Отдельный процесс — overhead на IPC, temp-файлы

---

## Целевая архитектура (to-be)

```
AudioRecorder ──▶ непрерывный поток f32 ──▶ StreamingEngine
                  (ring buffer, чанки)          │
                                    ┌───────────┴───────────┐
                                    │                       │
                               GigaAM (ONNX)         WhisperCpp
                               (без изменений)       whisper-rs (Rust)
                                                     модель в памяти
                                                     чанковый inference
                                                     ▼
                                                UI: partial text updates
```

---

## Этапы реализации

### Этап 1: Интеграция whisper-rs (замена mlx-whisper)

**Цель:** заменить Python-subprocess на нативный Rust через whisper-rs,
сохранив текущий не-стриминговый UX (запись → стоп → результат).

#### 1.1 Зависимости

```toml
# Cargo.toml
[dependencies.whisper-rs]
version = "0.13"  # или последняя стабильная
features = ["metal"]  # macOS GPU; на Linux: ["cuda"]
```

Для macOS (Apple Silicon) добавить в `.cargo/config.toml`:
```toml
[target.aarch64-apple-darwin]
rustflags = ["-lc++", "-l", "framework=Accelerate"]
```

#### 1.2 Модель

Скачать GGML-модель whisper-large-v3-turbo:
```bash
# scripts/download_whisper_model.sh
curl -L -o models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Размер: ~1.5 GB. Добавить в `.gitignore` (уже покрыто `*.bin` и `/models/`).

#### 1.3 Новый модуль `src/whisper_cpp.rs`

```rust
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState,
                 FullParams, SamplingStrategy};

pub struct WhisperCppTranscriber {
    ctx: WhisperContext,   // модель в памяти — живёт пока жив struct
    state: WhisperState,   // mutable state для inference
}

impl WhisperCppTranscriber {
    pub fn new(model_path: &Path) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),  // Metal/CUDA подхватится автоматически
        )?;
        let state = ctx.create_state()?;
        Ok(Self { ctx, state })
    }

    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));  // автодетект ru/en
        params.set_no_context(true);

        self.state.full(params, samples)?;

        let mut text = String::new();
        for i in 0..self.state.full_n_segments()? {
            text.push_str(&self.state.full_get_segment_text(i)?);
        }
        Ok(text.trim().to_owned())
    }
}
```

**Ключевой момент:** `WhisperContext` владеет загруженной моделью.
Пока `WhisperCppTranscriber` жив (а он живёт в worker thread) — модель в памяти.

#### 1.4 Обновить `BackendChoice`

```rust
pub enum BackendChoice {
    GigaAm,
    WhisperCpp,      // новый, нативный
    // WhisperMlx,   // deprecated, удалить или оставить как fallback
}
```

CLI-флаг: `--whisper` → `WhisperCpp`.

#### 1.5 Удалить / deprecate

- `src/whisper.rs` (Python-subprocess клиент)
- `scripts/whisper_server.py`
- Зависимость от Python venv для Whisper

#### 1.6 Критерий готовности

- `cargo run -- --whisper --file test_data/test_ru.wav` выдаёт корректный русский текст
- `cargo run -- --whisper --file test_data/test_en.wav` выдаёт корректный английский текст
- Модель загружается один раз, повторные вызовы мгновенные
- Работает без Python

---

### Этап 2: Streaming inference (чанковое распознавание)

**Цель:** текст появляется в UI пока пользователь говорит, без ожидания нажатия Stop.

#### 2.1 Стриминговая модель аудио

Заменить текущий паттерн «запись → стоп → отправка» на непрерывный поток:

```
AudioRecorder
    │
    │  каждые ~500ms отдаёт накопленные семплы
    ▼
StreamingEngine (новый)
    │
    │  sliding window: последние N секунд аудио
    │  вызывает whisper_full() на окне
    ▼
UI: обновляет partial_text в реальном времени
    │
    │  при Stop → финальное распознавание всего буфера
    ▼
UI: заменяет partial_text на final_text
```

#### 2.2 Новый компонент: `StreamingEngine`

```rust
pub struct StreamingEngine {
    transcriber: WhisperCppTranscriber,
    audio_buffer: Vec<f32>,        // весь записанный аудио
    last_stable_len: usize,        // позиция до которой текст стабилизировался
    partial_text: String,          // текущий промежуточный результат
    chunk_interval: Duration,      // 500ms — как часто запускать inference
}

impl StreamingEngine {
    /// Вызывается каждые ~500ms с новыми семплами
    pub fn push_samples(&mut self, new_samples: &[f32]) -> Option<String> {
        self.audio_buffer.extend_from_slice(new_samples);

        // Sliding window: берём последние 10-30 секунд
        let window_samples = 16_000 * 10; // 10 секунд
        let start = self.audio_buffer.len().saturating_sub(window_samples);
        let window = &self.audio_buffer[start..];

        // Запускаем inference на окне
        if let Ok(text) = self.transcriber.transcribe_samples(window) {
            self.partial_text = text.clone();
            return Some(text);
        }
        None
    }

    /// Финальное распознавание при остановке
    pub fn finalize(&mut self) -> Result<String> {
        self.transcriber.transcribe_samples(&self.audio_buffer)
    }
}
```

#### 2.3 Изменения в `AudioRecorder`

Добавить режим непрерывного чтения буфера без остановки потока:

```rust
impl AudioRecorder {
    /// Забрать новые семплы, не останавливая запись
    pub fn drain_new_samples(&self) -> Vec<f32> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.drain(..).collect()
    }
}
```

#### 2.4 Изменения в worker thread

Заменить одноразовую команду `Transcribe(Vec<f32>)` на поточные команды:

```rust
enum InferenceCommand {
    PushSamples(Vec<f32>),    // новые семплы из микрофона
    Finalize,                  // стоп — финальное распознавание
}

enum InferenceEvent {
    PartialResult(String),     // промежуточный текст (пока говорит)
    FinalResult(String),       // финальный текст (после Stop)
    Error(String),
}
```

#### 2.5 Изменения в GUI

- Во время записи: показывать `partial_text` серым/italic шрифтом
- После Stop: показать `final_text` нормальным шрифтом
- Таймер `request_repaint_after(500ms)` для периодического drain семплов

#### 2.6 Параметры стриминга

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Chunk interval | 500 ms | Баланс latency / CPU load |
| Window size | 10 сек | Минимум для стабильного качества |
| Max window | 30 сек | Лимит Whisper (trained on 30s chunks) |
| `no_context` | true | Предотвращает каскадные ошибки |
| `single_segment` | true | Одна фраза за вызов — меньше latency |
| Sampling | Greedy(1) | Быстрее beam search |

#### 2.7 Критерий готовности

- Текст появляется в UI через ~1-2 секунды после начала речи
- При продолжении речи текст обновляется каждые ~500ms
- После Stop — финальный результат заменяет промежуточный
- Нет видимых артефактов на границах окон

---

### Этап 3: VAD (Voice Activity Detection)

**Цель:** не тратить GPU на тишину, автоматически сегментировать речь.

#### 3.1 Silero VAD

whisper.cpp поддерживает Silero VAD нативно. whisper-rs экспортирует
`WhisperVadContext` и `WhisperVadSegments`.

```bash
# Скачать модель VAD (~2 MB)
curl -L -o models/silero-v5.1.2-ggml.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-v5.1.2.bin
```

#### 3.2 Интеграция в StreamingEngine

```rust
// Перед вызовом whisper inference:
// 1. Прогнать окно через Silero VAD (~1ms на CPU)
// 2. Если речи нет — пропустить inference (экономия ~500ms GPU)
// 3. Если речь закончилась — автоматически финализировать сегмент
```

Преимущества:
- Экономия GPU: inference только когда есть речь
- Меньше галлюцинаций: Whisper не обрабатывает тишину
- Автоматическая сегментация: можно финализировать по паузам

#### 3.3 Критерий готовности

- При молчании CPU/GPU idle (нет вызовов whisper inference)
- При паузе >1.5 сек текущий сегмент автоматически финализируется
- Галлюцинации на тишине отсутствуют

---

### Этап 4: Оптимизации и polish

#### 4.1 Стабилизация текста (LocalAgreement)

Промежуточные результаты стриминга нестабильны — текст может "прыгать".
Алгоритм LocalAgreement подтверждает слово только когда оно стабилизируется
в нескольких последовательных вызовах:

```
Вызов 1: "Привет как д..."     → показать: "Привет"
Вызов 2: "Привет как дела"     → показать: "Привет как"
Вызов 3: "Привет как дела вс"  → показать: "Привет как дела"
```

Подтверждённый текст показывать нормальным шрифтом, неподтверждённый — серым.

#### 4.2 Автоматический выбор модели

```rust
// По умолчанию: large-v3-turbo (лучший баланс скорость/качество)
// Fallback: base (если GPU слабый или памяти мало)
// Настройка: --whisper-model path/to/model.bin
```

#### 4.3 Квантизация

Для устройств с ограниченной памятью — поддержка квантизованных моделей:
- `ggml-large-v3-turbo-q5_0.bin` — ~600 MB вместо 1.5 GB
- Минимальная потеря качества

---

## Модель памяти (ключевое требование)

```
Запуск приложения
    │
    ▼
WhisperContext::new()          ← модель загружена в RAM/VRAM (~1.5 GB)
    │
    ▼
┌─────────────────────────────┐
│ Worker thread               │
│                             │
│  ctx.create_state()         │  ← state ~50 MB (KV cache)
│       │                     │
│       ▼                     │
│  loop {                     │
│    state.full(params, audio)│  ← inference, модель НЕ перезагружается
│    // отдать результат      │
│  }                          │
│                             │
│  // state Drop → ~50 MB     │
│  // ctx НЕ дропается        │
└─────────────────────────────┘
    │
    ▼
Закрытие приложения
    │
    ▼
WhisperContext Drop            ← модель выгружена
```

**Гарантия:** `WhisperContext` создаётся один раз в worker thread и живёт
до завершения программы. Между вызовами inference модель остаётся в памяти.
`WhisperState` можно пересоздавать (дешёвая операция, ~50 MB KV cache).

---

## Сравнение бэкендов

| | mlx-whisper (текущий) | whisper.cpp (целевой) |
|---|---|---|
| Язык | Python subprocess | Нативный Rust (C++ через FFI) |
| Платформы | только macOS (MLX) | macOS, Linux, Windows |
| GPU | Apple MLX | Metal (macOS), CUDA (Linux), Vulkan |
| Стриминг | нет | да (sliding window) |
| VAD | нет | Silero VAD встроенный |
| Модель в памяти | да (пока жив subprocess) | да (WhisperContext) |
| Зависимости | Python venv, mlx-whisper | только whisper-rs (Cargo) |
| Размер модели | ~3 GB (HF format) | ~1.5 GB (GGML) |
| Квантизация | нет | Q5_0, Q8_0 и др. |

---

## Риски и митигации

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| whisper-rs не компилируется с Metal | низкая | feature flag `metal`, проверено community |
| Качество стриминга ниже batch | средняя | Финальный проход по всему буферу при Stop |
| Галлюцинации на тишине | средняя | Silero VAD (этап 3) |
| Большой размер модели | низкая | Квантизация Q5_0 (~600 MB) |
| Нестабильный partial text | высокая | LocalAgreement (этап 4) |
| Latency на CPU >2s | средняя | GPU обязателен; fallback на base модель |

---

## Приоритет

1. **Этап 1** — критический, убирает Python-зависимость, даёт кроссплатформенность
2. **Этап 2** — ключевая фича, streaming UX
3. **Этап 3** — качество и эффективность
4. **Этап 4** — polish, не блокирует релиз

---

## Оценка объёма

| Этап | Файлы | Описание |
|------|-------|----------|
| 1 | `Cargo.toml`, `src/whisper_cpp.rs`, `src/app.rs`, `src/main.rs` | Новый модуль, замена бэкенда |
| 2 | `src/audio.rs`, `src/streaming.rs` (новый), `src/app.rs` | Streaming engine, UI updates |
| 3 | `src/streaming.rs`, скрипт скачивания VAD модели | VAD интеграция |
| 4 | `src/streaming.rs` | LocalAgreement, polish |
