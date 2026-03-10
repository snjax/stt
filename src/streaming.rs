use std::time::{Duration, Instant};

use anyhow::Result;

use crate::whisper_cpp::WhisperCppTranscriber;

const SAMPLE_RATE: usize = 16_000;
/// Sliding window size in seconds for streaming inference
const WINDOW_SECONDS: usize = 10;
/// Maximum window size (Whisper trained on 30s chunks)
const MAX_WINDOW_SECONDS: usize = 30;
/// Minimum interval between streaming inference calls
const MIN_CHUNK_INTERVAL: Duration = Duration::from_secs(5);
/// Minimum audio length to trigger inference (1 second)
const MIN_AUDIO_SAMPLES: usize = SAMPLE_RATE;

pub struct StreamingEngine {
    transcriber: WhisperCppTranscriber,
    audio_buffer: Vec<f32>,
    partial_text: String,
    last_inference: Option<Instant>,
}

impl StreamingEngine {
    pub fn new(transcriber: WhisperCppTranscriber) -> Self {
        Self {
            transcriber,
            audio_buffer: Vec::new(),
            partial_text: String::new(),
            last_inference: None,
        }
    }

    /// Append samples to the buffer without running inference.
    pub fn append_samples(&mut self, new_samples: &[f32]) {
        self.audio_buffer.extend_from_slice(new_samples);
    }

    /// Push new audio samples and optionally get a partial transcription.
    /// Returns Some(text) if enough time has passed and inference was run.
    pub fn push_samples(&mut self, new_samples: &[f32]) -> Option<String> {
        self.audio_buffer.extend_from_slice(new_samples);

        // Don't run inference too frequently
        if let Some(last) = self.last_inference
            && last.elapsed() < MIN_CHUNK_INTERVAL
        {
            return None;
        }

        // Need at least 1 second of audio
        if self.audio_buffer.len() < MIN_AUDIO_SAMPLES {
            return None;
        }

        // Sliding window: take last WINDOW_SECONDS of audio
        let window_samples = SAMPLE_RATE * WINDOW_SECONDS;
        let start = self.audio_buffer.len().saturating_sub(window_samples);
        let window = &self.audio_buffer[start..];

        self.last_inference = Some(Instant::now());

        match self.transcriber.transcribe_samples_streaming(window, true) {
            Ok(text) => {
                self.partial_text = text.clone();
                Some(text)
            }
            Err(e) => {
                eprintln!("streaming inference error: {e}");
                None
            }
        }
    }

    /// Finalize: run inference on the full audio buffer for best quality.
    pub fn finalize(&mut self) -> Result<String> {
        eprintln!("[finalize] audio_buffer len: {} ({:.1}s)", self.audio_buffer.len(), self.audio_buffer.len() as f64 / SAMPLE_RATE as f64);
        if self.audio_buffer.is_empty() {
            eprintln!("[finalize] buffer empty, returning empty string");
            return Ok(String::new());
        }

        // For final pass, use max window or full buffer, whichever is smaller
        let max_samples = SAMPLE_RATE * MAX_WINDOW_SECONDS;
        let samples = if self.audio_buffer.len() > max_samples {
            // For very long recordings, process in chunks and concatenate
            self.transcribe_long_audio()
        } else {
            self.transcriber.transcribe_samples(&self.audio_buffer)
        };

        // Reset state for next recording
        self.audio_buffer.clear();
        self.partial_text.clear();
        self.last_inference = None;

        samples
    }

    /// Handle audio longer than 30 seconds by splitting into overlapping chunks.
    fn transcribe_long_audio(&self) -> Result<String> {
        let chunk_size = SAMPLE_RATE * MAX_WINDOW_SECONDS;
        let step_size = SAMPLE_RATE * (MAX_WINDOW_SECONDS - 2); // 2s overlap
        let mut result = String::new();
        let mut offset = 0;

        while offset < self.audio_buffer.len() {
            let end = (offset + chunk_size).min(self.audio_buffer.len());
            let chunk = &self.audio_buffer[offset..end];

            match self.transcriber.transcribe_samples(chunk) {
                Ok(text) => {
                    if !result.is_empty() && !text.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&text);
                }
                Err(e) => {
                    eprintln!("chunk transcription error at offset {offset}: {e}");
                }
            }

            offset += step_size;
        }

        Ok(result)
    }

    /// Reset the engine, discarding all buffered audio.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.partial_text.clear();
        self.last_inference = None;
    }
}
