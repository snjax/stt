use std::time::{Duration, Instant};

use anyhow::Result;

use crate::whisper_cpp::WhisperCppTranscriber;

const SAMPLE_RATE: usize = 16_000;
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
    /// Runs full-buffer inference to produce accumulated text (not just last window).
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

        self.last_inference = Some(Instant::now());

        // Run inference on the FULL accumulated buffer so partial_text
        // contains everything transcribed so far, not just the last window.
        match self.transcriber.transcribe_samples(&self.audio_buffer) {
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
        eprintln!(
            "[finalize] audio_buffer len: {} ({:.1}s)",
            self.audio_buffer.len(),
            self.audio_buffer.len() as f64 / SAMPLE_RATE as f64
        );
        if self.audio_buffer.is_empty() {
            eprintln!("[finalize] buffer empty, returning empty string");
            return Ok(String::new());
        }

        let result = self.transcriber.transcribe_samples(&self.audio_buffer);

        // Reset state for next recording
        self.audio_buffer.clear();
        self.partial_text.clear();
        self.last_inference = None;

        result
    }

    /// Reset the engine, discarding all buffered audio.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.partial_text.clear();
        self.last_inference = None;
    }
}
