use std::time::{Duration, Instant};

use anyhow::Result;

use crate::whisper_cpp::WhisperCppTranscriber;

const SAMPLE_RATE: usize = 16_000;
/// Chunk size for committed segments (30 seconds)
const CHUNK_SAMPLES: usize = SAMPLE_RATE * 30;
/// Overlap between chunks for context (5 seconds)
const OVERLAP_SAMPLES: usize = SAMPLE_RATE * 5;
/// Minimum interval between partial inference calls
const MIN_PARTIAL_INTERVAL: Duration = Duration::from_secs(5);
/// Minimum uncommitted audio to trigger partial inference (2 seconds)
const MIN_PARTIAL_SAMPLES: usize = SAMPLE_RATE * 2;

pub struct StreamingEngine {
    transcriber: WhisperCppTranscriber,
    audio_buffer: Vec<f32>,
    /// Accumulated text from fully committed chunks
    confirmed_text: String,
    /// Number of samples covered by confirmed_text
    committed_samples: usize,
    partial_text: String,
    last_partial_inference: Option<Instant>,
}

impl StreamingEngine {
    pub fn new(transcriber: WhisperCppTranscriber) -> Self {
        Self {
            transcriber,
            audio_buffer: Vec::new(),
            confirmed_text: String::new(),
            committed_samples: 0,
            partial_text: String::new(),
            last_partial_inference: None,
        }
    }

    /// Append samples to the buffer without running inference.
    pub fn append_samples(&mut self, new_samples: &[f32]) {
        self.audio_buffer.extend_from_slice(new_samples);
    }

    /// Push new audio samples and optionally get a partial transcription.
    pub fn push_samples(&mut self, new_samples: &[f32]) -> Option<String> {
        self.audio_buffer.extend_from_slice(new_samples);

        // Commit any full 30s chunks
        self.commit_ready_chunks();

        // Rate-limit partial inference on the tail
        if let Some(last) = self.last_partial_inference {
            if last.elapsed() < MIN_PARTIAL_INTERVAL {
                return None;
            }
        }

        let uncommitted = self.audio_buffer.len().saturating_sub(self.committed_samples);
        if uncommitted < MIN_PARTIAL_SAMPLES {
            if !self.confirmed_text.is_empty() && self.partial_text.is_empty() {
                self.partial_text = self.confirmed_text.clone();
                return Some(self.partial_text.clone());
            }
            return None;
        }

        self.last_partial_inference = Some(Instant::now());

        let tail_text = self.transcribe_tail();
        let text = join_texts(&self.confirmed_text, &tail_text);
        self.partial_text = text.clone();
        Some(text)
    }

    /// Commit any complete 30s chunks to confirmed_text using timestamp-based merging.
    fn commit_ready_chunks(&mut self) {
        while self.audio_buffer.len().saturating_sub(self.committed_samples) >= CHUNK_SAMPLES {
            let overlap = if self.committed_samples > 0 { OVERLAP_SAMPLES } else { 0 };
            let chunk_start = self.committed_samples.saturating_sub(overlap);
            let chunk_end = self.committed_samples + CHUNK_SAMPLES;
            let chunk = &self.audio_buffer[chunk_start..chunk_end];
            let overlap_ms = (overlap as i64 * 1000) / SAMPLE_RATE as i64;

            eprintln!(
                "[streaming] committing chunk {:.0}s-{:.0}s (overlap {:.0}s, cutoff {}ms)",
                self.committed_samples as f64 / SAMPLE_RATE as f64,
                chunk_end as f64 / SAMPLE_RATE as f64,
                overlap as f64 / SAMPLE_RATE as f64,
                overlap_ms,
            );

            match self.transcriber.transcribe_samples_timed(chunk) {
                Ok(segments) => {
                    // Keep only segments starting after the overlap region
                    let new_text = collect_after_ms(&segments, overlap_ms);
                    self.confirmed_text = join_texts(&self.confirmed_text, &new_text);
                    eprintln!(
                        "[streaming] committed: {} chars confirmed",
                        self.confirmed_text.len()
                    );
                }
                Err(e) => {
                    eprintln!("chunk commit error: {e}");
                }
            }
            self.committed_samples += CHUNK_SAMPLES;
        }
    }

    /// Transcribe the uncommitted tail (with overlap for context), return only new text.
    fn transcribe_tail(&self) -> String {
        let overlap = if self.committed_samples > 0 { OVERLAP_SAMPLES } else { 0 };
        let tail_start = self.committed_samples.saturating_sub(overlap);
        let tail = &self.audio_buffer[tail_start..];
        let overlap_ms = (overlap as i64 * 1000) / SAMPLE_RATE as i64;

        eprintln!(
            "[streaming] partial tail: {:.1}s-{:.1}s ({:.1}s, cutoff {}ms)",
            tail_start as f64 / SAMPLE_RATE as f64,
            self.audio_buffer.len() as f64 / SAMPLE_RATE as f64,
            tail.len() as f64 / SAMPLE_RATE as f64,
            overlap_ms,
        );

        match self.transcriber.transcribe_samples_timed(tail) {
            Ok(segments) => collect_after_ms(&segments, overlap_ms),
            Err(e) => {
                eprintln!("streaming partial error: {e}");
                String::new()
            }
        }
    }

    /// Finalize: process remaining tail, merge with confirmed text.
    pub fn finalize(&mut self) -> Result<String> {
        self.commit_ready_chunks();

        let uncommitted = self.audio_buffer.len().saturating_sub(self.committed_samples);
        eprintln!(
            "[finalize] total={:.1}s, committed={:.1}s, tail={:.1}s",
            self.audio_buffer.len() as f64 / SAMPLE_RATE as f64,
            self.committed_samples as f64 / SAMPLE_RATE as f64,
            uncommitted as f64 / SAMPLE_RATE as f64,
        );

        let result = if uncommitted > 0 {
            let tail_text = self.transcribe_tail();
            join_texts(&self.confirmed_text, &tail_text)
        } else {
            self.confirmed_text.clone()
        };

        self.audio_buffer.clear();
        self.confirmed_text.clear();
        self.committed_samples = 0;
        self.partial_text.clear();
        self.last_partial_inference = None;

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.confirmed_text.clear();
        self.committed_samples = 0;
        self.partial_text.clear();
        self.last_partial_inference = None;
    }
}

/// Collect text from segments whose start time is at or after `cutoff_ms`.
fn collect_after_ms(segments: &[crate::whisper_cpp::TimedSegment], cutoff_ms: i64) -> String {
    let mut text = String::new();
    for seg in segments {
        if seg.t0_ms >= cutoff_ms {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&seg.text);
        }
    }
    text
}

/// Join two text fragments, adding a space separator if both are non-empty.
fn join_texts(a: &str, b: &str) -> String {
    if a.is_empty() {
        b.to_owned()
    } else if b.is_empty() {
        a.to_owned()
    } else {
        format!("{} {}", a, b)
    }
}
