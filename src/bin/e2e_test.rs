//! E2E test that replicates the EXACT app worker thread model at real-time speed.
//!
//! Feeds audio at 1x real-time to trigger the same streaming inference cadence
//! as a live recording. Sends Finalize exactly like stop_recording().
//! Verifies that the Finished event contains the COMPLETE transcription.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use stt::streaming::StreamingEngine;
use stt::whisper_cpp::WhisperCppTranscriber;

enum InferenceCommand {
    PushSamples(Vec<f32>),
    Finalize,
    Shutdown,
}

enum InferenceEvent {
    Finished(Result<String, String>),
    PartialResult(String),
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: e2e_test <file.wav>");

    eprintln!("=== E2E streaming test (real-time speed) ===");
    eprintln!("Loading: {path}");

    let samples = stt::audio::load_wav_mono_16k(std::path::Path::new(&path))
        .expect("Failed to load WAV");
    let duration_s = samples.len() as f64 / 16000.0;
    eprintln!("Audio: {} samples ({:.1}s)", samples.len(), duration_s);

    // --- Worker thread (identical to app.rs) ---
    let (cmd_tx, cmd_rx) = mpsc::channel::<InferenceCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<InferenceEvent>();

    let worker = thread::spawn(move || {
        let mut engine: Option<StreamingEngine> = None;

        while let Ok(command) = cmd_rx.recv() {
            if matches!(command, InferenceCommand::Shutdown) {
                break;
            }

            if engine.is_none() {
                match WhisperCppTranscriber::new() {
                    Ok(t) => engine = Some(StreamingEngine::new(t)),
                    Err(e) => {
                        let _ = evt_tx.send(InferenceEvent::Finished(Err(e.to_string())));
                        continue;
                    }
                }
            }

            let engine = engine.as_mut().unwrap();

            match command {
                InferenceCommand::PushSamples(samples) => {
                    if let Some(text) = engine.push_samples(&samples) {
                        if evt_tx
                            .send(InferenceEvent::PartialResult(text))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
                InferenceCommand::Finalize => {
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        if let InferenceCommand::PushSamples(samples) = cmd {
                            engine.append_samples(&samples);
                        }
                    }
                    let result = engine.finalize().map_err(|e| e.to_string());
                    if evt_tx.send(InferenceEvent::Finished(result)).is_err() {
                        break;
                    }
                }
                InferenceCommand::Shutdown => unreachable!(),
            }
        }
    });

    // --- Feed audio at 1x real-time (33ms chunks = 528 samples at 16kHz) ---
    let chunk_size = 528; // ~33ms worth of audio
    let chunk_delay = Duration::from_millis(33);
    let mut offset = 0;
    let mut partial_count = 0;
    let mut last_partial = String::new();
    let start = Instant::now();

    eprintln!("\n--- Recording at 1x real-time speed ---");
    eprintln!("(This will take {:.0}s...)", duration_s);

    while offset < samples.len() {
        let end = (offset + chunk_size).min(samples.len());
        let chunk = samples[offset..end].to_vec();
        offset = end;

        cmd_tx
            .send(InferenceCommand::PushSamples(chunk))
            .expect("worker died");

        // Poll for partial results (like poll_worker in the app)
        while let Ok(evt) = evt_rx.try_recv() {
            match evt {
                InferenceEvent::PartialResult(text) => {
                    partial_count += 1;
                    let elapsed = start.elapsed().as_secs_f64();
                    let audio_pos = offset as f64 / 16000.0;
                    eprintln!(
                        "[partial #{partial_count} wall={elapsed:.1}s audio={audio_pos:.1}s] {} chars",
                        text.len(),
                    );
                    last_partial = text;
                }
                InferenceEvent::Finished(_) => {
                    panic!("Got Finished before Finalize!");
                }
            }
        }

        thread::sleep(chunk_delay);
    }

    // --- stop_recording() simulation ---
    let stop_time = start.elapsed();
    eprintln!(
        "\n--- stop_recording() at {:.1}s ---",
        stop_time.as_secs_f64()
    );

    // Send remaining samples (drain + stop leftover — already sent in our case)
    // Send Finalize
    cmd_tx
        .send(InferenceCommand::Finalize)
        .expect("worker died");

    // Wait for Finished, collecting any late partials
    eprintln!("Waiting for Finished event...");
    let finalize_start = Instant::now();
    let mut final_text = String::new();

    loop {
        match evt_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(InferenceEvent::PartialResult(text)) => {
                partial_count += 1;
                eprintln!(
                    "[late partial #{partial_count}] {} chars",
                    text.len()
                );
                last_partial = text;
            }
            Ok(InferenceEvent::Finished(result)) => {
                let elapsed = finalize_start.elapsed();
                match result {
                    Ok(text) => {
                        eprintln!(
                            "[Finished] {} chars in {:.1}s",
                            text.len(),
                            elapsed.as_secs_f64()
                        );
                        final_text = text;
                    }
                    Err(e) => {
                        eprintln!("[Finished] FAILED: {e}");
                        std::process::exit(1);
                    }
                }
                break;
            }
            Err(_) => {
                eprintln!("TIMEOUT waiting for Finished!");
                std::process::exit(1);
            }
        }
    }

    let _ = cmd_tx.send(InferenceCommand::Shutdown);
    let _ = worker.join();

    // --- Results ---
    let total = start.elapsed();
    eprintln!(
        "\n=== RESULT ({:.1}s total) ===",
        total.as_secs_f64()
    );
    println!("{final_text}");

    // --- Verification ---
    eprintln!("\n=== Verification ===");
    let mut failures = 0;

    // Check text length
    eprintln!("  Final text: {} chars", final_text.len());
    eprintln!("  Last partial: {} chars", last_partial.len());
    if final_text.len() < 200 {
        eprintln!("  [FAIL] Final text too short!");
        failures += 1;
    }

    // Check EN sections
    for s in 1..=3 {
        let found = final_text.to_lowercase().contains(&format!("section {s}"));
        eprintln!(
            "  {} EN Section {s}",
            if found { "[OK]" } else { "[FAIL]" }
        );
        if !found {
            failures += 1;
        }
    }

    // Check RU content
    let has_ru = final_text.contains("Раздел") || final_text.contains("раздел");
    let has_konec = final_text.contains("Конец") || final_text.contains("конец");
    eprintln!(
        "  {} RU content present",
        if has_ru { "[OK]" } else { "[FAIL]" }
    );
    eprintln!(
        "  {} RU ending present",
        if has_konec { "[OK]" } else { "[FAIL]" }
    );
    if !has_ru {
        failures += 1;
    }

    // Partial accumulation check
    if partial_count == 0 {
        eprintln!("  [WARN] No partials during recording");
    } else {
        eprintln!("  [OK] {} partials during recording", partial_count);
    }

    eprintln!();
    if failures == 0 {
        eprintln!("[PASS]");
    } else {
        eprintln!("[FAIL] {failures} check(s) failed");
        std::process::exit(1);
    }
}
