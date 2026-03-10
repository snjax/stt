use std::{
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::Duration,
};

use anyhow::Result;
use arboard::Clipboard;
use eframe::egui::{self, Color32, RichText, Stroke, ViewportBuilder};

use crate::{
    audio::AudioRecorder,
    streaming::StreamingEngine,
    whisper_cpp::WhisperCppTranscriber,
};

const BACKEND_NAME: &str = "Whisper (whisper.cpp)";

enum InferenceCommand {
    /// Streaming: push new samples for partial result
    PushSamples(Vec<f32>),
    /// Streaming: finalize and get best result
    Finalize,
}

enum InferenceEvent {
    Finished(Result<String, String>),
    PartialResult(String),
}

pub fn run_gui() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("Speech To Text")
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([360.0, 240.0])
            .with_always_on_top(),
        ..Default::default()
    };

    eframe::run_native(
        "stt",
        options,
        Box::new(move |cc| Ok(Box::new(SpeechApp::new(cc)))),
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(())
}

pub struct SpeechApp {
    recorder: AudioRecorder,
    worker_tx: Sender<InferenceCommand>,
    worker_rx: Receiver<InferenceEvent>,
    transcription: String,
    partial_text: String,
    status_text: String,
    recording: bool,
    transcribing: bool,
    pinned: bool,
    auto_clipboard: bool,
}

impl SpeechApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (worker_tx, worker_rx) = spawn_inference_worker();
        Self {
            recorder: AudioRecorder::default(),
            worker_tx,
            worker_rx,
            transcription: String::new(),
            partial_text: String::new(),
            status_text: format!("Idle ({BACKEND_NAME})"),
            recording: false,
            transcribing: false,
            pinned: true,
            auto_clipboard: false,
        }
    }

    fn start_recording(&mut self) {
        match self.recorder.start() {
            Ok(()) => {
                self.recording = true;
                self.partial_text.clear();
                self.status_text = "Recording...".to_owned();
            }
            Err(error) => {
                self.status_text = format!("Audio error: {error}");
            }
        }
    }

    fn stop_recording(&mut self) {
        // Drain remaining samples
        let remaining = self.recorder.drain_new_samples();
        if !remaining.is_empty() {
            let _ = self
                .worker_tx
                .send(InferenceCommand::PushSamples(remaining));
        }
        // Stop the audio stream — capture any samples that arrived between drain and stop
        match self.recorder.stop() {
            Ok(leftover) => {
                if !leftover.is_empty() {
                    let _ = self
                        .worker_tx
                        .send(InferenceCommand::PushSamples(leftover));
                }
            }
            Err(e) => eprintln!("recorder stop error: {e}"),
        }
        self.recording = false;
        self.transcribing = true;
        self.status_text = "Finalizing...".to_owned();
        if let Err(error) = self.worker_tx.send(InferenceCommand::Finalize) {
            self.transcribing = false;
            self.status_text = format!("Worker error: {error}");
        }
    }

    /// Drain samples from recorder and send to streaming worker
    fn drain_streaming_samples(&mut self) {
        if !self.recording {
            return;
        }

        let samples = self.recorder.drain_new_samples();
        if !samples.is_empty() {
            let _ = self
                .worker_tx
                .send(InferenceCommand::PushSamples(samples));
        }
    }

    fn copy_transcription(&mut self) {
        if self.transcription.trim().is_empty() {
            self.status_text = "Nothing to copy.".to_owned();
            return;
        }

        match Clipboard::new()
            .and_then(|mut clipboard| clipboard.set_text(self.transcription.clone()))
        {
            Ok(()) => {
                self.status_text = "Copied to clipboard.".to_owned();
            }
            Err(error) => {
                self.status_text = format!("Clipboard error: {error}");
            }
        }
    }

    fn poll_worker(&mut self) {
        loop {
            match self.worker_rx.try_recv() {
                Ok(InferenceEvent::Finished(result)) => {
                    self.transcribing = false;
                    match result {
                        Ok(text) => {
                            self.transcription = text;
                            self.partial_text.clear();
                            if self.auto_clipboard {
                                self.copy_transcription();
                            } else {
                                self.status_text = "Transcription complete.".to_owned();
                            }
                        }
                        Err(error) => {
                            self.status_text = format!("Inference failed: {error}");
                        }
                    }
                }
                Ok(InferenceEvent::PartialResult(text)) => {
                    self.partial_text = text;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.transcribing = false;
                    self.status_text = "Inference worker stopped.".to_owned();
                    break;
                }
            }
        }
    }

    fn recording_indicator(&self, ui: &mut egui::Ui) {
        let height = 24.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);

        if self.recording {
            let time = ui.ctx().input(|input| input.time) as f32;
            let pulse = 0.5 + 0.5 * (time * 5.0).sin();
            let radius = 6.0 + pulse * 3.0;
            let alpha = (160.0 + pulse * 95.0) as u8;
            let center = egui::pos2(rect.left() + 12.0, rect.center().y);
            painter.circle_filled(
                center,
                radius,
                Color32::from_rgba_unmultiplied(220, 40, 40, alpha),
            );
            painter.text(
                egui::pos2(center.x + 14.0, rect.center().y - 8.0),
                egui::Align2::LEFT_TOP,
                "Recording...",
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::from_rgb(220, 40, 40),
            );
        } else if self.transcribing {
            painter.text(
                rect.left_center() + egui::vec2(0.0, -8.0),
                egui::Align2::LEFT_TOP,
                "Transcribing...",
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::from_rgb(240, 180, 60),
            );
        } else {
            painter.text(
                rect.left_center() + egui::vec2(0.0, -8.0),
                egui::Align2::LEFT_TOP,
                format!("Ready ({BACKEND_NAME})"),
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::GRAY,
            );
        }
    }
}

impl eframe::App for SpeechApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

        // Drain audio samples for streaming
        if self.recording {
            self.drain_streaming_samples();
        }

        if self.recording || self.transcribing {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.visuals_mut().widgets.active.bg_stroke =
                Stroke::new(1.5, Color32::from_rgb(220, 40, 40));

            let btn_size = egui::vec2(120.0, 48.0);
            let btn_font = 22.0;

            ui.horizontal(|ui| {
                let label = if self.recording { "Stop" } else { "Record" };
                let record_clicked = ui
                    .add_enabled(
                        !self.transcribing,
                        egui::Button::new(RichText::new(label).size(btn_font).strong())
                            .min_size(btn_size),
                    )
                    .clicked();
                if record_clicked {
                    if self.recording {
                        self.stop_recording();
                    } else {
                        self.start_recording();
                    }
                }

                if ui
                    .add(
                        egui::Button::new(RichText::new("Copy").size(btn_font))
                            .min_size(btn_size),
                    )
                    .clicked()
                {
                    self.copy_transcription();
                }

                let pin_label = if self.pinned { "Unpin" } else { "Pin" };
                if ui
                    .add(
                        egui::Button::new(RichText::new(pin_label).size(btn_font))
                            .min_size(egui::vec2(90.0, btn_size.y)),
                    )
                    .clicked()
                {
                    self.pinned = !self.pinned;
                    let level = if self.pinned {
                        egui::WindowLevel::AlwaysOnTop
                    } else {
                        egui::WindowLevel::Normal
                    };
                    ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
                }
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.auto_clipboard, "Auto-copy to clipboard");
            });

            ui.add_space(4.0);
            self.recording_indicator(ui);
            ui.label(RichText::new(&self.status_text).small());
            ui.add_space(4.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                // Show partial text in gray while recording (streaming)
                if self.recording && !self.partial_text.is_empty() {
                    ui.label(
                        RichText::new(&self.partial_text)
                            .italics()
                            .color(Color32::GRAY),
                    );
                    ui.add_space(4.0);
                }

                ui.add(
                    egui::TextEdit::multiline(&mut self.transcription)
                        .desired_width(f32::INFINITY)
                        .desired_rows(14)
                        .lock_focus(true),
                );
            });
        });
    }
}

fn spawn_inference_worker() -> (Sender<InferenceCommand>, Receiver<InferenceEvent>) {
    let (command_tx, command_rx) = mpsc::channel::<InferenceCommand>();
    let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();

    thread::spawn(move || {
        let mut engine: Option<StreamingEngine> = None;

        while let Ok(command) = command_rx.recv() {
            // Lazily initialize
            if engine.is_none() {
                match WhisperCppTranscriber::new() {
                    Ok(t) => engine = Some(StreamingEngine::new(t)),
                    Err(e) => {
                        let _ = event_tx.send(InferenceEvent::Finished(Err(e.to_string())));
                        continue;
                    }
                }
            }

            let engine = engine.as_mut().unwrap();

            match command {
                InferenceCommand::PushSamples(samples) => {
                    if let Some(text) = engine.push_samples(&samples)
                        && event_tx
                            .send(InferenceEvent::PartialResult(text))
                            .is_err()
                    {
                        break;
                    }
                }
                InferenceCommand::Finalize => {
                    // Drain all pending PushSamples without running inference
                    while let Ok(cmd) = command_rx.try_recv() {
                        if let InferenceCommand::PushSamples(samples) = cmd {
                            engine.append_samples(&samples);
                        }
                    }
                    let result = engine.finalize().map_err(|e| e.to_string());
                    if event_tx.send(InferenceEvent::Finished(result)).is_err() {
                        break;
                    }
                }
            }
        }
    });

    (command_tx, event_rx)
}
