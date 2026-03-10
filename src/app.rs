use std::{
    fmt,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::Duration,
};

use anyhow::Result;
use arboard::Clipboard;
use eframe::egui::{self, Color32, RichText, Stroke, ViewportBuilder};

use crate::{
    audio::AudioRecorder,
    inference::{ModelPaths, Transcriber},
    whisper::WhisperTranscriber,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendChoice {
    GigaAm,
    Whisper,
}

impl BackendChoice {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::GigaAm => "GigaAM-v3",
            Self::Whisper => "Whisper Large V3 Turbo",
        }
    }
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

enum InferenceCommand {
    Transcribe(Vec<f32>),
}

enum InferenceEvent {
    Finished(Result<String, String>),
}

enum WorkerTranscriber {
    GigaAm(Transcriber),
    Whisper(WhisperTranscriber),
}

impl WorkerTranscriber {
    fn new(backend: BackendChoice) -> Result<Self> {
        match backend {
            BackendChoice::GigaAm => {
                Ok(Self::GigaAm(Transcriber::new(ModelPaths::discover()?)?))
            }
            BackendChoice::Whisper => Ok(Self::Whisper(WhisperTranscriber::new()?)),
        }
    }

    fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        match self {
            Self::GigaAm(transcriber) => transcriber.transcribe_samples(samples),
            Self::Whisper(transcriber) => transcriber.transcribe_samples(samples),
        }
    }
}

pub fn run_gui(backend: BackendChoice) -> Result<()> {
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
        Box::new(move |cc| Ok(Box::new(SpeechApp::new(cc, backend)))),
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(())
}

pub struct SpeechApp {
    backend: BackendChoice,
    recorder: AudioRecorder,
    worker_tx: Sender<InferenceCommand>,
    worker_rx: Receiver<InferenceEvent>,
    transcription: String,
    status_text: String,
    recording: bool,
    transcribing: bool,
    pinned: bool,
    auto_clipboard: bool,
}

impl SpeechApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, backend: BackendChoice) -> Self {
        let (worker_tx, worker_rx) = spawn_inference_worker(backend);
        Self {
            backend,
            recorder: AudioRecorder::default(),
            worker_tx,
            worker_rx,
            transcription: String::new(),
            status_text: format!("Idle ({backend})"),
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
                self.status_text = format!("Recording for {}...", self.backend);
            }
            Err(error) => {
                self.status_text = format!("Audio error: {error}");
            }
        }
    }

    fn stop_recording(&mut self) {
        match self.recorder.stop() {
            Ok(samples) => {
                self.recording = false;
                if samples.is_empty() {
                    self.status_text = "No audio captured.".to_owned();
                    return;
                }

                self.transcribing = true;
                self.status_text = format!(
                    "Transcribing {:.1}s of audio with {}...",
                    samples.len() as f32 / 16_000.0,
                    self.backend
                );
                if let Err(error) = self.worker_tx.send(InferenceCommand::Transcribe(samples)) {
                    self.transcribing = false;
                    self.status_text = format!("Worker error: {error}");
                }
            }
            Err(error) => {
                self.recording = false;
                self.status_text = format!("Stop failed: {error}");
            }
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
                            if self.auto_clipboard {
                                self.copy_transcription();
                            } else {
                                self.status_text =
                                    format!("Transcription complete ({})", self.backend);
                            }
                        }
                        Err(error) => {
                            self.status_text = format!("Inference failed: {error}");
                        }
                    }
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
                format!("Recording for {}...", self.backend),
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::from_rgb(220, 40, 40),
            );
        } else if self.transcribing {
            painter.text(
                rect.left_center() + egui::vec2(0.0, -8.0),
                egui::Align2::LEFT_TOP,
                format!("Running {}...", self.backend),
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::from_rgb(240, 180, 60),
            );
        } else {
            painter.text(
                rect.left_center() + egui::vec2(0.0, -8.0),
                egui::Align2::LEFT_TOP,
                format!("Ready ({})", self.backend),
                egui::TextStyle::Body.resolve(ui.style()),
                Color32::GRAY,
            );
        }
    }
}

impl eframe::App for SpeechApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

        if self.recording || self.transcribing {
            ctx.request_repaint_after(Duration::from_millis(16));
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
            ui.label(RichText::new(format!("Backend: {}", self.backend)).small());
            self.recording_indicator(ui);
            ui.label(RichText::new(&self.status_text).small());
            ui.add_space(4.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
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

fn spawn_inference_worker(backend: BackendChoice) -> (Sender<InferenceCommand>, Receiver<InferenceEvent>) {
    let (command_tx, command_rx) = mpsc::channel::<InferenceCommand>();
    let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();

    thread::spawn(move || {
        let mut transcriber: Option<WorkerTranscriber> = None;

        while let Ok(command) = command_rx.recv() {
            match command {
                InferenceCommand::Transcribe(samples) => {
                    let result = (|| -> Result<String> {
                        if transcriber.is_none() {
                            transcriber = Some(WorkerTranscriber::new(backend)?);
                        }

                        let transcriber = transcriber.as_mut().expect("transcriber initialized");
                        transcriber.transcribe_samples(&samples)
                    })()
                    .map_err(|error| error.to_string());

                    if event_tx.send(InferenceEvent::Finished(result)).is_err() {
                        break;
                    }
                }
            }
        }
    });

    (command_tx, event_rx)
}
