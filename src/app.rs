use std::{
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use arboard::Clipboard;
use eframe::egui::{self, RichText, ViewportBuilder, ViewportClass, ViewportId};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::{Code, HotKey, Modifiers},
};

use crate::{
    audio::AudioRecorder,
    paste::{get_active_window, simulate_paste},
    streaming::StreamingEngine,
    whisper_cpp::WhisperCppTranscriber,
};

// ── AutoAction ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum AutoAction {
    None,
    Copy,
    Paste,
}

impl AutoAction {
    const ALL: [AutoAction; 3] = [AutoAction::None, AutoAction::Copy, AutoAction::Paste];

    fn label(self) -> &'static str {
        match self {
            AutoAction::None => "Nothing",
            AutoAction::Copy => "Copy to clipboard",
            AutoAction::Paste => "Paste to active window",
        }
    }

    fn description(self) -> &'static str {
        match self {
            AutoAction::None => "Do nothing after transcription completes.",
            AutoAction::Copy => "Automatically copy the transcribed text to the system clipboard.",
            AutoAction::Paste => {
                "Copy text and simulate Ctrl+V / Cmd+V into the window that was \
                 active when recording started. The previous clipboard contents \
                 are restored afterward. Falls back to copy if the target window \
                 has changed or cannot be detected."
            }
        }
    }
}

// ── Hotkey config ───────────────────────────────────────────────────

const KEY_OPTIONS: &[(&str, Code)] = &[
    ("A", Code::KeyA),
    ("B", Code::KeyB),
    ("C", Code::KeyC),
    ("D", Code::KeyD),
    ("E", Code::KeyE),
    ("F", Code::KeyF),
    ("G", Code::KeyG),
    ("H", Code::KeyH),
    ("I", Code::KeyI),
    ("J", Code::KeyJ),
    ("K", Code::KeyK),
    ("L", Code::KeyL),
    ("M", Code::KeyM),
    ("N", Code::KeyN),
    ("O", Code::KeyO),
    ("P", Code::KeyP),
    ("Q", Code::KeyQ),
    ("R", Code::KeyR),
    ("S", Code::KeyS),
    ("T", Code::KeyT),
    ("U", Code::KeyU),
    ("V", Code::KeyV),
    ("W", Code::KeyW),
    ("X", Code::KeyX),
    ("Y", Code::KeyY),
    ("Z", Code::KeyZ),
    ("F1", Code::F1),
    ("F2", Code::F2),
    ("F3", Code::F3),
    ("F4", Code::F4),
    ("F5", Code::F5),
    ("F6", Code::F6),
    ("F7", Code::F7),
    ("F8", Code::F8),
    ("F9", Code::F9),
    ("F10", Code::F10),
    ("F11", Code::F11),
    ("F12", Code::F12),
];

#[derive(Clone)]
struct HotkeyConfig {
    use_super: bool,
    use_ctrl: bool,
    use_shift: bool,
    use_alt: bool,
    key_idx: usize,
}

impl HotkeyConfig {
    fn default_config() -> Self {
        Self {
            use_super: true,
            use_ctrl: false,
            use_shift: true,
            use_alt: false,
            key_idx: 17, // R
        }
    }

    fn to_hotkey(&self) -> HotKey {
        let mut mods = Modifiers::empty();
        if self.use_super {
            mods |= Modifiers::SUPER;
        }
        if self.use_ctrl {
            mods |= Modifiers::CONTROL;
        }
        if self.use_shift {
            mods |= Modifiers::SHIFT;
        }
        if self.use_alt {
            mods |= Modifiers::ALT;
        }
        let modifiers = if mods.is_empty() { None } else { Some(mods) };
        HotKey::new(modifiers, KEY_OPTIONS[self.key_idx].1)
    }

    fn display(&self) -> String {
        let mut parts = Vec::new();
        if self.use_super {
            parts.push(if cfg!(target_os = "macos") {
                "Cmd"
            } else {
                "Super"
            });
        }
        if self.use_ctrl {
            parts.push("Ctrl");
        }
        if self.use_alt {
            parts.push(if cfg!(target_os = "macos") {
                "Option"
            } else {
                "Alt"
            });
        }
        if self.use_shift {
            parts.push("Shift");
        }
        parts.push(KEY_OPTIONS[self.key_idx].0);
        parts.join("+")
    }
}

// ── Worker protocol ─────────────────────────────────────────────────

enum InferenceCommand {
    PushSamples(Vec<f32>),
    Finalize,
    Shutdown,
}

enum InferenceEvent {
    Finished(Result<String, String>),
    PartialResult(String),
}

// ── Entry point ─────────────────────────────────────────────────────

pub fn run_gui() -> Result<()> {
    let hotkey_manager = GlobalHotKeyManager::new()
        .map_err(|e| anyhow::anyhow!("failed to create hotkey manager: {e}"))?;

    let config = HotkeyConfig::default_config();
    let hotkey = config.to_hotkey();
    hotkey_manager
        .register(hotkey)
        .map_err(|e| anyhow::anyhow!("failed to register hotkey: {e}"))?;

    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title("STT — Ready")
            .with_inner_size([200.0, 160.0])
            .with_min_inner_size([140.0, 100.0])
            .with_always_on_top(),
        ..Default::default()
    };

    eframe::run_native(
        "stt",
        options,
        Box::new(move |cc| {
            Ok(Box::new(SpeechApp::new(cc, hotkey, config, hotkey_manager)))
        }),
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(())
}

// ── App state ───────────────────────────────────────────────────────

pub struct SpeechApp {
    recorder: AudioRecorder,
    worker_tx: Sender<InferenceCommand>,
    worker_rx: Receiver<InferenceEvent>,
    worker_handle: Option<thread::JoinHandle<()>>,
    transcription: String,
    partial_text: String,
    recording: bool,
    transcribing: bool,

    // Settings
    pinned: bool,
    auto_action: AutoAction,
    hotkey_config: HotkeyConfig,
    pending_hotkey: HotkeyConfig,

    // Hotkey management
    current_hotkey: HotKey,
    hotkey_manager: GlobalHotKeyManager,

    // Paste target tracking
    paste_target_window: Option<String>,
    title_warning: Option<(String, Instant)>,

    // UI
    show_settings: bool,
}

impl SpeechApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        hotkey: HotKey,
        config: HotkeyConfig,
        hotkey_manager: GlobalHotKeyManager,
    ) -> Self {
        let (worker_tx, worker_rx, worker_handle) = spawn_inference_worker();
        Self {
            recorder: AudioRecorder::default(),
            worker_tx,
            worker_rx,
            worker_handle: Some(worker_handle),
            transcription: String::new(),
            partial_text: String::new(),
            recording: false,
            transcribing: false,

            pinned: true,
            auto_action: AutoAction::Paste,
            hotkey_config: config.clone(),
            pending_hotkey: config,

            current_hotkey: hotkey,
            hotkey_manager,

            paste_target_window: None,
            title_warning: None,

            show_settings: false,
        }
    }

    // ── Recording ───────────────────────────────────────────────────

    fn start_recording(&mut self) {
        self.paste_target_window = get_active_window();
        self.title_warning = None;

        match self.recorder.start() {
            Ok(()) => {
                self.recording = true;
                self.partial_text.clear();
            }
            Err(error) => {
                eprintln!("Audio error: {error}");
            }
        }
    }

    fn stop_recording(&mut self) {
        let remaining = self.recorder.drain_new_samples();
        if !remaining.is_empty() {
            let _ = self
                .worker_tx
                .send(InferenceCommand::PushSamples(remaining));
        }
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
        if let Err(error) = self.worker_tx.send(InferenceCommand::Finalize) {
            self.transcribing = false;
            eprintln!("Worker error: {error}");
        }
    }

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

    // ── Clipboard / paste ───────────────────────────────────────────

    fn copy_to_clipboard(&self) {
        if let Err(e) = Clipboard::new()
            .and_then(|mut cb| cb.set_text(self.transcription.clone()))
        {
            eprintln!("Clipboard error: {e}");
        }
    }

    fn handle_auto_action(&mut self) {
        if self.transcription.trim().is_empty() {
            return;
        }
        match self.auto_action {
            AutoAction::None => {}
            AutoAction::Copy => self.copy_to_clipboard(),
            AutoAction::Paste => {
                let current_window = get_active_window();
                let window_matches = match (&self.paste_target_window, &current_window) {
                    (Some(target), Some(current)) => target == current,
                    _ => false,
                };

                if !window_matches {
                    self.copy_to_clipboard();
                    self.title_warning = Some((
                        "Paste skipped: window changed".to_owned(),
                        Instant::now(),
                    ));
                    return;
                }

                let old_clipboard = Clipboard::new()
                    .and_then(|mut cb| cb.get_text())
                    .ok();
                self.copy_to_clipboard();
                simulate_paste();
                if let Some(old) = old_clipboard {
                    thread::spawn(move || {
                        thread::sleep(Duration::from_millis(200));
                        let _ = Clipboard::new().and_then(|mut cb| cb.set_text(old));
                    });
                }
            }
        }
    }

    // ── Polling ─────────────────────────────────────────────────────

    fn poll_worker(&mut self) {
        loop {
            match self.worker_rx.try_recv() {
                Ok(InferenceEvent::Finished(result)) => {
                    self.transcribing = false;
                    match result {
                        Ok(text) => {
                            self.transcription = text;
                            self.partial_text.clear();
                            self.handle_auto_action();
                        }
                        Err(error) => {
                            eprintln!("Inference failed: {error}");
                        }
                    }
                }
                Ok(InferenceEvent::PartialResult(text)) => {
                    self.partial_text = text;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.transcribing = false;
                    break;
                }
            }
        }
    }

    fn poll_global_hotkey(&mut self) {
        if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
            if event.id == self.current_hotkey.id()
                && event.state == global_hotkey::HotKeyState::Pressed
            {
                if self.transcribing {
                    return;
                }
                if self.recording {
                    self.stop_recording();
                } else {
                    self.start_recording();
                }
            }
        }
    }

    // ── Hotkey management ───────────────────────────────────────────

    fn apply_hotkey(&mut self) {
        let new_hotkey = self.pending_hotkey.to_hotkey();
        if new_hotkey.id() == self.current_hotkey.id() {
            return;
        }

        if let Err(e) = self.hotkey_manager.unregister(self.current_hotkey) {
            eprintln!("Failed to unregister old hotkey: {e}");
        }
        match self.hotkey_manager.register(new_hotkey) {
            Ok(()) => {
                self.current_hotkey = new_hotkey;
                self.hotkey_config = self.pending_hotkey.clone();
            }
            Err(e) => {
                eprintln!("Failed to register new hotkey: {e}, restoring old");
                let _ = self.hotkey_manager.register(self.current_hotkey);
                self.pending_hotkey = self.hotkey_config.clone();
            }
        }
    }

    // ── Title ───────────────────────────────────────────────────────

    fn window_title(&mut self) -> String {
        if let Some((ref msg, when)) = self.title_warning {
            if when.elapsed() < Duration::from_secs(5) {
                return format!("STT — {msg}");
            }
            self.title_warning = None;
        }

        if self.recording {
            "STT \u{23FA} Recording...".to_owned()
        } else if self.transcribing {
            "STT — Transcribing...".to_owned()
        } else {
            "STT — Ready".to_owned()
        }
    }
}

// ── eframe::App ─────────────────────────────────────────────────────

impl eframe::App for SpeechApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.poll_global_hotkey();

        if self.recording {
            self.drain_streaming_samples();
        }

        if self.recording || self.transcribing {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
        ctx.request_repaint_after(Duration::from_millis(200));

        ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.window_title()));

        // ── Settings viewport (separate OS window) ──────────────
        if self.show_settings {
            let was_pinned = self.pinned;
            let mut should_apply_hotkey = false;
            let mut should_close = false;

            ctx.show_viewport_immediate(
                ViewportId::from_hash_of("stt_settings"),
                ViewportBuilder::default()
                    .with_title("\u{2699}\u{FE0F} STT Settings")
                    .with_inner_size([380.0, 400.0])
                    .with_resizable(false),
                |inner_ctx, class| {
                    if inner_ctx.input(|i| i.viewport().close_requested()) {
                        should_close = true;
                    }

                    // If embedded (backend doesn't support multi-viewport),
                    // wrap in an egui::Window
                    if class == ViewportClass::Embedded {
                        let mut open = true;
                        egui::Window::new("\u{2699}\u{FE0F} Settings")
                            .open(&mut open)
                            .resizable(false)
                            .collapsible(false)
                            .default_width(360.0)
                            .show(inner_ctx, |ui| {
                                Self::draw_settings_ui(
                                    ui,
                                    &mut self.pending_hotkey,
                                    &self.hotkey_config,
                                    &mut self.auto_action,
                                    &mut self.pinned,
                                    &mut should_apply_hotkey,
                                );
                            });
                        if !open {
                            should_close = true;
                        }
                    } else {
                        egui::CentralPanel::default().show(inner_ctx, |ui| {
                            Self::draw_settings_ui(
                                ui,
                                &mut self.pending_hotkey,
                                &self.hotkey_config,
                                &mut self.auto_action,
                                &mut self.pinned,
                                &mut should_apply_hotkey,
                            );
                        });
                    }
                },
            );

            if should_close {
                self.show_settings = false;
            }
            if should_apply_hotkey {
                self.apply_hotkey();
            }
            if self.pinned != was_pinned {
                let level = if self.pinned {
                    egui::WindowLevel::AlwaysOnTop
                } else {
                    egui::WindowLevel::Normal
                };
                ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
            }
        }

        // ── Main panel ──────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Record / Stop
                let icon = if self.recording {
                    "\u{23F9}"
                } else {
                    "\u{23FA}"
                };
                let tooltip = if self.recording {
                    "Stop recording".to_owned()
                } else {
                    format!("Record ({})", self.hotkey_config.display())
                };
                if ui
                    .add_enabled(
                        !self.transcribing,
                        egui::Button::new(RichText::new(icon).size(18.0)),
                    )
                    .on_hover_text(tooltip)
                    .clicked()
                {
                    if self.recording {
                        self.stop_recording();
                    } else {
                        self.start_recording();
                    }
                }

                // Copy
                if ui
                    .add(egui::Button::new(RichText::new("\u{1F4CB}").size(18.0)))
                    .on_hover_text("Copy to clipboard")
                    .clicked()
                {
                    self.copy_to_clipboard();
                }

                // Settings
                if ui
                    .add(egui::Button::new(RichText::new("\u{2699}").size(18.0)))
                    .on_hover_text("Settings")
                    .clicked()
                {
                    self.show_settings = !self.show_settings;
                }
            });

            // Textarea — fills all remaining space
            let text = if self.recording && !self.partial_text.is_empty() {
                &self.partial_text
            } else {
                &self.transcription
            };
            let mut display = text.to_owned();
            let available = ui.available_size();
            ui.add_sized(
                available,
                egui::TextEdit::multiline(&mut display).desired_width(f32::INFINITY),
            );
            if !self.recording {
                self.transcription = display;
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.worker_tx.send(InferenceCommand::Shutdown);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

// ── Settings UI (shared between viewport and embedded fallback) ─────

impl SpeechApp {
    fn draw_settings_ui(
        ui: &mut egui::Ui,
        pending: &mut HotkeyConfig,
        current: &HotkeyConfig,
        auto_action: &mut AutoAction,
        pinned: &mut bool,
        apply_clicked: &mut bool,
    ) {
        ui.heading("Hotkey");
        ui.label("Global keyboard shortcut to start/stop recording.");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if cfg!(target_os = "macos") {
                ui.checkbox(&mut pending.use_super, "Cmd");
            } else {
                ui.checkbox(&mut pending.use_super, "Super");
            }
            ui.checkbox(&mut pending.use_ctrl, "Ctrl");
            ui.checkbox(&mut pending.use_shift, "Shift");
            if cfg!(target_os = "macos") {
                ui.checkbox(&mut pending.use_alt, "Option");
            } else {
                ui.checkbox(&mut pending.use_alt, "Alt");
            }
        });

        ui.horizontal(|ui| {
            ui.label("Key:");
            egui::ComboBox::from_id_salt("hotkey_key")
                .width(55.0)
                .selected_text(KEY_OPTIONS[pending.key_idx].0)
                .show_ui(ui, |ui| {
                    for (i, (name, _)) in KEY_OPTIONS.iter().enumerate() {
                        ui.selectable_value(&mut pending.key_idx, i, *name);
                    }
                });
        });

        ui.add_space(4.0);

        let pending_display = pending.display();
        let current_display = current.display();
        let changed = pending_display != current_display;

        ui.horizontal(|ui| {
            ui.label(format!("Current: {current_display}"));
            if changed && ui.button("Apply").clicked() {
                *apply_clicked = true;
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        // ── After transcription ─────────────────────────────────
        ui.heading("After transcription");
        ui.add_space(4.0);

        for action in AutoAction::ALL {
            ui.radio_value(auto_action, action, action.label());
        }

        ui.add_space(4.0);
        ui.indent("auto_action_help", |ui| {
            ui.label(RichText::new(auto_action.description()).weak().small());
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        // ── Window ──────────────────────────────────────────────
        ui.heading("Window");
        ui.add_space(4.0);

        ui.checkbox(pinned, "Always on top");
        ui.indent("pin_help", |ui| {
            ui.label(
                RichText::new("Keep the STT window above all other windows.")
                    .weak()
                    .small(),
            );
        });
    }
}

// ── Inference worker ────────────────────────────────────────────────

fn spawn_inference_worker() -> (
    Sender<InferenceCommand>,
    Receiver<InferenceEvent>,
    thread::JoinHandle<()>,
) {
    let (command_tx, command_rx) = mpsc::channel::<InferenceCommand>();
    let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();

    let handle = thread::spawn(move || {
        let mut engine: Option<StreamingEngine> = None;

        while let Ok(command) = command_rx.recv() {
            if matches!(command, InferenceCommand::Shutdown) {
                break;
            }

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
                InferenceCommand::Shutdown => unreachable!(),
            }
        }
    });

    (command_tx, event_rx, handle)
}
