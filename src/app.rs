use std::{
    fs,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use eframe::egui::{self, RichText, ViewportBuilder, ViewportClass, ViewportId};
use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::{Code, HotKey, Modifiers},
};
use serde::{Deserialize, Serialize};

use crate::{
    audio::AudioRecorder,
    paste::{clipboard_copy, clipboard_get, get_active_window},
    streaming::StreamingEngine,
    whisper_cpp::WhisperCppTranscriber,
};

// ── AutoAction ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
struct HotkeyConfig {
    use_super: bool,
    use_ctrl: bool,
    use_shift: bool,
    use_alt: bool,
    key_name: String,
}

impl HotkeyConfig {
    fn default_config() -> Self {
        Self {
            use_super: true,
            use_ctrl: false,
            use_shift: true,
            use_alt: false,
            key_name: "R".into(),
        }
    }

    fn to_hotkey(&self) -> Option<HotKey> {
        let code = key_name_to_code(&self.key_name)?;
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
        Some(HotKey::new(modifiers, code))
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
        parts.push(&self.key_name);
        parts.join("+")
    }

    #[cfg(target_os = "linux")]
    fn to_portal_trigger(&self) -> String {
        let mut trigger = String::new();
        if self.use_super {
            trigger.push_str("<Super>");
        }
        if self.use_ctrl {
            trigger.push_str("<Control>");
        }
        if self.use_alt {
            trigger.push_str("<Alt>");
        }
        if self.use_shift {
            trigger.push_str("<Shift>");
        }
        trigger.push_str(&self.key_name);
        trigger
    }
}

/// Map key name to global-hotkey Code for X11/macOS backend.
fn key_name_to_code(name: &str) -> Option<Code> {
    Some(match name {
        "A" => Code::KeyA,
        "B" => Code::KeyB,
        "C" => Code::KeyC,
        "D" => Code::KeyD,
        "E" => Code::KeyE,
        "F" => Code::KeyF,
        "G" => Code::KeyG,
        "H" => Code::KeyH,
        "I" => Code::KeyI,
        "J" => Code::KeyJ,
        "K" => Code::KeyK,
        "L" => Code::KeyL,
        "M" => Code::KeyM,
        "N" => Code::KeyN,
        "O" => Code::KeyO,
        "P" => Code::KeyP,
        "Q" => Code::KeyQ,
        "R" => Code::KeyR,
        "S" => Code::KeyS,
        "T" => Code::KeyT,
        "U" => Code::KeyU,
        "V" => Code::KeyV,
        "W" => Code::KeyW,
        "X" => Code::KeyX,
        "Y" => Code::KeyY,
        "Z" => Code::KeyZ,
        "1" => Code::Digit1,
        "2" => Code::Digit2,
        "3" => Code::Digit3,
        "4" => Code::Digit4,
        "5" => Code::Digit5,
        "6" => Code::Digit6,
        "7" => Code::Digit7,
        "8" => Code::Digit8,
        "9" => Code::Digit9,
        "0" => Code::Digit0,
        "F1" => Code::F1,
        "F2" => Code::F2,
        "F3" => Code::F3,
        "F4" => Code::F4,
        "F5" => Code::F5,
        "F6" => Code::F6,
        "F7" => Code::F7,
        "F8" => Code::F8,
        "F9" => Code::F9,
        "F10" => Code::F10,
        "F11" => Code::F11,
        "F12" => Code::F12,
        "Space" => Code::Space,
        "Enter" => Code::Enter,
        "Tab" => Code::Tab,
        "Escape" => Code::Escape,
        "Backspace" => Code::Backspace,
        "Up" => Code::ArrowUp,
        "Down" => Code::ArrowDown,
        "Left" => Code::ArrowLeft,
        "Right" => Code::ArrowRight,
        "Home" => Code::Home,
        "End" => Code::End,
        "PageUp" => Code::PageUp,
        "PageDown" => Code::PageDown,
        "Insert" => Code::Insert,
        "Delete" => Code::Delete,
        "Minus" => Code::Minus,
        "Equal" => Code::Equal,
        "LeftBracket" => Code::BracketLeft,
        "RightBracket" => Code::BracketRight,
        "Backslash" => Code::Backslash,
        "Semicolon" => Code::Semicolon,
        "Apostrophe" => Code::Quote,
        "Grave" => Code::Backquote,
        "Comma" => Code::Comma,
        "Dot" => Code::Period,
        "Slash" => Code::Slash,
        "CapsLock" => Code::CapsLock,
        "Num0" => Code::Numpad0,
        "Num1" => Code::Numpad1,
        "Num2" => Code::Numpad2,
        "Num3" => Code::Numpad3,
        "Num4" => Code::Numpad4,
        "Num5" => Code::Numpad5,
        "Num6" => Code::Numpad6,
        "Num7" => Code::Numpad7,
        "Num8" => Code::Numpad8,
        "Num9" => Code::Numpad9,
        "NumEnter" => Code::NumpadEnter,
        "NumPlus" => Code::NumpadAdd,
        "NumMinus" => Code::NumpadSubtract,
        "NumMultiply" => Code::NumpadMultiply,
        "NumDivide" => Code::NumpadDivide,
        "NumDot" => Code::NumpadDecimal,
        _ => return None,
    })
}

/// Map egui::Key to our key_name format.
fn egui_key_to_name(key: egui::Key) -> Option<&'static str> {
    Some(match key {
        egui::Key::A => "A",
        egui::Key::B => "B",
        egui::Key::C => "C",
        egui::Key::D => "D",
        egui::Key::E => "E",
        egui::Key::F => "F",
        egui::Key::G => "G",
        egui::Key::H => "H",
        egui::Key::I => "I",
        egui::Key::J => "J",
        egui::Key::K => "K",
        egui::Key::L => "L",
        egui::Key::M => "M",
        egui::Key::N => "N",
        egui::Key::O => "O",
        egui::Key::P => "P",
        egui::Key::Q => "Q",
        egui::Key::R => "R",
        egui::Key::S => "S",
        egui::Key::T => "T",
        egui::Key::U => "U",
        egui::Key::V => "V",
        egui::Key::W => "W",
        egui::Key::X => "X",
        egui::Key::Y => "Y",
        egui::Key::Z => "Z",
        egui::Key::Num0 => "0",
        egui::Key::Num1 => "1",
        egui::Key::Num2 => "2",
        egui::Key::Num3 => "3",
        egui::Key::Num4 => "4",
        egui::Key::Num5 => "5",
        egui::Key::Num6 => "6",
        egui::Key::Num7 => "7",
        egui::Key::Num8 => "8",
        egui::Key::Num9 => "9",
        egui::Key::F1 => "F1",
        egui::Key::F2 => "F2",
        egui::Key::F3 => "F3",
        egui::Key::F4 => "F4",
        egui::Key::F5 => "F5",
        egui::Key::F6 => "F6",
        egui::Key::F7 => "F7",
        egui::Key::F8 => "F8",
        egui::Key::F9 => "F9",
        egui::Key::F10 => "F10",
        egui::Key::F11 => "F11",
        egui::Key::F12 => "F12",
        egui::Key::F13 => "F13",
        egui::Key::F14 => "F14",
        egui::Key::F15 => "F15",
        egui::Key::F16 => "F16",
        egui::Key::F17 => "F17",
        egui::Key::F18 => "F18",
        egui::Key::F19 => "F19",
        egui::Key::F20 => "F20",
        egui::Key::ArrowUp => "Up",
        egui::Key::ArrowDown => "Down",
        egui::Key::ArrowLeft => "Left",
        egui::Key::ArrowRight => "Right",
        egui::Key::Home => "Home",
        egui::Key::End => "End",
        egui::Key::PageUp => "PageUp",
        egui::Key::PageDown => "PageDown",
        egui::Key::Insert => "Insert",
        egui::Key::Delete => "Delete",
        egui::Key::Space => "Space",
        egui::Key::Enter => "Enter",
        egui::Key::Tab => "Tab",
        egui::Key::Escape => "Escape",
        egui::Key::Backspace => "Backspace",
        egui::Key::Minus => "Minus",
        egui::Key::Plus => "Equal",
        _ => return None,
    })
}

// ── Hotkey backend ──────────────────────────────────────────────────

enum HotkeyBackend {
    GlobalHotkey {
        manager: GlobalHotKeyManager,
        hotkey: HotKey,
    },
    #[cfg(target_os = "linux")]
    Portal {
        listener: crate::wayland_hotkey::WaylandHotkeyListener,
    },
    #[cfg(target_os = "linux")]
    Evdev {
        listener: crate::evdev_hotkey::EvdevHotkeyListener,
    },
}

impl HotkeyBackend {
    fn hotkey_display(&self, config: &HotkeyConfig) -> String {
        match self {
            HotkeyBackend::GlobalHotkey { .. } => config.display(),
            #[cfg(target_os = "linux")]
            HotkeyBackend::Portal { listener } => listener.assigned_trigger.clone(),
            #[cfg(target_os = "linux")]
            HotkeyBackend::Evdev { .. } => config.display(),
        }
    }

    fn is_configurable(&self) -> bool {
        match self {
            HotkeyBackend::GlobalHotkey { .. } => true,
            #[cfg(target_os = "linux")]
            HotkeyBackend::Evdev { .. } => true,
            #[cfg(target_os = "linux")]
            HotkeyBackend::Portal { .. } => false,
        }
    }
}

fn setup_global_hotkey(config: &HotkeyConfig) -> Result<HotkeyBackend> {
    let manager = GlobalHotKeyManager::new()
        .map_err(|e| anyhow::anyhow!("failed to create hotkey manager: {e}"))?;
    let hotkey = config
        .to_hotkey()
        .ok_or_else(|| anyhow::anyhow!("key '{}' is not supported by X11/macOS backend", config.key_name))?;
    manager
        .register(hotkey)
        .map_err(|e| anyhow::anyhow!("failed to register hotkey: {e}"))?;
    Ok(HotkeyBackend::GlobalHotkey { manager, hotkey })
}

// ── Learn mode ──────────────────────────────────────────────────────

enum LearnState {
    Idle,
    #[cfg(target_os = "linux")]
    Evdev(crate::evdev_hotkey::EvdevKeyLearner),
    Egui,
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

// ── Persistent settings ─────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Settings {
    hotkey: HotkeyConfig,
    auto_action: AutoAction,
    pinned: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: HotkeyConfig::default_config(),
            auto_action: AutoAction::Paste,
            pinned: true,
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("stt").join("settings.json"))
}

fn load_settings() -> Settings {
    let Some(path) = settings_path() else {
        return Settings::default();
    };
    let Ok(data) = fs::read_to_string(&path) else {
        return Settings::default();
    };
    serde_json::from_str::<Settings>(&data).unwrap_or_else(|e| {
        eprintln!("Failed to parse settings: {e}");
        Settings::default()
    })
}

fn save_settings(settings: &Settings) {
    let Some(path) = settings_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(settings) {
        Ok(json) => {
            if let Err(e) = fs::write(&path, json) {
                eprintln!("Failed to save settings: {e}");
            }
        }
        Err(e) => eprintln!("Failed to serialize settings: {e}"),
    }
}

// ── Entry point ─────────────────────────────────────────────────────

pub fn run_gui() -> Result<()> {
    let settings = load_settings();
    let config = settings.hotkey.clone();

    let hotkey_backend = create_hotkey_backend(&config)?;

    let mut viewport = ViewportBuilder::default()
        .with_title("STT — Ready")
        .with_inner_size([200.0, 160.0])
        .with_min_inner_size([140.0, 100.0]);
    if settings.pinned {
        viewport = viewport.with_always_on_top();
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "stt",
        options,
        Box::new(move |cc| {
            Ok(Box::new(SpeechApp::new(cc, settings, hotkey_backend)))
        }),
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(())
}

fn create_hotkey_backend(config: &HotkeyConfig) -> Result<HotkeyBackend> {
    #[cfg(target_os = "linux")]
    {
        let is_wayland = std::env::var("WAYLAND_DISPLAY")
            .is_ok_and(|v| !v.is_empty());

        if is_wayland {
            let trigger = config.to_portal_trigger();
            match crate::wayland_hotkey::WaylandHotkeyListener::new(&trigger) {
                Ok(listener) => {
                    eprintln!(
                        "Using Wayland portal hotkey: {}",
                        listener.assigned_trigger
                    );
                    return Ok(HotkeyBackend::Portal { listener });
                }
                Err(e) => {
                    eprintln!("Wayland portal unavailable ({e}), trying evdev...");
                }
            }

            match crate::evdev_hotkey::EvdevHotkeyListener::new(
                &config.key_name,
                config.use_super,
                config.use_ctrl,
                config.use_shift,
                config.use_alt,
            ) {
                Ok(listener) => {
                    eprintln!("Using evdev hotkey: {}", config.display());
                    return Ok(HotkeyBackend::Evdev { listener });
                }
                Err(e) => {
                    eprintln!("evdev unavailable ({e}), falling back to X11 hotkey");
                }
            }
        }
    }

    setup_global_hotkey(config)
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

    // Hotkey management
    hotkey_backend: HotkeyBackend,
    learn_state: LearnState,

    // Paste target tracking
    paste_target_window: Option<String>,
    title_warning: Option<(String, Instant)>,

    // Wayland RemoteDesktop portal for paste simulation
    #[cfg(target_os = "linux")]
    wayland_paster: Option<crate::wayland_paste::WaylandPaster>,

    // UI
    show_settings: bool,
}

impl SpeechApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        settings: Settings,
        hotkey_backend: HotkeyBackend,
    ) -> Self {
        let (worker_tx, worker_rx, worker_handle) = spawn_inference_worker();

        #[cfg(target_os = "linux")]
        let wayland_paster = if std::env::var("WAYLAND_DISPLAY")
            .is_ok_and(|v| !v.is_empty())
        {
            match crate::wayland_paste::WaylandPaster::new() {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("RemoteDesktop portal unavailable ({e}), paste will use fallback");
                    None
                }
            }
        } else {
            None
        };

        Self {
            recorder: AudioRecorder::default(),
            worker_tx,
            worker_rx,
            worker_handle: Some(worker_handle),
            transcription: String::new(),
            partial_text: String::new(),
            recording: false,
            transcribing: false,

            pinned: settings.pinned,
            auto_action: settings.auto_action,
            hotkey_config: settings.hotkey,

            hotkey_backend,
            learn_state: LearnState::Idle,

            paste_target_window: None,
            title_warning: None,

            #[cfg(target_os = "linux")]
            wayland_paster,

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
        clipboard_copy(&self.transcription);
    }

    fn do_paste(&self) {
        #[cfg(target_os = "linux")]
        {
            crate::paste::simulate_paste(self.wayland_paster.as_ref());
        }

        #[cfg(target_os = "macos")]
        crate::paste::simulate_paste();
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
                    (None, None) => true,
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

                let old_clipboard = clipboard_get();
                self.copy_to_clipboard();
                self.do_paste();
                if let Some(old) = old_clipboard {
                    thread::spawn(move || {
                        thread::sleep(Duration::from_millis(500));
                        clipboard_copy(&old);
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
                            eprintln!("[app] Finished: {} chars", text.len());
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
                    eprintln!("[app] Partial: {} chars", text.len());
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

    fn poll_hotkey(&mut self) {
        let pressed = match &self.hotkey_backend {
            HotkeyBackend::GlobalHotkey { hotkey, .. } => {
                if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                    event.id == hotkey.id()
                        && event.state == global_hotkey::HotKeyState::Pressed
                } else {
                    false
                }
            }
            #[cfg(target_os = "linux")]
            HotkeyBackend::Portal { listener } => listener.try_recv(),
            #[cfg(target_os = "linux")]
            HotkeyBackend::Evdev { listener } => listener.try_recv(),
        };

        if pressed {
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

    // ── Hotkey management ───────────────────────────────────────────

    fn apply_learned_hotkey(&mut self, new_config: HotkeyConfig) -> bool {
        match &mut self.hotkey_backend {
            HotkeyBackend::GlobalHotkey { manager, hotkey } => {
                let Some(new_hotkey) = new_config.to_hotkey() else {
                    self.title_warning = Some((
                        format!("Key '{}' not supported on this backend", new_config.key_name),
                        Instant::now(),
                    ));
                    return false;
                };
                if let Err(e) = manager.unregister(*hotkey) {
                    eprintln!("Failed to unregister old hotkey: {e}");
                }
                match manager.register(new_hotkey) {
                    Ok(()) => {
                        *hotkey = new_hotkey;
                        self.hotkey_config = new_config;
                    }
                    Err(e) => {
                        eprintln!("Failed to register new hotkey: {e}, restoring old");
                        let _ = manager.register(*hotkey);
                        return false;
                    }
                }
            }
            #[cfg(target_os = "linux")]
            HotkeyBackend::Evdev { listener } => {
                match crate::evdev_hotkey::EvdevHotkeyListener::new(
                    &new_config.key_name,
                    new_config.use_super,
                    new_config.use_ctrl,
                    new_config.use_shift,
                    new_config.use_alt,
                ) {
                    Ok(new_listener) => {
                        *listener = new_listener;
                        self.hotkey_config = new_config;
                    }
                    Err(e) => {
                        eprintln!("Failed to apply evdev hotkey: {e}");
                        return false;
                    }
                }
            }
            #[cfg(target_os = "linux")]
            HotkeyBackend::Portal { .. } => return false,
        }
        self.save_current_settings();
        true
    }

    fn save_current_settings(&self) {
        save_settings(&Settings {
            hotkey: self.hotkey_config.clone(),
            auto_action: self.auto_action,
            pinned: self.pinned,
        });
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
        self.poll_hotkey();

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
            let was_auto_action = self.auto_action;
            let mut should_close = false;
            let hotkey_configurable = self.hotkey_backend.is_configurable();
            let hotkey_display = self.hotkey_backend.hotkey_display(&self.hotkey_config);
            let mut learned_config: Option<HotkeyConfig> = None;

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
                                    &hotkey_display,
                                    hotkey_configurable,
                                    &mut self.learn_state,
                                    &mut learned_config,
                                    &mut self.auto_action,
                                    &mut self.pinned,
                                );
                            });
                        if !open {
                            should_close = true;
                        }
                    } else {
                        egui::CentralPanel::default().show(inner_ctx, |ui| {
                            Self::draw_settings_ui(
                                ui,
                                &hotkey_display,
                                hotkey_configurable,
                                &mut self.learn_state,
                                &mut learned_config,
                                &mut self.auto_action,
                                &mut self.pinned,
                            );
                        });
                    }
                },
            );

            if should_close {
                self.show_settings = false;
                self.learn_state = LearnState::Idle;
            }
            if let Some(config) = learned_config {
                self.apply_learned_hotkey(config);
                self.learn_state = LearnState::Idle;
            }
            if self.pinned != was_pinned {
                let level = if self.pinned {
                    egui::WindowLevel::AlwaysOnTop
                } else {
                    egui::WindowLevel::Normal
                };
                ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
                self.save_current_settings();
            }
            if self.auto_action != was_auto_action {
                self.save_current_settings();
            }
        }

        // ── Main panel ──────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let icon = if self.recording {
                    "\u{23F9}"
                } else {
                    "\u{23FA}"
                };
                let tooltip = if self.recording {
                    "Stop recording".to_owned()
                } else {
                    format!(
                        "Record ({})",
                        self.hotkey_backend.hotkey_display(&self.hotkey_config)
                    )
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

                if ui
                    .add(egui::Button::new(RichText::new("\u{1F4CB}").size(18.0)))
                    .on_hover_text("Copy to clipboard")
                    .clicked()
                {
                    self.copy_to_clipboard();
                }

                if ui
                    .add(egui::Button::new(RichText::new("\u{2699}").size(18.0)))
                    .on_hover_text("Settings")
                    .clicked()
                {
                    self.show_settings = !self.show_settings;
                }
            });

            let available = ui.available_size();
            if self.recording || self.transcribing {
                // During recording/transcribing: show partial preview as a read-only label
                egui::ScrollArea::vertical()
                    .max_height(available.y)
                    .show(ui, |ui| {
                        if self.partial_text.is_empty() {
                            let status = if self.recording {
                                "Recording..."
                            } else {
                                "Transcribing..."
                            };
                            ui.label(RichText::new(status).weak().italics());
                        } else {
                            ui.label(
                                RichText::new(&self.partial_text)
                                    .weak()
                                    .italics(),
                            );
                        }
                    });
            } else {
                // After finalization: editable textarea with full text
                let mut display = self.transcription.clone();
                ui.add_sized(
                    available,
                    egui::TextEdit::multiline(&mut display).desired_width(f32::INFINITY),
                );
                self.transcription = display;
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_current_settings();
        let _ = self.worker_tx.send(InferenceCommand::Shutdown);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

// ── Settings UI ─────────────────────────────────────────────────────

impl SpeechApp {
    fn draw_settings_ui(
        ui: &mut egui::Ui,
        hotkey_display: &str,
        hotkey_configurable: bool,
        learn_state: &mut LearnState,
        learned_config: &mut Option<HotkeyConfig>,
        auto_action: &mut AutoAction,
        pinned: &mut bool,
    ) {
        ui.heading("Hotkey");
        ui.label("Global keyboard shortcut to start/stop recording.");
        ui.add_space(4.0);

        ui.label(format!("Current: {hotkey_display}"));
        ui.add_space(4.0);

        if hotkey_configurable {
            match learn_state {
                LearnState::Idle => {
                    if ui.button("Change hotkey...").clicked() {
                        #[cfg(target_os = "linux")]
                        if matches_evdev_backend() {
                            match crate::evdev_hotkey::EvdevKeyLearner::start() {
                                Ok(learner) => {
                                    *learn_state = LearnState::Evdev(learner);
                                    return;
                                }
                                Err(e) => {
                                    eprintln!("Failed to start evdev learner: {e}");
                                }
                            }
                        }
                        *learn_state = LearnState::Egui;
                    }
                }
                #[cfg(target_os = "linux")]
                LearnState::Evdev(learner) => {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Press your hotkey combination...")
                            .strong()
                            .color(egui::Color32::YELLOW),
                    );
                    ui.label(
                        RichText::new("Hold modifiers, then press the trigger key.")
                            .weak()
                            .small(),
                    );
                    ui.add_space(4.0);
                    if ui.button("Cancel").clicked() {
                        *learn_state = LearnState::Idle;
                        return;
                    }

                    if let Some(key) = learner.try_recv() {
                        *learned_config = Some(HotkeyConfig {
                            use_super: key.super_key,
                            use_ctrl: key.ctrl,
                            use_shift: key.shift,
                            use_alt: key.alt,
                            key_name: key.key_name,
                        });
                    } else {
                        // Keep polling
                        ui.ctx().request_repaint_after(Duration::from_millis(50));
                    }
                }
                LearnState::Egui => {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Press your hotkey combination...")
                            .strong()
                            .color(egui::Color32::YELLOW),
                    );
                    ui.label(
                        RichText::new("Hold modifiers, then press the trigger key.")
                            .weak()
                            .small(),
                    );
                    ui.add_space(4.0);
                    if ui.button("Cancel").clicked() {
                        *learn_state = LearnState::Idle;
                        return;
                    }

                    // Capture key from egui events
                    let events = ui.input(|i| i.events.clone());
                    for event in &events {
                        if let egui::Event::Key {
                            key,
                            pressed: true,
                            modifiers,
                            ..
                        } = event
                            && let Some(name) = egui_key_to_name(*key) {
                                *learned_config = Some(HotkeyConfig {
                                    use_super: modifiers.mac_cmd || modifiers.command,
                                    use_ctrl: modifiers.ctrl,
                                    use_shift: modifiers.shift,
                                    use_alt: modifiers.alt,
                                    key_name: name.into(),
                                });
                                break;
                            }
                    }

                    ui.ctx().request_repaint_after(Duration::from_millis(50));
                }
            }
        } else {
            ui.indent("wayland_note", |ui| {
                ui.label(
                    RichText::new(
                        "Hotkey is managed by the desktop environment via \
                         xdg-desktop-portal. Change it in your compositor settings.",
                    )
                    .weak()
                    .small(),
                );
            });
        }

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

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

/// Helper to check if the current backend is evdev (to decide learn mode).
#[cfg(target_os = "linux")]
fn matches_evdev_backend() -> bool {
    // If WAYLAND_DISPLAY is set, evdev is likely the backend
    // (portal doesn't allow configuring, global-hotkey is X11 fallback)
    std::env::var("WAYLAND_DISPLAY").is_ok_and(|v| !v.is_empty())
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
