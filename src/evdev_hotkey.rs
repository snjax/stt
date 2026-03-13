use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use evdev::{Device, InputEventKind, Key};

// ── Shared modifier state ──────────────────────────────────────────

#[derive(Default, Clone)]
pub struct ModState {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

fn is_modifier(key: Key) -> bool {
    matches!(
        key,
        Key::KEY_LEFTSHIFT
            | Key::KEY_RIGHTSHIFT
            | Key::KEY_LEFTCTRL
            | Key::KEY_RIGHTCTRL
            | Key::KEY_LEFTALT
            | Key::KEY_RIGHTALT
            | Key::KEY_LEFTMETA
            | Key::KEY_RIGHTMETA
    )
}

fn update_mods(mods: &mut ModState, key: Key, active: bool) {
    match key {
        Key::KEY_LEFTSHIFT | Key::KEY_RIGHTSHIFT => mods.shift = active,
        Key::KEY_LEFTCTRL | Key::KEY_RIGHTCTRL => mods.ctrl = active,
        Key::KEY_LEFTALT | Key::KEY_RIGHTALT => mods.alt = active,
        Key::KEY_LEFTMETA | Key::KEY_RIGHTMETA => mods.super_key = active,
        _ => {}
    }
}

// ── Key name mapping ───────────────────────────────────────────────

pub fn evdev_key_to_name(key: Key) -> String {
    match key {
        // Letters
        Key::KEY_A => "A".into(),
        Key::KEY_B => "B".into(),
        Key::KEY_C => "C".into(),
        Key::KEY_D => "D".into(),
        Key::KEY_E => "E".into(),
        Key::KEY_F => "F".into(),
        Key::KEY_G => "G".into(),
        Key::KEY_H => "H".into(),
        Key::KEY_I => "I".into(),
        Key::KEY_J => "J".into(),
        Key::KEY_K => "K".into(),
        Key::KEY_L => "L".into(),
        Key::KEY_M => "M".into(),
        Key::KEY_N => "N".into(),
        Key::KEY_O => "O".into(),
        Key::KEY_P => "P".into(),
        Key::KEY_Q => "Q".into(),
        Key::KEY_R => "R".into(),
        Key::KEY_S => "S".into(),
        Key::KEY_T => "T".into(),
        Key::KEY_U => "U".into(),
        Key::KEY_V => "V".into(),
        Key::KEY_W => "W".into(),
        Key::KEY_X => "X".into(),
        Key::KEY_Y => "Y".into(),
        Key::KEY_Z => "Z".into(),
        // Digits
        Key::KEY_1 => "1".into(),
        Key::KEY_2 => "2".into(),
        Key::KEY_3 => "3".into(),
        Key::KEY_4 => "4".into(),
        Key::KEY_5 => "5".into(),
        Key::KEY_6 => "6".into(),
        Key::KEY_7 => "7".into(),
        Key::KEY_8 => "8".into(),
        Key::KEY_9 => "9".into(),
        Key::KEY_0 => "0".into(),
        // Function keys
        Key::KEY_F1 => "F1".into(),
        Key::KEY_F2 => "F2".into(),
        Key::KEY_F3 => "F3".into(),
        Key::KEY_F4 => "F4".into(),
        Key::KEY_F5 => "F5".into(),
        Key::KEY_F6 => "F6".into(),
        Key::KEY_F7 => "F7".into(),
        Key::KEY_F8 => "F8".into(),
        Key::KEY_F9 => "F9".into(),
        Key::KEY_F10 => "F10".into(),
        Key::KEY_F11 => "F11".into(),
        Key::KEY_F12 => "F12".into(),
        // Navigation
        Key::KEY_UP => "Up".into(),
        Key::KEY_DOWN => "Down".into(),
        Key::KEY_LEFT => "Left".into(),
        Key::KEY_RIGHT => "Right".into(),
        Key::KEY_HOME => "Home".into(),
        Key::KEY_END => "End".into(),
        Key::KEY_PAGEUP => "PageUp".into(),
        Key::KEY_PAGEDOWN => "PageDown".into(),
        Key::KEY_INSERT => "Insert".into(),
        Key::KEY_DELETE => "Delete".into(),
        // Special
        Key::KEY_SPACE => "Space".into(),
        Key::KEY_ENTER => "Enter".into(),
        Key::KEY_TAB => "Tab".into(),
        Key::KEY_ESC => "Escape".into(),
        Key::KEY_BACKSPACE => "Backspace".into(),
        Key::KEY_CAPSLOCK => "CapsLock".into(),
        // Punctuation
        Key::KEY_MINUS => "Minus".into(),
        Key::KEY_EQUAL => "Equal".into(),
        Key::KEY_LEFTBRACE => "LeftBracket".into(),
        Key::KEY_RIGHTBRACE => "RightBracket".into(),
        Key::KEY_BACKSLASH => "Backslash".into(),
        Key::KEY_SEMICOLON => "Semicolon".into(),
        Key::KEY_APOSTROPHE => "Apostrophe".into(),
        Key::KEY_GRAVE => "Grave".into(),
        Key::KEY_COMMA => "Comma".into(),
        Key::KEY_DOT => "Dot".into(),
        Key::KEY_SLASH => "Slash".into(),
        // Media
        Key::KEY_PLAYPAUSE => "PlayPause".into(),
        Key::KEY_NEXTSONG => "NextSong".into(),
        Key::KEY_PREVIOUSSONG => "PrevSong".into(),
        Key::KEY_STOPCD => "StopMedia".into(),
        Key::KEY_VOLUMEUP => "VolumeUp".into(),
        Key::KEY_VOLUMEDOWN => "VolumeDown".into(),
        Key::KEY_MUTE => "Mute".into(),
        // Numpad
        Key::KEY_KP0 => "Num0".into(),
        Key::KEY_KP1 => "Num1".into(),
        Key::KEY_KP2 => "Num2".into(),
        Key::KEY_KP3 => "Num3".into(),
        Key::KEY_KP4 => "Num4".into(),
        Key::KEY_KP5 => "Num5".into(),
        Key::KEY_KP6 => "Num6".into(),
        Key::KEY_KP7 => "Num7".into(),
        Key::KEY_KP8 => "Num8".into(),
        Key::KEY_KP9 => "Num9".into(),
        Key::KEY_KPENTER => "NumEnter".into(),
        Key::KEY_KPPLUS => "NumPlus".into(),
        Key::KEY_KPMINUS => "NumMinus".into(),
        Key::KEY_KPASTERISK => "NumMultiply".into(),
        Key::KEY_KPSLASH => "NumDivide".into(),
        Key::KEY_KPDOT => "NumDot".into(),
        // Mouse buttons
        Key::BTN_SIDE => "BtnSide".into(),
        Key::BTN_EXTRA => "BtnExtra".into(),
        Key::BTN_FORWARD => "BtnForward".into(),
        Key::BTN_BACK => "BtnBack".into(),
        Key::BTN_MIDDLE => "BtnMiddle".into(),
        // Fallback: raw code
        other => format!("Evdev({})", other.0),
    }
}

pub fn name_to_evdev_key(name: &str) -> Option<Key> {
    Some(match name {
        "A" => Key::KEY_A,
        "B" => Key::KEY_B,
        "C" => Key::KEY_C,
        "D" => Key::KEY_D,
        "E" => Key::KEY_E,
        "F" => Key::KEY_F,
        "G" => Key::KEY_G,
        "H" => Key::KEY_H,
        "I" => Key::KEY_I,
        "J" => Key::KEY_J,
        "K" => Key::KEY_K,
        "L" => Key::KEY_L,
        "M" => Key::KEY_M,
        "N" => Key::KEY_N,
        "O" => Key::KEY_O,
        "P" => Key::KEY_P,
        "Q" => Key::KEY_Q,
        "R" => Key::KEY_R,
        "S" => Key::KEY_S,
        "T" => Key::KEY_T,
        "U" => Key::KEY_U,
        "V" => Key::KEY_V,
        "W" => Key::KEY_W,
        "X" => Key::KEY_X,
        "Y" => Key::KEY_Y,
        "Z" => Key::KEY_Z,
        "1" => Key::KEY_1,
        "2" => Key::KEY_2,
        "3" => Key::KEY_3,
        "4" => Key::KEY_4,
        "5" => Key::KEY_5,
        "6" => Key::KEY_6,
        "7" => Key::KEY_7,
        "8" => Key::KEY_8,
        "9" => Key::KEY_9,
        "0" => Key::KEY_0,
        "F1" => Key::KEY_F1,
        "F2" => Key::KEY_F2,
        "F3" => Key::KEY_F3,
        "F4" => Key::KEY_F4,
        "F5" => Key::KEY_F5,
        "F6" => Key::KEY_F6,
        "F7" => Key::KEY_F7,
        "F8" => Key::KEY_F8,
        "F9" => Key::KEY_F9,
        "F10" => Key::KEY_F10,
        "F11" => Key::KEY_F11,
        "F12" => Key::KEY_F12,
        "Up" => Key::KEY_UP,
        "Down" => Key::KEY_DOWN,
        "Left" => Key::KEY_LEFT,
        "Right" => Key::KEY_RIGHT,
        "Home" => Key::KEY_HOME,
        "End" => Key::KEY_END,
        "PageUp" => Key::KEY_PAGEUP,
        "PageDown" => Key::KEY_PAGEDOWN,
        "Insert" => Key::KEY_INSERT,
        "Delete" => Key::KEY_DELETE,
        "Space" => Key::KEY_SPACE,
        "Enter" => Key::KEY_ENTER,
        "Tab" => Key::KEY_TAB,
        "Escape" => Key::KEY_ESC,
        "Backspace" => Key::KEY_BACKSPACE,
        "CapsLock" => Key::KEY_CAPSLOCK,
        "Minus" => Key::KEY_MINUS,
        "Equal" => Key::KEY_EQUAL,
        "LeftBracket" => Key::KEY_LEFTBRACE,
        "RightBracket" => Key::KEY_RIGHTBRACE,
        "Backslash" => Key::KEY_BACKSLASH,
        "Semicolon" => Key::KEY_SEMICOLON,
        "Apostrophe" => Key::KEY_APOSTROPHE,
        "Grave" => Key::KEY_GRAVE,
        "Comma" => Key::KEY_COMMA,
        "Dot" => Key::KEY_DOT,
        "Slash" => Key::KEY_SLASH,
        "PlayPause" => Key::KEY_PLAYPAUSE,
        "NextSong" => Key::KEY_NEXTSONG,
        "PrevSong" => Key::KEY_PREVIOUSSONG,
        "StopMedia" => Key::KEY_STOPCD,
        "VolumeUp" => Key::KEY_VOLUMEUP,
        "VolumeDown" => Key::KEY_VOLUMEDOWN,
        "Mute" => Key::KEY_MUTE,
        "Num0" => Key::KEY_KP0,
        "Num1" => Key::KEY_KP1,
        "Num2" => Key::KEY_KP2,
        "Num3" => Key::KEY_KP3,
        "Num4" => Key::KEY_KP4,
        "Num5" => Key::KEY_KP5,
        "Num6" => Key::KEY_KP6,
        "Num7" => Key::KEY_KP7,
        "Num8" => Key::KEY_KP8,
        "Num9" => Key::KEY_KP9,
        "NumEnter" => Key::KEY_KPENTER,
        "NumPlus" => Key::KEY_KPPLUS,
        "NumMinus" => Key::KEY_KPMINUS,
        "NumMultiply" => Key::KEY_KPASTERISK,
        "NumDivide" => Key::KEY_KPSLASH,
        "NumDot" => Key::KEY_KPDOT,
        "BtnSide" => Key::BTN_SIDE,
        "BtnExtra" => Key::BTN_EXTRA,
        "BtnForward" => Key::BTN_FORWARD,
        "BtnBack" => Key::BTN_BACK,
        "BtnMiddle" => Key::BTN_MIDDLE,
        other => {
            // Parse "Evdev(123)" format
            if let Some(code) = other
                .strip_prefix("Evdev(")
                .and_then(|s| s.strip_suffix(")"))
                .and_then(|s| s.parse::<u16>().ok())
            {
                return Some(Key::new(code));
            }
            return None;
        }
    })
}

// ── Hotkey listener ────────────────────────────────────────────────

pub struct EvdevHotkeyListener {
    rx: mpsc::Receiver<()>,
}

impl EvdevHotkeyListener {
    pub fn new(
        key_name: &str,
        use_super: bool,
        use_ctrl: bool,
        use_shift: bool,
        use_alt: bool,
    ) -> Result<Self> {
        let target_key = name_to_evdev_key(key_name)
            .ok_or_else(|| anyhow::anyhow!("unsupported key for evdev: {key_name}"))?;

        // Monitor keyboards (for modifiers) + any device with the target key
        let devices: Vec<_> = evdev::enumerate()
            .filter(|(_, dev)| {
                dev.supported_keys().is_some_and(|keys| {
                    keys.contains(Key::KEY_A) || keys.contains(target_key)
                })
            })
            .map(|(path, _)| path)
            .collect();

        if devices.is_empty() {
            bail!(
                "no input devices found in /dev/input/ \
                 (is your user in the 'input' group?)"
            );
        }

        eprintln!(
            "evdev: monitoring {} device(s) for {}",
            devices.len(),
            key_name,
        );

        let (tx, rx) = mpsc::channel();
        let shared_mods = Arc::new(Mutex::new(ModState::default()));
        let last_trigger = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1)));

        for path in devices {
            let tx = tx.clone();
            let shared_mods = Arc::clone(&shared_mods);
            let last_trigger = Arc::clone(&last_trigger);

            thread::spawn(move || {
                let Ok(mut device) = Device::open(&path) else {
                    return;
                };
                loop {
                    let Ok(events) = device.fetch_events() else {
                        return;
                    };
                    for event in events {
                        if let InputEventKind::Key(key) = event.kind() {
                            let active = event.value() != 0;

                            if is_modifier(key) {
                                if let Ok(mut mods) = shared_mods.lock() {
                                    update_mods(&mut mods, key, active);
                                }
                                continue;
                            }

                            if key == target_key && event.value() == 1 {
                                let mods = match shared_mods.lock() {
                                    Ok(m) => m.clone(),
                                    Err(_) => return,
                                };
                                if mods.super_key == use_super
                                    && mods.ctrl == use_ctrl
                                    && mods.shift == use_shift
                                    && mods.alt == use_alt
                                {
                                    let Ok(mut last) = last_trigger.lock() else {
                                        return;
                                    };
                                    if last.elapsed() > Duration::from_millis(200) {
                                        *last = Instant::now();
                                        if tx.send(()).is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        Ok(Self { rx })
    }

    pub fn try_recv(&self) -> bool {
        self.rx.try_recv().is_ok()
    }
}

// ── Key learner ────────────────────────────────────────────────────

pub struct LearnedKey {
    pub key_name: String,
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

pub struct EvdevKeyLearner {
    rx: mpsc::Receiver<LearnedKey>,
    stop: Arc<AtomicBool>,
}

impl EvdevKeyLearner {
    /// Start listening on all input devices for the next non-modifier keypress.
    pub fn start() -> Result<Self> {
        let devices: Vec<_> = evdev::enumerate()
            .filter(|(_, dev)| dev.supported_keys().is_some_and(|keys| keys.iter().next().is_some()))
            .map(|(path, _)| path)
            .collect();

        if devices.is_empty() {
            bail!("no input devices found (is your user in the 'input' group?)");
        }

        let (tx, rx) = mpsc::channel::<LearnedKey>();
        let stop = Arc::new(AtomicBool::new(false));
        let shared_mods = Arc::new(Mutex::new(ModState::default()));
        let sent = Arc::new(AtomicBool::new(false));

        for path in devices {
            let tx = tx.clone();
            let stop = Arc::clone(&stop);
            let shared_mods = Arc::clone(&shared_mods);
            let sent = Arc::clone(&sent);

            thread::spawn(move || {
                let Ok(mut device) = Device::open(&path) else {
                    return;
                };
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let Ok(events) = device.fetch_events() else {
                        return;
                    };
                    for event in events {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        if let InputEventKind::Key(key) = event.kind() {
                            let active = event.value() != 0;

                            if is_modifier(key) {
                                if let Ok(mut mods) = shared_mods.lock() {
                                    update_mods(&mut mods, key, active);
                                }
                                continue;
                            }

                            // Non-modifier key pressed
                            if event.value() == 1
                                && !sent.swap(true, Ordering::Relaxed)
                            {
                                let mods = shared_mods.lock().map(|m| m.clone()).unwrap_or_default();
                                let key_name = evdev_key_to_name(key);
                                let _ = tx.send(LearnedKey {
                                    key_name,
                                    shift: mods.shift,
                                    ctrl: mods.ctrl,
                                    alt: mods.alt,
                                    super_key: mods.super_key,
                                });
                                stop.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            });
        }

        Ok(Self { rx, stop })
    }

    pub fn try_recv(&self) -> Option<LearnedKey> {
        self.rx.try_recv().ok()
    }
}

impl Drop for EvdevKeyLearner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
