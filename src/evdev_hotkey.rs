use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use evdev::{Device, InputEventKind, Key};

pub struct EvdevHotkeyListener {
    rx: std::sync::mpsc::Receiver<()>,
}

impl EvdevHotkeyListener {
    pub fn new(
        key_name: &str,
        use_super: bool,
        use_ctrl: bool,
        use_shift: bool,
        use_alt: bool,
    ) -> Result<Self> {
        let target_key =
            key_name_to_evdev(key_name).ok_or_else(|| anyhow::anyhow!("unsupported key for evdev: {key_name}"))?;

        let keyboards: Vec<_> = evdev::enumerate()
            .filter(|(_, dev)| {
                dev.supported_keys()
                    .map_or(false, |keys| keys.contains(Key::KEY_A))
            })
            .map(|(path, _)| path)
            .collect();

        if keyboards.is_empty() {
            bail!(
                "no keyboard devices found in /dev/input/ \
                 (is your user in the 'input' group?)"
            );
        }

        eprintln!(
            "evdev: monitoring {} keyboard device(s)",
            keyboards.len()
        );

        let (tx, rx) = std::sync::mpsc::channel();

        // Debounce: multiple keyboard devices may report the same keypress
        let last_trigger = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1)));

        for path in keyboards {
            let tx = tx.clone();
            let last_trigger = Arc::clone(&last_trigger);

            thread::spawn(move || {
                let Ok(mut device) = Device::open(&path) else {
                    return;
                };
                let mut mods = ModState::default();

                loop {
                    let Ok(events) = device.fetch_events() else {
                        return;
                    };
                    for event in events {
                        if let InputEventKind::Key(key) = event.kind() {
                            let active = event.value() != 0;

                            match key {
                                Key::KEY_LEFTSHIFT | Key::KEY_RIGHTSHIFT => mods.shift = active,
                                Key::KEY_LEFTCTRL | Key::KEY_RIGHTCTRL => mods.ctrl = active,
                                Key::KEY_LEFTALT | Key::KEY_RIGHTALT => mods.alt = active,
                                Key::KEY_LEFTMETA | Key::KEY_RIGHTMETA => mods.super_key = active,
                                k if k == target_key && event.value() == 1 => {
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
                                _ => {}
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

#[derive(Default)]
struct ModState {
    shift: bool,
    ctrl: bool,
    alt: bool,
    super_key: bool,
}

fn key_name_to_evdev(name: &str) -> Option<Key> {
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
        _ => return None,
    })
}
