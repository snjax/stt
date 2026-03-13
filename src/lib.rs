pub mod audio;
pub mod streaming;
pub mod whisper_cpp;

mod paste;

#[cfg(target_os = "linux")]
mod evdev_hotkey;
#[cfg(target_os = "linux")]
mod wayland_hotkey;
#[cfg(target_os = "linux")]
mod wayland_paste;

pub mod app;
