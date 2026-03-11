use std::{process::Command, thread, time::Duration};

/// Get an identifier for the currently focused window.
/// Returns None if detection fails.
pub fn get_active_window() -> Option<String> {
    #[cfg(target_os = "macos")]
    return macos_active_window();

    #[cfg(target_os = "linux")]
    return linux_active_window();
}

/// Simulate Ctrl+V / Cmd+V keystroke in the currently focused window.
/// Small delay before paste to let the OS settle after clipboard write.
pub fn simulate_paste() {
    thread::sleep(Duration::from_millis(50));

    #[cfg(target_os = "macos")]
    macos_paste();

    #[cfg(target_os = "linux")]
    linux_paste();
}

#[cfg(target_os = "macos")]
fn macos_active_window() -> Option<String> {
    // Get bundle identifier + window title of frontmost app
    let output = Command::new("osascript")
        .arg("-e")
        .arg(concat!(
            "tell application \"System Events\"\n",
            "  set fp to first process whose frontmost is true\n",
            "  set bid to bundle identifier of fp\n",
            "  set wt to \"\"\n",
            "  try\n",
            "    set wt to name of front window of fp\n",
            "  end try\n",
            "  return bid & \"|\" & wt\n",
            "end tell",
        ))
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "macos")]
fn macos_paste() {
    let _ = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"System Events\" to keystroke \"v\" using command down")
        .output();
}

#[cfg(target_os = "linux")]
fn linux_active_window() -> Option<String> {
    let wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    if wayland {
        // No reliable universal way on Wayland; return session-level id
        None
    } else {
        let output = Command::new("xdotool")
            .arg("getactivewindow")
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if s.is_empty() { None } else { Some(s) }
    }
}

#[cfg(target_os = "linux")]
fn linux_paste() {
    let wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    if wayland {
        let _ = Command::new("wtype").arg("-M").arg("ctrl").arg("-k").arg("v").output();
    } else {
        let _ = Command::new("xdotool").arg("key").arg("ctrl+v").output();
    }
}
