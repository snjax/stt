use std::{process::Command, thread, time::Duration};

#[cfg(target_os = "linux")]
use std::{io::Write, process::Stdio};

/// Get an identifier for the currently focused window.
/// Returns None if detection fails.
pub fn get_active_window() -> Option<String> {
    #[cfg(target_os = "macos")]
    return macos_active_window();

    #[cfg(target_os = "linux")]
    return linux_active_window();
}

/// Simulate Ctrl+V / Cmd+V keystroke in the currently focused window.
/// On Wayland, uses the RemoteDesktop portal session if available.
/// Small delay before paste to let the OS settle after clipboard write.
#[cfg(target_os = "linux")]
pub fn simulate_paste(wayland_paster: Option<&crate::wayland_paste::WaylandPaster>) {
    thread::sleep(Duration::from_millis(50));
    linux_paste(wayland_paster);
}

#[cfg(target_os = "macos")]
pub fn simulate_paste() {
    thread::sleep(Duration::from_millis(50));
    macos_paste();
}

pub fn clipboard_copy(text: &str) {
    #[cfg(target_os = "macos")]
    macos_clipboard_copy(text);

    #[cfg(target_os = "linux")]
    linux_clipboard_copy(text);
}

pub fn clipboard_get() -> Option<String> {
    #[cfg(target_os = "macos")]
    return macos_clipboard_get();

    #[cfg(target_os = "linux")]
    return linux_clipboard_get();
}

#[cfg(target_os = "macos")]
fn macos_active_window() -> Option<String> {
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
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        wayland_active_window()
    } else {
        x11_active_window()
    }
}

#[cfg(target_os = "linux")]
fn x11_active_window() -> Option<String> {
    let output = Command::new("xdotool")
        .arg("getactivewindow")
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "linux")]
fn wayland_active_window() -> Option<String> {
    // Hyprland
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
        && let Some(w) = hyprland_active_window()
    {
        return Some(w);
    }

    // Sway / wlroots
    if std::env::var("SWAYSOCK").is_ok()
        && let Some(w) = sway_active_window()
    {
        return Some(w);
    }

    // KDE Plasma
    if std::env::var("KDE_SESSION_VERSION").is_ok()
        && let Some(w) = kde_active_window()
    {
        return Some(w);
    }

    x11_active_window()
}

#[cfg(target_os = "linux")]
fn hyprland_active_window() -> Option<String> {
    let output = Command::new("hyprctl")
        .args(["activewindow", "-j"])
        .output()
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let class = json.get("class")?.as_str()?;
    let title = json.get("title").and_then(|v| v.as_str()).unwrap_or("");
    Some(format!("{class}|{title}"))
}

#[cfg(target_os = "linux")]
fn sway_active_window() -> Option<String> {
    let output = Command::new("swaymsg")
        .args(["-t", "get_tree"])
        .output()
        .ok()?;
    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    find_focused_sway(&tree)
}

#[cfg(target_os = "linux")]
fn find_focused_sway(node: &serde_json::Value) -> Option<String> {
    if node.get("focused").and_then(|v| v.as_bool()) == Some(true) {
        let name = node.get("name").and_then(|v| v.as_str()).unwrap_or("");
        // Native Wayland windows use app_id
        if let Some(app_id) = node.get("app_id").and_then(|v| v.as_str())
            && !app_id.is_empty()
        {
            return Some(format!("{app_id}|{name}"));
        }
        // XWayland windows use window_properties.class
        if let Some(props) = node.get("window_properties")
            && let Some(class) = props.get("class").and_then(|v| v.as_str())
            && !class.is_empty()
        {
            return Some(format!("{class}|{name}"));
        }
    }

    for key in &["nodes", "floating_nodes"] {
        if let Some(children) = node.get(key).and_then(|v| v.as_array()) {
            for child in children {
                if let Some(result) = find_focused_sway(child) {
                    return Some(result);
                }
            }
        }
    }

    None
}

#[cfg(target_os = "linux")]
fn kde_active_window() -> Option<String> {
    let output = Command::new("kdotool")
        .arg("getactivewindow")
        .output()
        .ok()?;
    let id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if id.is_empty() {
        return None;
    }

    let class_output = Command::new("kdotool")
        .args(["getwindowclassname", &id])
        .output()
        .ok()?;
    let class = String::from_utf8_lossy(&class_output.stdout).trim().to_owned();
    Some(format!("{class}|{id}"))
}

#[cfg(target_os = "linux")]
fn linux_paste(wayland_paster: Option<&crate::wayland_paste::WaylandPaster>) {
    // Try RemoteDesktop portal first (proper Wayland way)
    if let Some(paster) = wayland_paster {
        if paster.paste() {
            return;
        }
        eprintln!("RemoteDesktop portal paste failed, falling back");
    }

    if is_wayland() {
        if !wtype_paste() {
            let _ = xdotool_paste();
        }
    } else {
        let _ = xdotool_paste();
    }
}

#[cfg(target_os = "macos")]
fn macos_clipboard_copy(text: &str) {
    let _ = arboard::Clipboard::new().and_then(|mut cb| cb.set_text(text.to_owned()));
}

#[cfg(target_os = "macos")]
fn macos_clipboard_get() -> Option<String> {
    arboard::Clipboard::new()
        .and_then(|mut cb| cb.get_text())
        .ok()
}

#[cfg(target_os = "linux")]
fn linux_clipboard_copy(text: &str) {
    if is_wayland() && wl_copy_text(text) {
        return;
    }

    arboard_clipboard_copy(text);
}

#[cfg(target_os = "linux")]
fn linux_clipboard_get() -> Option<String> {
    if is_wayland() && let Some(text) = wl_paste_text() {
        return Some(text);
    }

    arboard_clipboard_get()
}

#[cfg(target_os = "linux")]
fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

#[cfg(target_os = "linux")]
fn wtype_paste() -> bool {
    Command::new("wtype")
        .args(["-M", "ctrl", "-k", "v"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn xdotool_paste() -> bool {
    Command::new("xdotool")
        .args(["key", "ctrl+v"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn wl_copy_text(text: &str) -> bool {
    let mut child = match Command::new("wl-copy").stdin(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(_) => return false,
    };

    let Some(mut stdin) = child.stdin.take() else {
        let _ = child.wait();
        return false;
    };

    if stdin.write_all(text.as_bytes()).is_err() {
        drop(stdin);
        let _ = child.wait();
        return false;
    }
    drop(stdin);

    child.wait().map(|status| status.success()).unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn wl_paste_text() -> Option<String> {
    let output = Command::new("wl-paste").output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout).ok()
}

#[cfg(target_os = "linux")]
fn arboard_clipboard_copy(text: &str) {
    let _ = arboard::Clipboard::new().and_then(|mut cb| cb.set_text(text.to_owned()));
}

#[cfg(target_os = "linux")]
fn arboard_clipboard_get() -> Option<String> {
    arboard::Clipboard::new()
        .and_then(|mut cb| cb.get_text())
        .ok()
}
