#!/usr/bin/env bash
# Test script for clipboard, paste simulation, and active window detection on Wayland
set -u

PASS=0
FAIL=0
SKIP=0

result() {
    local status="$1" name="$2" detail="${3:-}"
    case "$status" in
        PASS) ((PASS++)); printf "\033[32mPASS\033[0m %s %s\n" "$name" "$detail" ;;
        FAIL) ((FAIL++)); printf "\033[31mFAIL\033[0m %s %s\n" "$name" "$detail" ;;
        SKIP) ((SKIP++)); printf "\033[33mSKIP\033[0m %s %s\n" "$name" "$detail" ;;
    esac
}

echo "=== Environment ==="
echo "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
echo "DISPLAY=${DISPLAY:-unset}"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-unset}"
echo "XDG_CURRENT_DESKTOP=${XDG_CURRENT_DESKTOP:-unset}"
echo ""

# ── 1. Active window detection ──────────────────────────────────────
echo "=== Active Window Detection ==="

# Hyprland
if [ -n "${HYPRLAND_INSTANCE_SIGNATURE:-}" ]; then
    out=$(hyprctl activewindow -j 2>/dev/null)
    if [ -n "$out" ] && echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('class',''))" 2>/dev/null | grep -q .; then
        result PASS "hyprland_active_window" "$(echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('class',''))")"
    else
        result FAIL "hyprland_active_window" "empty or parse error"
    fi
else
    result SKIP "hyprland_active_window" "not Hyprland"
fi

# Sway
if [ -n "${SWAYSOCK:-}" ]; then
    out=$(swaymsg -t get_tree 2>/dev/null | python3 -c "
import sys,json
def find(n):
    if n.get('focused'): return n.get('app_id') or (n.get('window_properties',{}).get('class',''))
    for k in ('nodes','floating_nodes'):
        for c in n.get(k,[]):
            r = find(c)
            if r: return r
    return ''
print(find(json.load(sys.stdin)))" 2>/dev/null)
    if [ -n "$out" ]; then
        result PASS "sway_active_window" "$out"
    else
        result FAIL "sway_active_window" "empty"
    fi
else
    result SKIP "sway_active_window" "not Sway"
fi

# KDE
if [ -n "${KDE_SESSION_VERSION:-}" ]; then
    out=$(kdotool getactivewindow 2>/dev/null)
    if [ -n "$out" ]; then
        result PASS "kde_active_window" "$out"
    else
        result FAIL "kde_active_window" "empty"
    fi
else
    result SKIP "kde_active_window" "not KDE"
fi

# GNOME (no standard method)
if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]]; then
    # Try the focused-window-dbus extension
    out=$(gdbus call --session --dest org.gnome.Shell \
        --object-path /org/gnome/shell/extensions/FocusedWindow \
        --method org.gnome.shell.extensions.FocusedWindow.Get 2>/dev/null)
    if [ -n "$out" ]; then
        result PASS "gnome_active_window (extension)" "$out"
    else
        result FAIL "gnome_active_window" "no focused-window-dbus extension or GNOME method available"
    fi
fi

# xdotool (X11/XWayland)
if command -v xdotool >/dev/null 2>&1; then
    out=$(xdotool getactivewindow 2>/dev/null)
    if [ -n "$out" ]; then
        result PASS "xdotool_active_window" "window_id=$out"
    else
        result FAIL "xdotool_active_window" "failed (expected on pure Wayland)"
    fi
fi

echo ""

# ── 2. Clipboard copy ──────────────────────────────────────────────
echo "=== Clipboard Copy ==="

TEST_STRING="stt_test_$(date +%s)"

# wl-copy + wl-paste
if command -v wl-copy >/dev/null 2>&1 && command -v wl-paste >/dev/null 2>&1; then
    echo -n "$TEST_STRING" | wl-copy 2>/dev/null
    sleep 0.2
    got=$(wl-paste --no-newline 2>/dev/null)
    if [ "$got" = "$TEST_STRING" ]; then
        result PASS "wl-copy/wl-paste" "round-trip OK"
    else
        result FAIL "wl-copy/wl-paste" "expected='$TEST_STRING' got='$got'"
    fi
else
    result SKIP "wl-copy/wl-paste" "not installed"
fi

# xsel
if command -v xsel >/dev/null 2>&1; then
    echo -n "${TEST_STRING}_xsel" | xsel --clipboard --input 2>/dev/null
    sleep 0.2
    got=$(xsel --clipboard --output 2>/dev/null)
    if [ "$got" = "${TEST_STRING}_xsel" ]; then
        result PASS "xsel" "round-trip OK"
    else
        result FAIL "xsel" "expected='${TEST_STRING}_xsel' got='$got'"
    fi
else
    result SKIP "xsel" "not installed"
fi

# Test arboard via a small Rust program
cat > /tmp/test_arboard.rs << 'RUSTEOF'
use std::process::ExitCode;
fn main() -> ExitCode {
    let test_str = format!("arboard_test_{}", std::process::id());
    match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(test_str.clone())) {
        Ok(()) => {
            std::thread::sleep(std::time::Duration::from_millis(200));
            match arboard::Clipboard::new().and_then(|mut cb| cb.get_text()) {
                Ok(got) if got == test_str => {
                    eprintln!("PASS arboard round-trip");
                    ExitCode::SUCCESS
                }
                Ok(got) => {
                    eprintln!("FAIL arboard: expected='{}' got='{}'", test_str, got);
                    ExitCode::from(1)
                }
                Err(e) => {
                    eprintln!("FAIL arboard get_text: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        Err(e) => {
            eprintln!("FAIL arboard set_text: {}", e);
            ExitCode::from(1)
        }
    }
}
RUSTEOF
echo "(arboard test requires compilation — skipping inline, test in cargo)"

echo ""

# ── 3. Paste simulation ────────────────────────────────────────────
echo "=== Paste Simulation (tool availability) ==="

for cmd in wtype ydotool dotool xdotool; do
    if command -v "$cmd" >/dev/null 2>&1; then
        result PASS "$cmd" "available"
    else
        result FAIL "$cmd" "not installed"
    fi
done

echo ""

# ── Summary ─────────────────────────────────────────────────────────
echo "=== Summary ==="
echo "PASS=$PASS FAIL=$FAIL SKIP=$SKIP"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
