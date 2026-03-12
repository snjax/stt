app_name := "STT"
bundle_id := "com.snjax.stt"
version := "0.2.0"
model_name := "ggml-large-v3-turbo.bin"

# macOS paths
app_dir := "/Applications/" + app_name + ".app"
contents := app_dir + "/Contents"

# Linux paths
linux_bin_dir := env("HOME") / ".local/bin"
linux_share_dir := env("HOME") / ".local/share"
linux_model_dir := linux_share_dir / "stt/models"

# Detect OS
os := os()

# ── Build ────────────────────────────────────────────────────────────

# Build release binary with platform-appropriate features (auto-detects CUDA on Linux)
build:
    @if [ "{{os}}" = "macos" ]; then \
        cargo build --release --features metal; \
    elif [ "{{os}}" = "linux" ]; then \
        if command -v nvcc >/dev/null 2>&1; then \
            echo "CUDA detected, building with GPU support..."; \
            cargo build --release --features cuda; \
        else \
            echo "No CUDA found, building CPU-only..."; \
            cargo build --release; \
        fi; \
    else \
        echo "Unsupported OS: {{os}}"; exit 1; \
    fi

# Build release binary without GPU (CPU only)
build-cpu:
    cargo build --release

# Build debug binary
build-debug:
    cargo build

# Run cargo check and clippy
check:
    cargo check
    cargo clippy

# ── Run ──────────────────────────────────────────────────────────────

# Run the GUI (debug build)
run: build-debug
    cargo run

# Run release build directly
run-release: build
    ./target/release/stt

# Transcribe a WAV file
transcribe file: build
    ./target/release/stt --file "{{file}}"

# ── Install / Uninstall ─────────────────────────────────────────────

# Install as native app
install: build (_install-os)

# Uninstall the native app
uninstall: (_uninstall-os)

[macos]
_install-os:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "=== Installing {{app_dir}} ==="

    rm -rf "{{app_dir}}"
    mkdir -p "{{contents}}/MacOS"
    mkdir -p "{{contents}}/Resources/models"

    # Binary
    cp target/release/stt "{{contents}}/MacOS/stt"

    # Model
    if [ -f "models/{{model_name}}" ]; then
        echo "Copying model ($(du -h "models/{{model_name}}" | cut -f1))..."
        cp "models/{{model_name}}" "{{contents}}/Resources/models/{{model_name}}"
    else
        echo "WARNING: models/{{model_name}} not found"
        echo "  Run: just download-model"
    fi

    # Info.plist
    cat > "{{contents}}/Info.plist" << 'EOF'
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>CFBundleName</key>
        <string>STT</string>
        <key>CFBundleDisplayName</key>
        <string>Speech To Text</string>
        <key>CFBundleIdentifier</key>
        <string>{{bundle_id}}</string>
        <key>CFBundleVersion</key>
        <string>{{version}}</string>
        <key>CFBundleShortVersionString</key>
        <string>{{version}}</string>
        <key>CFBundleExecutable</key>
        <string>stt-launcher</string>
        <key>CFBundlePackageType</key>
        <string>APPL</string>
        <key>LSMinimumSystemVersion</key>
        <string>12.0</string>
        <key>NSMicrophoneUsageDescription</key>
        <string>STT needs microphone access to record speech for transcription.</string>
        <key>NSHighResolutionCapable</key>
        <true/>
        <key>LSUIElement</key>
        <false/>
    </dict>
    </plist>
    EOF

    # Launcher (sets cwd to Resources/ so model is found)
    cat > "{{contents}}/MacOS/stt-launcher" << 'SCRIPT'
    #!/usr/bin/env bash
    DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
    cd "$DIR"
    exec "$(dirname "$0")/stt" "$@"
    SCRIPT
    chmod +x "{{contents}}/MacOS/stt-launcher"

    echo ""
    echo "Installed {{app_dir}}"
    echo ""
    echo "Post-install:"
    echo "  1. Open STT from /Applications or Spotlight"
    echo "  2. Allow Microphone when prompted"
    echo "  3. System Settings > Privacy & Security > Accessibility > add STT.app"
    echo "     (required for global hotkey and auto-paste)"

[linux]
_install-os:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "=== Installing STT for Linux ==="

    # Binary
    mkdir -p "{{linux_bin_dir}}"
    cp target/release/stt "{{linux_bin_dir}}/stt"
    echo "Installed binary to {{linux_bin_dir}}/stt"

    # Model
    mkdir -p "{{linux_model_dir}}"
    if [ -f "models/{{model_name}}" ]; then
        echo "Copying model ($(du -h "models/{{model_name}}" | cut -f1))..."
        cp "models/{{model_name}}" "{{linux_model_dir}}/{{model_name}}"
    else
        echo "WARNING: models/{{model_name}} not found"
        echo "  Run: just download-model"
    fi

    # Desktop entry
    mkdir -p "{{linux_share_dir}}/applications"
    cat > "{{linux_share_dir}}/applications/stt.desktop" << EOF
    [Desktop Entry]
    Name=STT
    Comment=Speech to Text transcription
    Exec=env STT_WHISPER_MODEL={{linux_model_dir}}/{{model_name}} {{linux_bin_dir}}/stt
    Terminal=false
    Type=Application
    Categories=AudioVideo;Audio;Utility;
    Keywords=speech;transcription;whisper;stt;
    EOF

    echo ""
    echo "Installed:"
    echo "  Binary:  {{linux_bin_dir}}/stt"
    echo "  Model:   {{linux_model_dir}}/{{model_name}}"
    echo "  Desktop: {{linux_share_dir}}/applications/stt.desktop"
    echo ""
    echo "Make sure {{linux_bin_dir}} is in your PATH."
    echo "Run 'stt' or launch from your app menu."

[windows]
_install-os:
    @echo "Windows: not supported"
    @exit 1

[macos]
_uninstall-os:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d "{{app_dir}}" ]; then
        rm -rf "{{app_dir}}"
        echo "Removed {{app_dir}}"
    else
        echo "{{app_dir}} not found, nothing to remove"
    fi

[linux]
_uninstall-os:
    #!/usr/bin/env bash
    set -euo pipefail
    removed=0
    for f in "{{linux_bin_dir}}/stt" \
             "{{linux_share_dir}}/applications/stt.desktop"; do
        if [ -f "$f" ]; then
            rm "$f"
            echo "Removed $f"
            removed=1
        fi
    done
    if [ -d "{{linux_model_dir}}" ]; then
        rm -rf "{{linux_model_dir}}"
        echo "Removed {{linux_model_dir}}"
        removed=1
    fi
    # Clean up empty parent dir
    rmdir "{{linux_share_dir}}/stt" 2>/dev/null || true
    if [ "$removed" = 0 ]; then
        echo "Nothing to uninstall"
    fi

[windows]
_uninstall-os:
    @echo "Windows: not supported"
    @exit 1

# ── Model ────────────────────────────────────────────────────────────

# Download the whisper GGML model
download-model:
    mkdir -p models
    curl -L -o "models/{{model_name}}" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{{model_name}}"

# ── Helpers ──────────────────────────────────────────────────────────

# Show all available recipes
help:
    @just --list
