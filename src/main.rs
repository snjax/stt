mod app;
mod audio;
#[cfg(target_os = "linux")]
mod evdev_hotkey;
mod paste;
mod streaming;
#[cfg(target_os = "linux")]
mod wayland_hotkey;
#[cfg(target_os = "linux")]
mod wayland_paste;
mod whisper_cpp;

use std::{
    env,
    path::{Path, PathBuf},
    process,
};

use anyhow::{Context, Result, bail};

use crate::{
    app::run_gui,
    whisper_cpp::WhisperCppTranscriber,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:?}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let file_path = parse_args()?;

    match file_path {
        Some(path) => transcribe_file(&path),
        None => run_gui(),
    }
}

fn parse_args() -> Result<Option<PathBuf>> {
    let mut file_path: Option<PathBuf> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => {
                let path = args
                    .next()
                    .map(PathBuf::from)
                    .context("missing path after --file")?;

                if file_path.replace(path).is_some() {
                    bail!("`--file` may only be provided once");
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => bail!("unknown argument: {other}. Supported usage: {}", usage()),
        }
    }

    Ok(file_path)
}

fn transcribe_file(path: &Path) -> Result<()> {
    let transcriber = WhisperCppTranscriber::new()?;
    let text = transcriber.transcribe_wav_file(path)?;
    println!("{text}");
    Ok(())
}

fn usage() -> &'static str {
    "stt [--file path/to.wav]"
}
