use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_WHISPER_MODEL: &str = "mlx-community/whisper-large-v3-turbo";

pub struct WhisperTranscriber {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Serialize)]
struct RpcRequest<T> {
    method: &'static str,
    params: T,
}

#[derive(Deserialize)]
struct RpcResponse {
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize)]
struct InitParams<'a> {
    model: &'a str,
}

#[derive(Serialize)]
struct AudioPathParams<'a> {
    audio_path: &'a Path,
}

#[derive(Serialize)]
struct EmptyParams {}

#[derive(Deserialize)]
struct TranscriptionResult {
    text: String,
    #[allow(dead_code)]
    language: Option<String>,
}

impl WhisperTranscriber {
    pub fn new() -> Result<Self> {
        let script_path = find_server_script()?;
        let python = find_python(&script_path);

        let mut child = Command::new(&python)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start whisper server via `{python} {}`",
                    script_path.display()
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .context("failed to capture whisper server stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to capture whisper server stdout")?;

        let mut transcriber = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        transcriber.send_request(
            "init",
            InitParams {
                model: DEFAULT_WHISPER_MODEL,
            },
        )?;
        Ok(transcriber)
    }

    pub fn transcribe_wav_file(&mut self, path: &Path) -> Result<String> {
        let result = self.send_request("transcribe", AudioPathParams { audio_path: path })?;
        let result: TranscriptionResult =
            serde_json::from_value(result).context("invalid whisper transcribe response")?;
        Ok(result.text)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let wav_path = write_temp_wav(samples)?;
        let transcription = self.transcribe_wav_file(&wav_path);

        if let Err(error) = fs::remove_file(&wav_path) {
            let cleanup_error =
                anyhow!(error).context(format!("failed to remove temporary wav {}", wav_path.display()));
            return match transcription {
                Ok(_) => Err(cleanup_error),
                Err(transcription_error) => Err(transcription_error.context(cleanup_error)),
            };
        }

        transcription
    }

    fn send_request<T>(&mut self, method: &'static str, params: T) -> Result<Value>
    where
        T: Serialize,
    {
        let request = RpcRequest { method, params };
        let encoded = serde_json::to_string(&request).context("failed to encode whisper request")?;
        writeln!(self.stdin, "{encoded}").context("failed to write whisper request")?;
        self.stdin.flush().context("failed to flush whisper request")?;

        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .context("failed to read whisper response")?;
        if read == 0 {
            let status = self
                .child
                .try_wait()
                .context("failed to query whisper server status")?;
            bail!("whisper server closed stdout unexpectedly (status: {status:?})");
        }

        let response: RpcResponse =
            serde_json::from_str(&line).context("failed to decode whisper response")?;

        if let Some(error) = response.error {
            bail!("whisper server error: {error}");
        }

        response
            .result
            .ok_or_else(|| anyhow!("whisper server response missing `result`"))
    }
}

impl Drop for WhisperTranscriber {
    fn drop(&mut self) {
        let _ = self.send_request("shutdown", EmptyParams {});
        let _ = self.child.wait();
    }
}

fn write_temp_wav(samples: &[f32]) -> Result<PathBuf> {
    let mut path = env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_nanos();
    path.push(format!("stt-whisper-{}-{unique}.wav", std::process::id()));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(&path, spec)
        .with_context(|| format!("failed to create temporary wav {}", path.display()))?;
    for &sample in samples {
        writer
            .write_sample(sample.clamp(-1.0, 1.0))
            .context("failed to write temporary wav sample")?;
    }
    writer.finalize().context("failed to finalize temporary wav")?;

    Ok(path)
}

fn find_server_script() -> Result<PathBuf> {
    if let Ok(path) = env::var("STT_WHISPER_SERVER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }

    for candidate in candidate_script_paths() {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    bail!("unable to find scripts/whisper_server.py")
}

fn candidate_script_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("scripts").join("whisper_server.py"));
    }

    if let Ok(executable) = env::current_exe() {
        let mut cursor = executable.parent().map(Path::to_path_buf);
        while let Some(dir) = cursor {
            candidates.push(dir.join("scripts").join("whisper_server.py"));
            cursor = dir.parent().map(Path::to_path_buf);
        }
    }

    dedupe_paths(candidates)
}

fn find_python(script_path: &Path) -> String {
    if let Ok(path) = env::var("PYTHON") {
        return path;
    }

    // Look for .venv/bin/python relative to the script's parent directories
    let mut cursor = script_path.parent();
    while let Some(dir) = cursor {
        let venv_python = dir.join(".venv").join("bin").join("python");
        if venv_python.is_file() {
            return venv_python.to_string_lossy().into_owned();
        }
        cursor = dir.parent();
    }

    "python3".to_owned()
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}
