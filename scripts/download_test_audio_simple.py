#!/usr/bin/env python3
"""Download test audio files using only soundfile + numpy (no torch)."""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "test_data"

TEST_SAMPLES = {
    "test_ru.wav": {
        "url": "https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/russian/russian-i-love-you.wav",
        "reference": "Я люблю тебя.",
        "language": "ru",
    },
    "test_en.wav": {
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav",
        "reference": "THE ICE IS THICK ENOUGH TO WALK ON.",
        "language": "en",
    },
}


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def main() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, meta in TEST_SAMPLES.items():
        target = TEST_DATA_DIR / filename
        print(f"Downloading {filename}...")
        audio_bytes = download_bytes(meta["url"])

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Downmix to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz
        data = resample_linear(data, sr, SAMPLE_RATE)
        data = np.clip(data, -1.0, 1.0)

        sf.write(str(target), data, SAMPLE_RATE)
        print(f"  Saved {target} ({len(data)/SAMPLE_RATE:.1f}s, {sr}->{SAMPLE_RATE}Hz)")

    refs = {k: {"reference": v["reference"], "language": v["language"]} for k, v in TEST_SAMPLES.items()}
    refs_path = TEST_DATA_DIR / "references.json"
    refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {refs_path}")


if __name__ == "__main__":
    main()
