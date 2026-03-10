#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path


SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "test_data"

TEST_SAMPLES = {
    "test_ru.wav": {
        "url": "https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/russian/russian-i-love-you.wav",
        "reference": "Я люблю тебя.",
        "language": "ru",
        "license_note": "Publicly downloadable sample hosted on Hugging Face Spaces.",
    },
    "test_en.wav": {
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav",
        "reference": "THE ICE IS THICK ENOUGH TO WALK ON.",
        "language": "en",
        "license_note": "Publicly downloadable sample hosted on Hugging Face.",
    },
}


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def prepare_waveform(path: Path, torch_module, torchaudio_module, functional_module, audio_bytes: bytes) -> None:
    waveform, sample_rate = torchaudio_module.load(io.BytesIO(audio_bytes), format="wav")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = functional_module.resample(waveform, sample_rate, SAMPLE_RATE)

    waveform = waveform.to(dtype=torch_module.float32).clamp_(-1.0, 1.0)
    torchaudio_module.save(str(path), waveform, SAMPLE_RATE)


def main() -> None:
    import torch  # pylint: disable=import-outside-toplevel
    import torchaudio  # pylint: disable=import-outside-toplevel
    import torchaudio.functional as F  # pylint: disable=import-outside-toplevel

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    references: dict[str, dict[str, str]] = {}

    for filename, meta in TEST_SAMPLES.items():
        target = TEST_DATA_DIR / filename
        print(f"Downloading {filename} from {meta['url']}")
        audio_bytes = download_bytes(meta["url"])
        prepare_waveform(target, torch, torchaudio, F, audio_bytes)
        references[filename] = {
            "reference": meta["reference"],
            "language": meta["language"],
            "source_url": meta["url"],
            "license_note": meta["license_note"],
        }
        print(f"Saved {target}")

    references_path = TEST_DATA_DIR / "references.json"
    references_path.write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {references_path}")


if __name__ == "__main__":
    main()
