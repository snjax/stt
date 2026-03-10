#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


MODEL_ID = "ai-sage/GigaAM-v3"
MODEL_REVISION = "e2e_rnnt"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_DIR = REPO_ROOT / "models" / "onnx"
DEFAULT_AUDIO_FILES = [REPO_ROOT / "test_data" / "test_ru.wav", REPO_ROOT / "test_data" / "test_en.wav"]
DEFAULT_REFERENCES = {
    "test_ru.wav": {
        "reference": "Я люблю тебя.",
        "language": "ru",
        "source_url": "https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/russian/russian-i-love-you.wav",
    },
    "test_en.wav": {
        "reference": "THE ICE IS THICK ENOUGH TO WALK ON.",
        "language": "en",
        "source_url": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav",
    },
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_references() -> dict[str, dict[str, str]]:
    references_path = REPO_ROOT / "test_data" / "references.json"
    if references_path.exists():
        return json.loads(references_path.read_text(encoding="utf-8"))
    return DEFAULT_REFERENCES


def load_audio_raw(path: Path, torchaudio_module, functional_module, torch_module):
    """Load audio as raw waveform (1, samples) for PyTorch model."""
    waveform, sample_rate = torchaudio_module.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = functional_module.resample(waveform, sample_rate, 16000)
    waveform = waveform.to(dtype=torch_module.float32).clamp_(-1.0, 1.0)
    return waveform  # (1, samples)


def compute_mel_spectrogram(waveform, torch_module, torchaudio_module):
    """Compute log-mel spectrogram features for the ONNX encoder.

    Parameters from GigaAM-v3 FeatureExtractor:
    - sample_rate: 16000
    - n_mels: 64
    - n_fft: 400 (sample_rate // 40)
    - win_length: 400 (sample_rate // 40)
    - hop_length: 160 (sample_rate // 100)
    - center: True
    - Scaling: log(clamp(x, 1e-9, 1e9))
    """
    mel_transform = torchaudio_module.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=400,
        win_length=400,
        hop_length=160,
        center=True,
    )
    mel = mel_transform(waveform)  # (1, 64, time)
    mel = torch_module.log(mel.clamp(1e-9, 1e9))
    return mel


def prepare_onnx_input(waveform, torch_module, torchaudio_module):
    """Prepare mel spectrogram input for ONNX encoder."""
    mel = compute_mel_spectrogram(waveform, torch_module, torchaudio_module)
    mel_signal = mel.unsqueeze(0)  # (1, 1, 64, time) -> no, mel is already (1, 64, time)
    # encoder expects (batch, 64, seq_len) and length (batch,)
    mel_len = torch_module.tensor([mel.shape[-1]], dtype=torch_module.int64)
    return mel.numpy(), mel_len.numpy()


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def infer_hidden_sizes(config, decoder_session) -> tuple[int, int]:
    decoder_cfg = getattr(getattr(config, "head", None), "decoder", None)
    if decoder_cfg is not None and hasattr(decoder_cfg, "pred_hidden") and hasattr(decoder_cfg, "pred_rnn_layers"):
        return int(decoder_cfg.pred_rnn_layers), int(decoder_cfg.pred_hidden)

    h_shape = decoder_session.get_inputs()[1].shape
    if len(h_shape) != 3 or not all(isinstance(dim, int) for dim in h_shape):
        raise ValueError("Unable to infer decoder hidden state shape from config or ONNX metadata.")
    return int(h_shape[0]), int(h_shape[2])


class OnnxRNNTGreedyDecoder:
    def __init__(self, onnx_dir: Path, ort_module, spm_module, omegaconf_module):
        self.onnx_dir = onnx_dir
        self.encoder_path = find_single_file(onnx_dir, "*encoder.onnx")
        self.decoder_path = find_single_file(onnx_dir, "*decoder.onnx")
        self.joint_path = find_single_file(onnx_dir, "*joint.onnx")
        self.tokenizer_path = onnx_dir / "tokenizer.model"
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.tokenizer_path}")

        yaml_files = sorted(onnx_dir.glob("*.yaml"))
        config_path = next((f for f in yaml_files if f.name != "config.yaml"), yaml_files[0])
        self.config = omegaconf_module.OmegaConf.load(config_path)
        self.tokenizer = spm_module.SentencePieceProcessor(model_file=str(self.tokenizer_path))
        self.blank_id = len(self.tokenizer)

        self.encoder = ort_module.InferenceSession(
            str(self.encoder_path),
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort_module.InferenceSession(
            str(self.decoder_path),
            providers=["CPUExecutionProvider"],
        )
        self.joint = ort_module.InferenceSession(
            str(self.joint_path),
            providers=["CPUExecutionProvider"],
        )

        self.encoder_input_names = [item.name for item in self.encoder.get_inputs()]
        self.decoder_input_names = [item.name for item in self.decoder.get_inputs()]
        self.joint_input_names = [item.name for item in self.joint.get_inputs()]
        self.num_layers, self.hidden_size = infer_hidden_sizes(self.config, self.decoder)

    def run_encoder(self, audio_signal: np.ndarray, audio_length: np.ndarray) -> tuple[np.ndarray, int]:
        outputs = self.encoder.run(
            None,
            {
                self.encoder_input_names[0]: audio_signal.astype(np.float32),
                self.encoder_input_names[1]: audio_length.astype(np.int64),
            },
        )
        encoded = outputs[0]
        encoded_len = int(np.asarray(outputs[1]).reshape(-1)[0])
        return encoded, encoded_len

    def select_frame(self, encoded: np.ndarray, encoded_len: int, t: int) -> np.ndarray:
        if encoded.ndim != 3:
            raise ValueError(f"Expected encoder output rank 3, got shape {encoded.shape}")

        if encoded.shape[2] == encoded_len:
            frame = encoded[0, :, t]
        elif encoded.shape[1] == encoded_len:
            frame = encoded[0, t, :]
        elif encoded.shape[2] < encoded.shape[1]:
            frame = encoded[0, :, t]
        elif encoded.shape[1] < encoded.shape[2]:
            frame = encoded[0, t, :]
        else:
            raise ValueError(f"Unable to locate time dimension in encoder output shape {encoded.shape}")

        return frame.reshape(1, -1, 1).astype(np.float32)

    def decode(self, audio_signal: np.ndarray, audio_length: np.ndarray, max_symbols_per_step: int = 10) -> tuple[list[int], str]:
        encoded, encoded_len = self.run_encoder(audio_signal, audio_length)

        h = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        c = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        label = np.full((1, 1), self.blank_id, dtype=np.int64)
        emitted: list[int] = []

        for t in range(encoded_len):
            frame = self.select_frame(encoded, encoded_len, t)
            symbols = 0

            while symbols < max_symbols_per_step:
                dec, h_new, c_new = self.decoder.run(
                    None,
                    {
                        self.decoder_input_names[0]: label,
                        self.decoder_input_names[1]: h,
                        self.decoder_input_names[2]: c,
                    },
                )
                # Transpose dec from (1, 1, 320) to (1, 320, 1) for joint network
                dec_transposed = np.transpose(dec, (0, 2, 1)).astype(np.float32)
                joint = self.joint.run(
                    None,
                    {
                        self.joint_input_names[0]: frame,
                        self.joint_input_names[1]: dec_transposed,
                    },
                )[0]
                token_id = int(np.argmax(joint[0, 0, 0, :]))
                if token_id == self.blank_id:
                    break

                emitted.append(token_id)
                label = np.array([[token_id]], dtype=np.int64)
                h = h_new.astype(np.float32)
                c = c_new.astype(np.float32)
                symbols += 1

        return emitted, self.tokenizer.decode(emitted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX GigaAM-v3 inference.")
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help="Directory containing encoder/decoder/joint ONNX files and tokenizer.model.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        action="append",
        help="Optional audio file(s) to test. Defaults to test_data/test_ru.wav and test_data/test_en.wav.",
    )
    parser.add_argument(
        "--max-symbols-per-step",
        type=int,
        default=10,
        help="Maximum number of symbols emitted per encoder frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    import omegaconf  # pylint: disable=import-outside-toplevel
    import sentencepiece as spm  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import torchaudio  # pylint: disable=import-outside-toplevel
    import torchaudio.functional as F  # pylint: disable=import-outside-toplevel
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel

    audio_files = args.audio or DEFAULT_AUDIO_FILES
    missing = [path for path in audio_files if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing audio file(s): {missing_str}. Run scripts/download_test_audio.py first."
        )

    print(f"Loading PyTorch model {MODEL_ID}@{MODEL_REVISION}")
    torch_model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    torch_model.eval()

    onnx_decoder = OnnxRNNTGreedyDecoder(args.onnx_dir.resolve(), ort, spm, omegaconf)
    references = load_references()

    for audio_path in audio_files:
        audio_path = audio_path.resolve()
        waveform = load_audio_raw(audio_path, torchaudio, F, torch)
        with torch.inference_mode():
            pytorch_text = torch_model.transcribe(str(audio_path))
        mel_np, mel_len_np = prepare_onnx_input(waveform, torch, torchaudio)
        token_ids, onnx_text = onnx_decoder.decode(
            mel_np,
            mel_len_np,
            max_symbols_per_step=args.max_symbols_per_step,
        )

        reference = references.get(audio_path.name, {})
        reference_text = reference.get("reference", "<missing reference>")
        language = reference.get("language", "unknown")
        source_url = reference.get("source_url", "<unknown source>")

        print(f"\nAudio: {audio_path}")
        print(f"Language: {language}")
        print(f"Source: {source_url}")
        print(f"Reference: {reference_text}")
        print(f"PyTorch:   {pytorch_text}")
        print(f"ONNX:      {onnx_text}")
        print(f"ONNX token ids: {token_ids}")

        exact_match = pytorch_text == onnx_text
        normalized_match = normalize_text(pytorch_text) == normalize_text(onnx_text)
        print(f"Exact match: {exact_match}")
        print(f"Normalized match: {normalized_match}")


if __name__ == "__main__":
    main()
