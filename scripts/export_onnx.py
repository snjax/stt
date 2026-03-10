#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "scripts" / "requirements.txt"
MODEL_ID = "ai-sage/GigaAM-v3"
MODEL_REVISION = "e2e_rnnt"


def install_dependencies(skip_install: bool) -> None:
    if skip_install:
        return

    if REQUIREMENTS_PATH.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.8.0",
            "torchaudio==2.8.0",
            "transformers==4.57.1",
            "sentencepiece",
            "onnx",
            "onnxruntime",
            "hydra-core",
            "omegaconf",
            "numba",
            "pyannote.audio==4.0.0",
            "torchcodec==0.7.0",
        ]

    subprocess.check_call(cmd)


def iter_shape_dims(value_info: "onnx.ValueInfoProto") -> list[str]:
    dims: list[str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return dims


def print_onnx_signature(model_path: Path, onnx_module: "onnx") -> None:
    model = onnx_module.load(str(model_path))
    print(f"\n== {model_path.name} ==")
    for prefix, items in (("input", model.graph.input), ("output", model.graph.output)):
        for item in items:
            shape = iter_shape_dims(item)
            print(f"{prefix}: {item.name} shape={shape}")


def copy_tokenizer_model(output_dir: Path, cached_file_fn) -> Path:
    tokenizer_path = Path(
        cached_file_fn(
            MODEL_ID,
            "tokenizer.model",
            revision=MODEL_REVISION,
        )
    )
    target = output_dir / "tokenizer.model"
    shutil.copy2(tokenizer_path, target)
    return target


def save_config_yaml(output_dir: Path, model, omegaconf_module) -> Path:
    cfg = getattr(model.model, "cfg", None)
    if cfg is None:
        cfg = omegaconf_module.OmegaConf.create(model.config.to_dict())

    config_path = output_dir / "config.yaml"
    omegaconf_module.OmegaConf.save(cfg, config_path, resolve=True)
    return config_path


def find_onnx_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.onnx"))


def run_verification(output_dir: Path) -> None:
    download_script = REPO_ROOT / "scripts" / "download_test_audio.py"
    test_script = REPO_ROOT / "scripts" / "test_inference.py"
    test_data_dir = REPO_ROOT / "test_data"

    expected_audio = [test_data_dir / "test_ru.wav", test_data_dir / "test_en.wav"]
    if any(not path.exists() for path in expected_audio):
        subprocess.check_call([sys.executable, str(download_script)])

    subprocess.check_call([sys.executable, str(test_script), "--onnx-dir", str(output_dir)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ai-sage/GigaAM-v3 e2e_rnnt model to ONNX."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "onnx",
        help="Directory for exported ONNX files.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip dependency installation.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Export only; do not run the post-export inference check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_dependencies(skip_install=args.skip_install)

    import onnx  # pylint: disable=import-outside-toplevel
    import omegaconf  # pylint: disable=import-outside-toplevel
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel
    from transformers.utils import cached_file  # pylint: disable=import-outside-toplevel

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}@{MODEL_REVISION}")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Exporting ONNX files to {output_dir}")
    model.model.to_onnx(str(output_dir))

    tokenizer_path = copy_tokenizer_model(output_dir, cached_file)
    config_path = save_config_yaml(output_dir, model, omegaconf)

    print(f"Copied tokenizer model to {tokenizer_path}")
    print(f"Saved config YAML to {config_path}")

    onnx_files = find_onnx_files(output_dir)
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files were exported to {output_dir}")

    for onnx_file in onnx_files:
        print_onnx_signature(onnx_file, onnx)

    if args.skip_test:
        return

    print("\nRunning sample inference verification")
    run_verification(output_dir)


if __name__ == "__main__":
    main()
