#!/usr/bin/env python3

import json
import sys
import traceback
from typing import Optional

try:
    import mlx_whisper
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import path depends on local env
    mlx_whisper = None
    IMPORT_ERROR = exc


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def require_model(current_model: Optional[str]) -> str:
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"mlx_whisper import failed: {IMPORT_ERROR}")
    if not current_model:
        raise RuntimeError("server not initialized")
    return current_model


def main() -> int:
    model = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params") or {}

            if method == "init":
                model = params.get("model")
                if not model:
                    raise ValueError("missing init.params.model")

                require_model(model)
                log(f"Initialized mlx-whisper with model {model}")
                send({"result": {"status": "ready"}})
                continue

            if method == "transcribe":
                current_model = require_model(model)
                audio_path = params.get("audio_path")
                if not audio_path:
                    raise ValueError("missing transcribe.params.audio_path")

                log(f"Transcribing {audio_path}")
                result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=current_model)
                if not isinstance(result, dict):
                    raise RuntimeError("mlx_whisper.transcribe returned an unexpected response")

                send(
                    {
                        "result": {
                            "text": result.get("text", ""),
                            "language": result.get("language"),
                        }
                    }
                )
                continue

            if method == "shutdown":
                log("Shutting down mlx-whisper server")
                send({"result": {"status": "bye"}})
                return 0

            raise ValueError(f"unknown method: {method}")
        except Exception as exc:  # pragma: no cover - runtime error handling
            log(f"Error: {exc}")
            traceback.print_exc(file=sys.stderr)
            send({"error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
