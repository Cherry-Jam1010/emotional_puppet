import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import torch

from infer_seed_vii_emotion import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    find_feature_path,
    load_manifest,
    load_model,
    predict_feature,
)


def build_handler(
    checkpoint_path: Path,
    manifest_rows: List[Dict[str, str]],
    stride: int,
):
    device = torch.device("cpu")
    checkpoint, model = load_model(checkpoint_path, device)
    window_len = int(checkpoint["window_len"])

    class EmotionHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "checkpoint": str(checkpoint_path.resolve()),
                        "window_len": window_len,
                        "stride": stride,
                    },
                )
                return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/predict":
                self._send_json(404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8"))

                feature_path = payload.get("feature_path")
                subject_id = payload.get("subject_id")
                trial_id = payload.get("trial_id")

                if feature_path:
                    feature_path = Path(feature_path)
                elif subject_id is not None and trial_id is not None:
                    feature_path = find_feature_path(
                        manifest_rows,
                        int(subject_id),
                        int(trial_id),
                    )
                else:
                    raise ValueError("Provide feature_path or subject_id + trial_id")

                feature = np.load(feature_path).astype(np.float32)
                result = predict_feature(model, feature, window_len=window_len, stride=stride, device=device)
                response = {
                    "feature_path": str(Path(feature_path).resolve()),
                    "window_len": window_len,
                    "stride": stride,
                    **result,
                }
                self._send_json(200, response)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    return EmotionHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local HTTP API for SEED-VII emotion inference.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest_rows = load_manifest(args.manifest)
    handler = build_handler(args.checkpoint, manifest_rows, args.stride)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving on http://{args.host}:{args.port}")
    print("GET  /health")
    print("POST /predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
