import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from seed_vii_models import ConvFeatureClassifier


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "artifacts" / "seed_vii" / "feature_window_run2" / "best_model.pt"
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "seed_vii" / "feature_cache" / "de_LDS" / "manifest.csv"

ID_TO_EMOTION = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_feature_path(
    manifest_rows: List[Dict[str, str]],
    subject_id: int,
    trial_id: int,
) -> Path:
    for row in manifest_rows:
        if int(row["subject_id"]) == subject_id and int(row["trial_id"]) == trial_id:
            return Path(row["feature_path"])
    raise FileNotFoundError(f"Could not find subject_id={subject_id}, trial_id={trial_id} in manifest")


def load_model(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ConvFeatureClassifier(
        input_dim=int(checkpoint["input_dim"]),
        n_classes=7,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def build_windows(feature: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    seq_len = feature.shape[0]
    if seq_len < window_len:
        raise ValueError(f"Feature length {seq_len} is smaller than window_len {window_len}")

    starts = list(range(0, seq_len - window_len + 1, stride))
    windows = []
    for start in starts:
        window = feature[start:start + window_len]
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        window = (window - mean) / np.maximum(std, 1e-6)
        windows.append(window.transpose(1, 0))
    return np.stack(windows, axis=0).astype(np.float32)


def predict_feature(
    model: torch.nn.Module,
    feature: np.ndarray,
    window_len: int,
    stride: int,
    device: torch.device,
) -> Dict[str, object]:
    windows = build_windows(feature, window_len, stride)
    tensor = torch.from_numpy(windows).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probs = probs.mean(axis=0)
    pred_id = int(np.argmax(mean_probs))
    sorted_pairs = sorted(
        (
            {"emotion": ID_TO_EMOTION[idx], "probability": float(prob)}
            for idx, prob in enumerate(mean_probs)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "predicted_label_id": pred_id,
        "predicted_emotion": ID_TO_EMOTION[pred_id],
        "window_count": int(windows.shape[0]),
        "probabilities": sorted_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer emotion from cached SEED-VII feature trials.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.feature_path is None and (args.subject_id is None or args.trial_id is None):
        raise ValueError("Provide either --feature-path or both --subject-id and --trial-id")

    feature_path: Optional[Path] = args.feature_path
    if feature_path is None:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        manifest_rows = load_manifest(args.manifest)
        feature_path = find_feature_path(manifest_rows, args.subject_id, args.trial_id)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    device = torch.device("cpu")
    checkpoint, model = load_model(args.checkpoint, device)
    window_len = int(checkpoint["window_len"])
    feature = np.load(feature_path).astype(np.float32)
    result = predict_feature(model, feature, window_len=window_len, stride=args.stride, device=device)

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "feature_path": str(feature_path.resolve()),
        "window_len": window_len,
        "stride": args.stride,
        **result,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
