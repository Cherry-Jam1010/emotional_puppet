import argparse
import csv
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from seed_vii_data import DEFAULT_OUTPUT_DIR, EMOTION_TO_ID


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRIAL_MANIFEST = DEFAULT_OUTPUT_DIR / "trial_manifest.csv"
DEFAULT_RUN_DIR = ROOT_DIR / "artifacts" / "seed_vii" / "training"
DEFAULT_SAMPLING_RATE = 200
DEFAULT_WINDOW_SECONDS = 4.0

ID_TO_EMOTION = {value: key for key, value in EMOTION_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_trial_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_subjects(
    subjects: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    subject_list = sorted(subjects)
    rng.shuffle(subject_list)

    n_subjects = len(subject_list)
    n_train = max(1, int(round(n_subjects * train_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))
    n_test = n_subjects - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Not enough subjects to create train/val/test split")

    train_subjects = sorted(subject_list[:n_train])
    val_subjects = sorted(subject_list[n_train:n_train + n_val])
    test_subjects = sorted(subject_list[n_train + n_val:])
    return train_subjects, val_subjects, test_subjects


def filter_records_by_subject(records: Sequence[Dict[str, str]], subjects: Sequence[int]) -> List[Dict[str, str]]:
    subject_set = set(subjects)
    return [record for record in records if int(record["subject_id"]) in subject_set]


@lru_cache(maxsize=256)
def load_trial_array(file_path: str, mat_key: str) -> np.ndarray:
    mat_data = sio.loadmat(file_path)
    array = np.asarray(mat_data[mat_key], dtype=np.float32)
    return array


def build_window_starts(
    n_samples: int,
    window_size: int,
    windows_per_trial: int,
    evenly_spaced: bool,
    rng: random.Random,
) -> List[int]:
    if n_samples < window_size:
        return []

    max_start = n_samples - window_size
    if max_start == 0:
        return [0]

    if windows_per_trial <= 1:
        return [max_start // 2 if evenly_spaced else rng.randint(0, max_start)]

    if evenly_spaced:
        return sorted({int(round(x)) for x in np.linspace(0, max_start, windows_per_trial)})

    if max_start + 1 <= windows_per_trial:
        return list(range(0, max_start + 1))

    return sorted(rng.sample(range(0, max_start + 1), windows_per_trial))


class SeedVIIWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        window_size: int,
        windows_per_trial: int,
        seed: int,
        evenly_spaced: bool,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize
        self.samples: List[Dict[str, object]] = []
        rng = random.Random(seed)

        for record in records:
            n_samples = int(record["n_samples"])
            starts = build_window_starts(
                n_samples=n_samples,
                window_size=window_size,
                windows_per_trial=windows_per_trial,
                evenly_spaced=evenly_spaced,
                rng=rng,
            )
            for start in starts:
                self.samples.append(
                    {
                        "file_path": record["file_path"],
                        "mat_key": record["mat_key"],
                        "label_id": int(record["label_id"]),
                        "subject_id": int(record["subject_id"]),
                        "trial_id": int(record["trial_id"]),
                        "emotion": record["emotion"],
                        "start": start,
                        "stop": start + window_size,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        signal = load_trial_array(sample["file_path"], sample["mat_key"])
        window = signal[:, sample["start"]:sample["stop"]]

        if self.normalize:
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True)
            window = (window - mean) / np.maximum(std, 1e-6)

        tensor = torch.from_numpy(window).unsqueeze(0)
        label = torch.tensor(sample["label_id"], dtype=torch.long)
        subject_id = torch.tensor(sample["subject_id"], dtype=torch.long)
        trial_id = torch.tensor(sample["trial_id"], dtype=torch.long)
        return tensor, label, subject_id, trial_id


class EEGNetClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout: float = 0.35,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_dim = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def make_class_weights(records: Sequence[Dict[str, str]]) -> torch.Tensor:
    counts = defaultdict(int)
    for record in records:
        counts[int(record["label_id"])] += 1

    total = sum(counts.values())
    weights = []
    for class_id in range(len(EMOTION_TO_ID)):
        class_count = counts[class_id]
        weights.append(total / (len(EMOTION_TO_ID) * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def aggregate_trial_predictions(
    logits_list: List[np.ndarray],
    labels_list: List[int],
    subject_ids: List[int],
    trial_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    grouped_logits: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    grouped_labels: Dict[Tuple[int, int], int] = {}

    for logits, label, subject_id, trial_id in zip(logits_list, labels_list, subject_ids, trial_ids):
        key = (subject_id, trial_id)
        grouped_logits[key].append(logits)
        grouped_labels[key] = label

    y_true = []
    y_pred = []
    for key in sorted(grouped_logits):
        mean_logits = np.mean(grouped_logits[key], axis=0)
        y_true.append(grouped_labels[key])
        y_pred.append(int(np.argmax(mean_logits)))

    return np.asarray(y_true), np.asarray(y_pred)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_list: List[np.ndarray] = []
    labels_list: List[int] = []
    subject_ids: List[int] = []
    trial_ids: List[int] = []

    with torch.no_grad():
        for inputs, labels, batch_subject_ids, batch_trial_ids in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            logits_list.extend(logits.cpu().numpy())
            labels_list.extend(labels.cpu().numpy().tolist())
            subject_ids.extend(batch_subject_ids.numpy().tolist())
            trial_ids.extend(batch_trial_ids.numpy().tolist())

    y_true, y_pred = aggregate_trial_predictions(logits_list, labels_list, subject_ids, trial_ids)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID_TO_EMOTION[i] for i in range(len(ID_TO_EMOTION))],
        output_dict=True,
        zero_division=0,
    )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": report,
        "num_trials": int(len(y_true)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for inputs, labels, _, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline EEG emotion model on SEED-VII.")
    parser.add_argument("--trial-manifest", type=Path, default=DEFAULT_TRIAL_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--train-windows-per-trial", type=int, default=3)
    parser.add_argument("--eval-windows-per-trial", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    if not args.trial_manifest.exists():
        raise FileNotFoundError(f"Trial manifest not found: {args.trial_manifest}")

    set_seed(args.seed)
    device = torch.device("cpu")

    records = read_trial_manifest(args.trial_manifest)
    unique_subjects = sorted({int(record["subject_id"]) for record in records})
    train_subjects, val_subjects, test_subjects = split_subjects(
        unique_subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_records = filter_records_by_subject(records, train_subjects)
    val_records = filter_records_by_subject(records, val_subjects)
    test_records = filter_records_by_subject(records, test_subjects)

    window_size = int(round(args.window_seconds * args.sampling_rate))
    train_dataset = SeedVIIWindowDataset(
        train_records,
        window_size=window_size,
        windows_per_trial=args.train_windows_per_trial,
        seed=args.seed,
        evenly_spaced=False,
    )
    val_dataset = SeedVIIWindowDataset(
        val_records,
        window_size=window_size,
        windows_per_trial=args.eval_windows_per_trial,
        seed=args.seed + 1,
        evenly_spaced=True,
    )
    test_dataset = SeedVIIWindowDataset(
        test_records,
        window_size=window_size,
        windows_per_trial=args.eval_windows_per_trial,
        seed=args.seed + 2,
        evenly_spaced=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = EEGNetClassifier(
        n_channels=62,
        n_samples=window_size,
        n_classes=len(EMOTION_TO_ID),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = make_class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_model.pt"
    history_path = args.run_dir / "history.json"
    results_path = args.run_dir / "results.json"
    split_path = args.run_dir / "subject_split.json"

    save_json(
        split_path,
        {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        },
    )

    history: List[Dict[str, float]] = []
    best_val_f1 = -math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    save_json(history_path, {"history": history})

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    results = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "val_metrics": {
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "num_trials": int(val_metrics["num_trials"]),
            "report": val_metrics["report"],
        },
        "test_metrics": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "num_trials": int(test_metrics["num_trials"]),
            "report": test_metrics["report"],
        },
        "dataset_sizes": {
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
            "train_trials": len(train_records),
            "val_trials": len(val_records),
            "test_trials": len(test_records),
        },
    }
    save_json(results_path, results)

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Results written to: {results_path}")
    print(
        f"Test accuracy={results['test_metrics']['accuracy']:.4f}, "
        f"test macro_f1={results['test_metrics']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
