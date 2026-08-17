"""Train YOLO26m with native Ultralytics and optional Aim tracking.

Example:
    python main.py --data /workspace/data/frames --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/workspace/data/frames"),
        help="folder containing images/, labels/, and classes.txt",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", type=Path, default=BASE_DIR / "artifacts")

    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="for example 0, 0,1, cpu, or mps")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mixed precision (off by default because it produced NaN losses on this setup)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the project's existing Albumentations pipeline (default: enabled)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Ultralytics last.pt checkpoint")

    parser.add_argument(
        "--aim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="track every epoch in Aim (default: enabled)",
    )
    parser.add_argument("--aim-repo", default=None, help="Aim repo path or aim:// URL")
    parser.add_argument("--aim-experiment", default="yolo26m-abandoned-objects")
    parser.add_argument("--aim-run-name", default=None)

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Ultralytics early-stopping patience for no validation improvement",
    )
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--target-map50", type=float, default=0.90)
    parser.add_argument("--target-consecutive-epochs", type=int, default=3)
    return parser.parse_args()


def read_class_names(root: Path) -> list[str]:
    classes_file = root / "classes.txt"
    if not classes_file.is_file():
        raise FileNotFoundError(f"Missing {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"{classes_file} is empty")
    return names


def validate_labels(images: list[Path], labels_dir: Path, class_count: int) -> None:
    """Fail early on malformed YOLO rows; absent/empty files are backgrounds."""
    for image in images:
        label_file = labels_dir / f"{image.stem}.txt"
        if not label_file.exists():
            continue
        for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_file}:{line_number}: expected 5 values, got {len(parts)}")
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as error:
                raise ValueError(f"{label_file}:{line_number}: invalid YOLO label") from error
            if not 0 <= class_id < class_count:
                raise ValueError(f"{label_file}:{line_number}: class {class_id} is out of range")
            if any(not 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(f"{label_file}:{line_number}: coordinates must be in [0, 1]")


def prepare_dataset(root: Path, artifacts: Path, val_fraction: float, seed: int) -> tuple[Path, int, int]:
    """Create Ultralytics manifests without copying or changing source data."""
    root = root.expanduser().resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} and {labels_dir}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    names = read_class_names(root)
    images = sorted(path.resolve() for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError(f"Need at least two images in {images_dir}; found {len(images)}")
    validate_labels(images, labels_dir, len(names))

    random.Random(seed).shuffle(images)
    val_count = max(1, min(len(images) - 1, round(len(images) * val_fraction)))
    val_images = sorted(images[:val_count])
    train_images = sorted(images[val_count:])

    split_dir = artifacts.expanduser().resolve() / "dataset_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    train_file.write_text("\n".join(path.as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(path.as_posix() for path in val_images) + "\n", encoding="utf-8")

    data_yaml = split_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": train_file.as_posix(),
                "val": val_file.as_posix(),
                "names": {index: name for index, name in enumerate(names)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml, len(train_images), len(val_images)


def build_augmentations() -> list[Any]:
    """The augmentation pipeline used by the former Lightning data loader."""
    try:
        import albumentations as A
    except (ImportError, AttributeError) as error:
        raise SystemExit(
            "Augmentation requires a working albumentations/OpenCV installation. "
            "Install only opencv-python-headless (not multiple opencv-* wheels), "
            "or run with --no-augment."
        ) from error

    return [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-13, 13),
            fill=114,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.8,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ToGray(p=0.15),
        A.SaltAndPepper(amount=(0.0, 0.001), p=0.2),
    ]


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_float(metrics.get(key))
        if value is not None:
            return value
    return None


class AimUltralyticsTracker:
    """Epoch tracking and metric-target stopping for native Ultralytics."""

    def __init__(self, args: argparse.Namespace, parameters: dict[str, Any]) -> None:
        self.run = None
        if args.aim:
            try:
                from aim import Run
            except ImportError as error:
                raise SystemExit("Aim tracking is enabled but aim is not installed: pip install aim") from error

            options: dict[str, Any] = {"experiment": args.aim_experiment}
            if args.aim_repo:
                options["repo"] = args.aim_repo
            self.run = Run(**options)
            if args.aim_run_name:
                self.run.name = args.aim_run_name
            self.run["hparams"] = parameters
            print(f"Aim run : {self.run.hash}")
        self.targets = {
            "precision": args.target_precision,
            "recall": args.target_recall,
            "f1": args.target_f1,
            "mAP50": args.target_map50,
        }
        self.required_streak = args.target_consecutive_epochs
        self.streak = 0

    def track(self, name: str, value: Any, epoch: int, subset: str) -> None:
        number = finite_float(value)
        if self.run is not None and number is not None:
            self.run.track(number, name=name, step=epoch, epoch=epoch, context={"subset": subset})

    def on_train_epoch_end(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        if getattr(trainer, "tloss", None) is not None:
            losses = trainer.label_loss_items(trainer.tloss, prefix="train")
            for name, value in losses.items():
                self.track(name.removeprefix("train/"), value, epoch, "train")
        for name, value in getattr(trainer, "lr", {}).items():
            self.track(name, value, epoch, "train")

    def on_fit_epoch_end(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        precision = metric(metrics, "metrics/precision(B)", "metrics/precision")
        recall = metric(metrics, "metrics/recall(B)", "metrics/recall")
        map50 = metric(metrics, "metrics/mAP50(B)", "metrics/mAP50")
        map5095 = metric(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)

        for name, value in {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
            "mAP50-95": map5095,
        }.items():
            self.track(name, value, epoch, "validation")
        for key, value in metrics.items():
            if key.startswith("val/") and key.endswith("_loss"):
                self.track(key.removeprefix("val/"), value, epoch, "validation")

        required = {"precision": precision, "recall": recall, "f1": f1, "mAP50": map50}
        reached = all(value is not None and value >= self.targets[name] for name, value in required.items())
        self.streak = self.streak + 1 if reached else 0
        if self.run is not None:
            self.run["early_stopping/target_streak"] = self.streak
        if self.required_streak > 0 and self.streak >= self.required_streak:
            print(f"Metric targets reached for {self.streak} consecutive epochs; stopping training.")
            trainer.stop = True

    def close(self) -> None:
        if self.run is not None:
            self.run.close()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Ultralytics is not installed: pip install -r requirements.txt") from error

    data_yaml, train_count, val_count = prepare_dataset(
        args.data, args.artifacts, args.val_fraction, args.seed
    )
    print(f"dataset : {args.data.expanduser().resolve()}")
    print(f"train   : {train_count} images")
    print(f"val     : {val_count} images")
    print(f"manifest: {data_yaml}")
    print(f"augment : {'enabled' if args.augment else 'disabled'}")

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.artifacts.expanduser().resolve() / "runs"),
        "name": "yolo26m",
        "plots": True,
        "val": True,
        # Disable overlapping built-ins: custom Albumentations below preserves
        # the project's original probabilities and ranges exactly.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "augmentations": build_augmentations() if args.augment else None,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True

    model = YOLO(str(args.resume.expanduser()) if args.resume else args.model)
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    parameters["data_yaml"] = str(data_yaml)
    tracker = AimUltralyticsTracker(args, parameters)
    model.add_callback("on_train_epoch_end", tracker.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", tracker.on_fit_epoch_end)

    try:
        model.train(**train_args)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
