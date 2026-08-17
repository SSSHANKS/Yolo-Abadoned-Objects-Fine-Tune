"""Reads a detection dataset laid out in the standard YOLO directory format."""

import random
from pathlib import Path
from typing import Sequence

import cv2

if not hasattr(cv2, "CV_8U"):
    raise ImportError(
        "The installed cv2 package is incomplete or conflicts with another OpenCV wheel. "
        "Uninstall every opencv-* wheel, then install only opencv-python-headless."
    )

import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from torch.utils.data import Dataset

# each worker process would otherwise start its own pool of 16 OpenCV threads
cv2.setNumThreads(0)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def read_class_names(dataset_root: Path) -> tuple[str, ...]:
    """Class ids are line numbers in classes.txt, counting from zero."""
    classes_file = Path(dataset_root) / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"{classes_file} is empty")
    return tuple(names)


def split_flat_image_paths(
    dataset_root: Path, val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    """Split a flat images/ + labels/ dataset without moving or copying files."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    images_dir = Path(dataset_root) / "images"
    labels_dir = Path(dataset_root) / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing {labels_dir}")

    paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {images_dir}; found {len(paths)}")

    random.Random(seed).shuffle(paths)
    val_size = max(1, min(len(paths) - 1, round(len(paths) * val_fraction)))
    return sorted(paths[val_size:]), sorted(paths[:val_size])


def build_augmentation() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-13, 13),
                fill=114,          # same grey as the letterbox padding
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=30, p=0.8
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ToGray(p=0.15),      # night-vision footage carries almost no colour
            A.SaltAndPepper(amount=(0.0, 0.001), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["labels"], min_visibility=0.3
        ),
    )


class DetectionDataset(Dataset):
    """One split of a dataset stored as images/ + labels/ + classes.txt.

    Every label file holds one box per line, "class cx cy w h", with the
    coordinates as fractions of the image. A file with no lines marks a
    background image and is dropped unless keep_empty says otherwise.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str | None,
        image_size: int = 640,
        augment: bool = False,
        keep_empty: bool = False,
        image_paths: Sequence[Path] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split or "selected"
        self.img_size = image_size
        self.augment = augment

        self.class_names = read_class_names(self.dataset_root)

        if image_paths is None:
            if split is None:
                raise ValueError("split is required when image_paths is not supplied")
            images_dir = self.dataset_root / split / "images"
            labels_dir = self.dataset_root / split / "labels"
            if not images_dir.is_dir():
                raise FileNotFoundError(f"Missing {images_dir}")
            selected_paths = sorted(images_dir.iterdir())
        else:
            images_dir = self.dataset_root / "images"
            labels_dir = self.dataset_root / "labels"
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Missing {labels_dir}")
            selected_paths = sorted(Path(path) for path in image_paths)

        self.image_paths: list[Path] = []
        self.labels: list[np.ndarray] = []
        for image_path in selected_paths:
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            boxes = self._read_label(labels_dir / f"{image_path.stem}.txt")
            if len(boxes) == 0 and not keep_empty:
                continue
            self.image_paths.append(image_path)
            self.labels.append(boxes)

        if not self.image_paths:
            raise ValueError(f"No usable samples selected from {images_dir}")

        highest = max((boxes[:, 0].max() for boxes in self.labels if len(boxes)), default=-1)
        if highest >= len(self.class_names):
            raise ValueError(
                f"A label file references class {int(highest)}, but classes.txt "
                f"lists only {len(self.class_names)} names"
            )

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.augmentation = build_augmentation()

    @staticmethod
    def _read_label(label_path: Path) -> np.ndarray:
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        rows = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number} expected 5 values, got {len(parts)}"
                )
            rows.append([float(value) for value in parts])

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def has_class(self, name: str) -> list[bool]:
        """One flag per sample: does it hold at least one box of this class?"""
        if name not in self.class_names:
            raise ValueError(f"Unknown class {name!r}; classes.txt lists {self.class_names}")
        class_id = self.class_names.index(name)
        return [bool((boxes[:, 0] == class_id).any()) for boxes in self.labels]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image, new_width, new_height, pad_x, pad_y = self._get_image(idx)
        bboxes, labels = self._get_annotations(idx, new_width, new_height, pad_x, pad_y)

        if self.augment:
            augmented = self.augmentation(image=image, bboxes=bboxes, labels=labels)
            image = augmented["image"]
            # an empty result would otherwise collapse to shape (0,) and break collation
            bboxes = np.asarray(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(augmented["labels"], dtype=np.int64)

        return self.transform(image), (
            torch.from_numpy(bboxes),
            torch.from_numpy(labels),
        )

    def _get_image(self, idx: int):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._letterbox(image)

    def _get_annotations(
        self, idx: int, new_width: int, new_height: int, pad_x: int, pad_y: int
    ):
        stored = self.labels[idx]
        bboxes = stored[:, 1:].copy()
        bboxes[:, [0, 2]] *= new_width
        bboxes[:, [1, 3]] *= new_height
        bboxes[:, 0] += pad_x
        bboxes[:, 1] += pad_y
        bboxes /= self.img_size
        # a source annotation may reach past the edge of its own image
        np.clip(bboxes, 0.0, 1.0, out=bboxes)
        return bboxes, stored[:, 0].astype(np.int64)

    def _letterbox(self, image: np.ndarray):
        height, width = image.shape[:2]
        ratio = min(self.img_size / height, self.img_size / width)
        new_height, new_width = round(height * ratio), round(width * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        top = (self.img_size - new_height) // 2
        bottom = self.img_size - new_height - top
        left = (self.img_size - new_width) // 2
        right = self.img_size - new_width - left

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return image, new_width, new_height, left, top
