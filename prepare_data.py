"""Converts public detection datasets into the YOLO directory layout.

Run once per split. Everything downstream reads only the converted copy, so the
training code never has to know that COCO stores JSON and Open Images stores CSV.

    python prepare_data.py --split train --output data \
        --coco-images ../dataset/train2017 \
        --coco-annotations ../dataset/annotations/instances_train2017.json \
        --open-images-roots C:/Users/me/fiftyone/open-images-v7/train \
        --require bag
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

# The class list of the converted dataset. Line order here becomes classes.txt,
# and the index becomes the class id in every label file.
TARGET_CLASSES = ("person", "bag")

# source class name -> target class; anything missing here is dropped
COCO_CLASS_MAP = {
    "person": "person",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
}

OPEN_IMAGES_CLASS_MAP = {
    "Person": "person",
    "Man": "person",
    "Woman": "person",
    "Boy": "person",
    "Girl": "person",
    "Backpack": "bag",
    "Handbag": "bag",
    "Suitcase": "bag",
    "Briefcase": "bag",
    "Luggage and bags": "bag",
    "Plastic bag": "bag",
}


def class_id(name: str) -> int:
    return TARGET_CLASSES.index(name)


def read_coco(images_dir: Path, annotations_file: Path) -> dict[str, tuple[Path, list]]:
    """Return stem -> (image path, boxes), boxes as (class, cx, cy, w, h) fractions."""
    with open(annotations_file, encoding="utf-8") as handle:
        raw = json.load(handle)

    target_of = {
        category["id"]: class_id(COCO_CLASS_MAP[category["name"]])
        for category in raw["categories"]
        if category["name"] in COCO_CLASS_MAP
    }
    sizes = {image["id"]: (image["width"], image["height"]) for image in raw["images"]}

    per_image = defaultdict(list)
    for annotation in raw["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        target = target_of.get(annotation["category_id"])
        if target is None:
            continue
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue
        image_width, image_height = sizes[annotation["image_id"]]
        per_image[annotation["image_id"]].append((
            target,
            (x + width / 2) / image_width,
            (y + height / 2) / image_height,
            width / image_width,
            height / image_height,
        ))

    samples = {}
    for image_id, boxes in per_image.items():
        image_path = images_dir / f"{image_id:012d}.jpg"
        if image_path.exists():
            samples[f"coco_{image_id:012d}"] = (image_path, boxes)
    return samples


def read_open_images(root: Path) -> dict[str, tuple[Path, list]]:
    """Open Images ships fractions already, so only the corners need converting."""
    target_of = {}
    with open(root / "metadata" / "classes.csv", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[1] in OPEN_IMAGES_CLASS_MAP:
                target_of[row[0]] = class_id(OPEN_IMAGES_CLASS_MAP[row[1]])

    # the annotation file describes the whole split, only a fraction was downloaded
    paths = {path.stem: path for path in (root / "data").glob("*.jpg")}

    per_image = defaultdict(list)
    with open(root / "labels" / "detections.csv", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        column = {name: index for index, name in enumerate(next(reader))}

        for row in reader:
            image_id = row[column["ImageID"]]
            if image_id not in paths:
                continue
            target = target_of.get(row[column["LabelName"]])
            if target is None:
                continue
            if row[column["IsGroupOf"]] == "1" or row[column["IsDepiction"]] == "1":
                continue

            x_min, x_max = float(row[column["XMin"]]), float(row[column["XMax"]])
            y_min, y_max = float(row[column["YMin"]]), float(row[column["YMax"]])
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0:
                continue

            per_image[image_id].append((
                target, (x_min + x_max) / 2, (y_min + y_max) / 2, width, height
            ))

    return {f"oi_{image_id}": (paths[image_id], boxes) for image_id, boxes in per_image.items()}


def write_split(
    samples: dict[str, tuple[Path, list]],
    output_root: Path,
    split: str,
    required: set[int],
    symlink: bool,
) -> tuple[int, dict[str, int]]:
    images_dir = output_root / split / "images"
    labels_dir = output_root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    per_class: dict[str, int] = defaultdict(int)

    for stem, (source_path, boxes) in sorted(samples.items()):
        if required and not any(box[0] in required for box in boxes):
            continue

        destination = images_dir / f"{stem}{source_path.suffix}"
        if not destination.exists():
            if symlink:
                destination.symlink_to(source_path.resolve())
            else:
                shutil.copy2(source_path, destination)

        lines = [
            f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}" for box in boxes
        ]
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        written += 1
        for box in boxes:
            per_class[TARGET_CLASSES[box[0]]] += 1

    return written, dict(per_class)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="dataset root to write into")
    parser.add_argument("--split", required=True, help="train, val, test - the subdirectory name")
    parser.add_argument("--coco-images", type=Path, default=None)
    parser.add_argument("--coco-annotations", type=Path, default=None)
    parser.add_argument("--open-images-roots", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        help="keep an image only if it holds one of these classes; "
             "without it, any labelled image is kept",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="link the images instead of copying; needs developer mode on Windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.coco_images) != bool(args.coco_annotations):
        raise SystemExit("--coco-images and --coco-annotations go together")
    if not args.coco_images and not args.open_images_roots:
        raise SystemExit("Nothing to convert: give a COCO source, Open Images roots, or both")

    unknown = [name for name in args.require if name not in TARGET_CLASSES]
    if unknown:
        raise SystemExit(f"--require lists classes that are not targets: {unknown}")
    required = {class_id(name) for name in args.require}

    samples: dict[str, tuple[Path, list]] = {}
    if args.coco_images:
        found = read_coco(args.coco_images, args.coco_annotations)
        print(f"coco         : {len(found)} images with a target class")
        samples.update(found)
    for root in args.open_images_roots:
        found = read_open_images(root)
        print(f"open images  : {len(found)} images from {root.name}")
        samples.update(found)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classes.txt").write_text("\n".join(TARGET_CLASSES) + "\n", encoding="utf-8")

    written, per_class = write_split(samples, args.output, args.split, required, args.symlink)

    print(f"\nwritten to {(args.output / args.split).resolve()}")
    print(f"images : {written}")
    print(f"boxes  : {per_class}")


if __name__ == "__main__":
    main()
