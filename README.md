# YOLO26m fine-tuning

The training entry point uses the native Ultralytics trainer. Lightning is not
used. Aim tracking is enabled by default and can be disabled on demand.

## Dataset

`--data` must point to the existing flat YOLO dataset:

```text
/workspace/data/frames/
├── classes.txt
├── images/
└── labels/
```

Each line of `classes.txt` is a class name; its zero-based line number is the
class ID. Each label row is standard YOLO format:

```text
<class_id> <center_x> <center_y> <width> <height>
```

The coordinates are normalized to `[0, 1]`. Missing and empty label files are
treated as background images.

At startup, `main.py` deterministically splits the images into training and
validation sets. It writes only `train.txt`, `val.txt`, and `data.yaml` manifests
under `artifacts/dataset_split`. It does not copy, move, or modify source data.

## Install

Install PyTorch using the CUDA build appropriate for the machine, then:

```bash
pip install -r requirements.txt
```

Only `opencv-python-headless` should be installed. Do not install it alongside
`opencv-python`, `opencv-contrib-python`, or another OpenCV wheel because those
packages all provide the same `cv2` module.

## Train

From the project root in the cloud environment:

```bash
python main.py \
    --data /workspace/data/frames \
    --model yolo26m.pt \
    --epochs 100 \
    --batch-size 8 \
    --workers 8
```

Aim is on by default. The run records per-epoch training and validation losses,
learning rates, precision, recall, F1, mAP50, and mAP50-95. Point it at an
explicit local or remote Aim repository when needed:

```bash
python main.py --data /workspace/data/frames --aim-repo /workspace/aim
python main.py --data /workspace/data/frames --aim-repo aim://aim-server:53800
```

Disable tracking with `--no-aim`.

## Augmentation

Augmentation is enabled by default and preserves the original project pipeline:

- horizontal flip;
- affine scale, translation, and rotation;
- hue, saturation, and value shift;
- random brightness and contrast;
- grayscale conversion;
- salt-and-pepper noise.

The overlapping built-in Ultralytics transformations are set to zero so the
same operation is not applied twice. Use `--no-augment` to disable the custom
pipeline.

## Early stopping and resume

Native Ultralytics early stopping stops after `--patience` epochs without
validation improvement (default: 30). Training also stops when precision,
recall, F1, and mAP50 are all at least `0.90` for three consecutive epochs.
The thresholds are configurable:

```bash
python main.py \
    --data /workspace/data/frames \
    --target-precision 0.92 \
    --target-recall 0.90 \
    --target-f1 0.91 \
    --target-map50 0.93 \
    --target-consecutive-epochs 3
```

Resume from the native Ultralytics checkpoint:

```bash
python main.py \
    --data /workspace/data/frames \
    --resume artifacts/runs/yolo26m/weights/last.pt
```

Use `python main.py --help` for all options.
