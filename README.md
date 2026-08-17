# YOLO transfer learning

Fine-tunes a pretrained YOLO detector onto a small class list, and reports
validation separately for the case the project actually cares about: an object
with nobody standing next to it.

The training code knows one data format and nothing else. Whatever the source —
COCO, Open Images, frames you annotated yourself — you convert it once into the
layout below, and from then on the trainer just reads a directory.

## Data layout

`--data` points at a dataset root shaped like this:

```
data/
├── classes.txt          one class name per line; the line number is the class id
├── train/
│   ├── images/          any of .jpg .jpeg .png .bmp
│   └── labels/          one .txt per image, matched by filename stem
└── val/
    ├── images/
    └── labels/
```

A label file holds one box per line:

```
<class> <cx> <cy> <w> <h>
```

- `class` is the zero-based line number from `classes.txt`
- `cx cy` is the box centre, `w h` its size
- all four are fractions of the image, between 0 and 1

So `1 0.512 0.433 0.180 0.267` is a box of class 1, centred slightly left of
middle, about a fifth of the image wide.

An image whose label file is missing or empty counts as background. Those are
skipped by default and kept only when the dataset is built with `keep_empty`.

This is the layout every annotation tool exports, so hand-labelled frames drop in
without any conversion.

## Preparing public datasets

`prepare_data.py` converts COCO and Open Images into that layout. Class mapping
lives at the top of the file as two plain dictionaries — edit them to change what
gets merged into what.

```bash
python prepare_data.py --split train --output data \
    --coco-images ../dataset/train2017 \
    --coco-annotations ../dataset/annotations/instances_train2017.json \
    --open-images-roots ~/fiftyone/open-images-v7/train \
    --require bag
```

`--require bag` keeps only images holding at least one bag. Without it every
labelled image is kept, which would flood the set with photographs whose only
label is a person.

Filenames are prefixed by source (`coco_…`, `oi_…`) so two datasets can never
collide. Images are copied; `--symlink` links them instead, which saves the disk
space but needs developer mode on Windows.

Run it once per split.

## Training

```bash
python main.py --data data --epochs 30 --batch-size 8
python main.py --data data --resume artifacts/checkpoints/epoch=12.ckpt
```

### Flat cloud dataset with Aim

The command-line trainer also accepts the cloud dataset directly:

```text
/workspace/data/frames/
├── classes.txt
├── images/
└── labels/
```

It creates a deterministic 80/20 split in memory, so it does not move, copy, or
generate files in the dataset directory. Run training from the project root:

```bash
python main.py \
    --data /workspace/data/frames \
    --epochs 100 \
    --batch-size 8 \
    --workers 8
```

Aim is enabled by default. The trainer logs losses, learning rates, precision,
recall, F1, mAP50, and mAP50-95 once per epoch. To point at a specific Aim
repository or remote tracking server:

```bash
python main.py --data /workspace/data/frames --aim-repo /workspace/aim
python main.py --data /workspace/data/frames --aim-repo aim://aim-server:53800
```

Use `--no-aim` to train without Aim. By default, training stops after 30 epochs
without mAP50 improvement, or when precision, recall, F1, and mAP50 are all at
least `0.90` for three consecutive epochs. Override those values with
`--patience`, `--target-precision`, `--target-recall`, `--target-f1`,
`--target-map50`, and `--target-consecutive-epochs`.

For cloud training from a flat `data/frames/images` + `data/frames/labels`
dataset, open `train_yolo26m_aim.ipynb`. It fine-tunes the pretrained YOLO26m
checkpoint and logs validation precision, recall, F1, and mAP50 to Aim.

The number of output classes comes from `classes.txt`, so changing the class list
means editing one file, not chasing a constant through the code.

Useful switches: `--workers`, `--imgsz`, `--lr`, `--accumulate` (sum gradients
over several batches when a batch will not fit), `--no-augment`.

## Validation split by context

Validation runs as two loaders, split on whether `--context-class` (default
`person`) appears in the image. Metrics are reported for each:

```
val_map/without_context     the object standing alone
val_map/with_context        the object next to its owner
```

The gap between them is the number worth watching. Detectors trained on ordinary
photographs learn that a bag is a thing near a person, because in those datasets
it always is — and then lose the bag the moment its owner walks away. One
combined figure hides exactly that failure.

Training uses a weighted sampler so both groups reach the model equally often,
regardless of how lopsided the dataset is. `--no-balance` turns that off.

Checkpoints are selected on `val_map/without_context`, not on loss.

## Layout

```
yolo_transfer_learning/
├── main.py              training entry point
├── prepare_data.py      COCO / Open Images -> the layout above
├── modules/
│   ├── dataset.py       reads the layout, letterbox, augmentation
│   ├── datamodule.py    sampler, the two validation loaders, collation
│   └── model.py         the Lightning module around a YOLO detection head
├── data/                converted dataset
└── artifacts/           checkpoints and metrics
```

## Links

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [COCO](https://cocodataset.org)
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
