"""Fine-tunes a YOLO detector on a dataset in the YOLO directory layout.

    python main.py --data data --epochs 30 --batch-size 8
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from modules import DetectionDataModule, DetectionDataset, YOLOFineTuner, read_class_names

BASE_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data",
                        help="dataset root holding classes.txt and the split directories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--artifacts", type=Path, default=BASE_DIR / "artifacts")

    parser.add_argument("--model-cfg", default="yolo26m.yaml")
    parser.add_argument("--weights", default="yolo26m.pt",
                        help="pretrained weights the backbone and neck start from")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="sum gradients over this many batches before stepping")

    parser.add_argument("--context-class", default="person",
                        help="validation is reported separately with and without this class; "
                             "pass an empty string to switch that off")
    parser.add_argument("--no-balance", action="store_true",
                        help="stop evening out the two context groups during training")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    class_names = read_class_names(args.data)
    context_class = args.context_class or None

    train_dataset = DetectionDataset(
        args.data, args.train_split, image_size=args.imgsz, augment=not args.no_augment
    )
    val_dataset = DetectionDataset(
        args.data, args.val_split, image_size=args.imgsz, augment=False
    )

    print(f"classes : {list(class_names)}")
    print(f"train   : {len(train_dataset)} images")
    print(f"val     : {len(val_dataset)} images")
    if context_class:
        for name, dataset in (("train", train_dataset), ("val", val_dataset)):
            flags = dataset.has_class(context_class)
            print(f"{name:8s}  without {context_class}: {len(flags) - sum(flags)}")

    datamodule = DetectionDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        context_class=context_class,
        balance_context=not args.no_balance,
    )

    model = YOLOFineTuner(
        model_cfg=args.model_cfg,
        pretrained_weights=args.weights,
        num_classes=len(class_names),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    monitor = "val_map/without_context" if context_class else "val/loss"
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=10.0,
        accumulate_grad_batches=args.accumulate,
        default_root_dir=args.artifacts,
        logger=CSVLogger(args.artifacts, name="metrics"),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.artifacts / "checkpoints",
                filename="{epoch:02d}",
                monitor=monitor,
                mode="max" if monitor.startswith("val_map") else "min",
                save_top_k=3,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
