"""Fine-tunes a YOLO detector on a dataset in the YOLO directory layout.

    python main.py --data /workspace/data/frames --epochs 100 --batch-size 8
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from modules import (
    DetectionDataModule,
    DetectionDataset,
    YOLOFineTuner,
    read_class_names,
    split_flat_image_paths,
)

BASE_DIR = Path(__file__).parent


class MetricTargetStopping(Callback):
    """Stop after all requested validation targets hold for several epochs."""

    def __init__(
        self,
        group: str,
        precision: float,
        recall: float,
        f1: float,
        map50: float,
        consecutive_epochs: int,
    ) -> None:
        self.keys = {
            "precision": f"val_precision/{group}",
            "recall": f"val_recall/{group}",
            "f1": f"val_f1/{group}",
            "mAP50": f"val_map50/{group}",
        }
        self.targets = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
        }
        self.consecutive_epochs = consecutive_epochs
        self.streak = 0

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        values = trainer.callback_metrics
        if any(key not in values for key in self.keys.values()):
            return
        reached = all(
            float(values[self.keys[name]]) >= target
            for name, target in self.targets.items()
        )
        self.streak = self.streak + 1 if reached else 0
        if self.streak >= self.consecutive_epochs:
            print(
                f"Metric targets reached for {self.streak} consecutive epochs; "
                "stopping training."
            )
            trainer.should_stop = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("/workspace/data/frames"),
                        help="dataset root holding images/, labels/, and classes.txt")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="validation fraction when --data uses the flat images/labels layout")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="enable mixed precision; disabled by default to avoid NaN YOLO26 losses")

    parser.add_argument("--context-class", default="person",
                        help="validation is reported separately with and without this class; "
                             "pass an empty string to switch that off")
    parser.add_argument("--no-balance", action="store_true",
                        help="stop evening out the two context groups during training")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")

    parser.add_argument("--aim", action=argparse.BooleanOptionalAction, default=True,
                        help="track metrics in Aim (default: enabled; use --no-aim to disable)")
    parser.add_argument("--aim-repo", type=str, default=None,
                        help="Aim repository path or aim:// URL; default uses the current directory")
    parser.add_argument("--aim-experiment", default="yolo26m-abandoned-objects")
    parser.add_argument("--aim-run-name", default=None)

    parser.add_argument("--patience", type=int, default=30,
                        help="stop after this many epochs without mAP50 improvement")
    parser.add_argument("--metric-confidence", type=float, default=0.25,
                        help="confidence threshold used for precision, recall, and F1")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--target-f1", type=float, default=0.90)
    parser.add_argument("--target-map50", type=float, default=0.90)
    parser.add_argument("--target-consecutive-epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    class_names = read_class_names(args.data)
    context_class = args.context_class or None

    if (args.data / "images").is_dir():
        train_paths, val_paths = split_flat_image_paths(
            args.data, val_fraction=args.val_fraction, seed=args.seed
        )
        train_dataset = DetectionDataset(
            args.data, None, image_size=args.imgsz, augment=not args.no_augment,
            keep_empty=True, image_paths=train_paths,
        )
        val_dataset = DetectionDataset(
            args.data, None, image_size=args.imgsz, augment=False,
            keep_empty=True, image_paths=val_paths,
        )
        print(f"layout  : flat images/ + labels/ (seed={args.seed})")
    else:
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
        validation_groups=("without_context", "with_context") if context_class else ("all",),
        metric_confidence=args.metric_confidence,
    )

    target_group = "without_context" if context_class else "all"
    monitor = f"val_map50/{target_group}"
    csv_logger = CSVLogger(args.artifacts, name="metrics")
    loggers = [csv_logger]
    if args.aim:
        try:
            from aim.pytorch_lightning import AimLogger
        except ImportError as error:
            raise SystemExit("Aim tracking is enabled but aim is not installed: pip install aim") from error
        aim_options = {
            "experiment": args.aim_experiment,
            "flush_frequency": 1,
        }
        if args.aim_run_name:
            aim_options["run_name"] = args.aim_run_name
        if args.aim_repo:
            aim_options["repo"] = args.aim_repo
        aim_logger = AimLogger(**aim_options)
        loggers.append(aim_logger)
        print(f"aim run : {aim_logger.experiment.hash}")
        print(f"aim repo: {aim_logger.experiment.repo.path}")

    callbacks = [
        ModelCheckpoint(
            dirpath=args.artifacts / "checkpoints",
            filename="{epoch:02d}",
            monitor=monitor,
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor=monitor, mode="max", patience=args.patience),
        MetricTargetStopping(
            group=target_group,
            precision=args.target_precision,
            recall=args.target_recall,
            f1=args.target_f1,
            map50=args.target_map50,
            consecutive_epochs=args.target_consecutive_epochs,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="16-mixed" if args.amp else "32-true",
        gradient_clip_val=10.0,
        accumulate_grad_batches=args.accumulate,
        default_root_dir=args.artifacts,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
