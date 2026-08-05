import lightning as L
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from torchmetrics.detection import MeanAveragePrecision

# the two validation loaders: without the context class, then with it
VAL_GROUPS = ("without_context", "with_context")

class YOLOFineTuner(L.LightningModule):
    def __init__(
        self,
        model_cfg: str = "yolo26m.yaml",
        pretrained_weights: str = "yolo26m.pt",
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        pretrained = YOLO(pretrained_weights)

        # the architecture is rebuilt with a 3-class head, then every layer whose
        # shape still matches is copied over; the head itself starts from scratch
        self.model = DetectionModel(cfg=model_cfg, nc=num_classes, verbose=False)
        self.model.args = get_cfg(overrides=pretrained.model.args)
        self.model.load(pretrained.model)

        self.criterion = None
        self.val_map = torch.nn.ModuleList(
            [MeanAveragePrecision(box_format="xyxy") for _ in VAL_GROUPS]
        )

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def _step(self, batch: dict, stage: str):
        if self.criterion is None:
            self.criterion = self.model.init_criterion()

        output = self.model(batch["img"])
        loss, components = self.criterion(output, batch)
        total = loss.sum()

        batch_size = batch["img"].shape[0]
        self.log(f"{stage}/loss", total, prog_bar=True, batch_size=batch_size,
                 add_dataloader_idx=False)
        self.log_dict(
            {f"{stage}/{name}": value for name, value in components.items()},
            batch_size=batch_size, add_dataloader_idx=False,
        )
        return total, output

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")[0]

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        stage = f"val_{VAL_GROUPS[dataloader_idx]}"
        total, output = self._step(batch, stage)
        self._update_map(batch, output[0], dataloader_idx)
        return total

    def _update_map(self, batch: dict, detections: torch.Tensor, group: int) -> None:
        size = batch["img"].shape[-1]
        for i in range(detections.shape[0]):
            # everything below 0.001 is noise; torchmetrics keeps the best 100, as COCO does
            kept = detections[i][detections[i][:, 4] > 0.001]

            selected = batch["batch_idx"] == i
            boxes = batch["bboxes"][selected] * size
            half_w, half_h = boxes[:, 2] / 2, boxes[:, 3] / 2
            target_boxes = torch.stack(
                [boxes[:, 0] - half_w, boxes[:, 1] - half_h,
                 boxes[:, 0] + half_w, boxes[:, 1] + half_h], dim=1
            )

            self.val_map[group].update(
                [{"boxes": kept[:, :4], "scores": kept[:, 4], "labels": kept[:, 5].long()}],
                [{"boxes": target_boxes, "labels": batch["cls"][selected]}],
            )

    def on_validation_epoch_end(self) -> None:
        for name, metric in zip(VAL_GROUPS, self.val_map):
            result = metric.compute()
            self.log(f"val_map/{name}", result["map"], prog_bar=True)
            self.log(f"val_map50/{name}", result["map_50"])
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
