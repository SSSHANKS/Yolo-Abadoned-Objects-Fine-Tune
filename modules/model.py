import lightning as L
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import nms

from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

# the two validation loaders: without the context class, then with it
VAL_GROUPS = ("without_context", "with_context")


class DetectionQuality(Metric):
    """Dataset-level precision, recall and F1 at fixed confidence/IoU thresholds."""

    def __init__(self, confidence: float = 0.25, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, prediction: dict, target: dict) -> None:
        keep = prediction["scores"] >= self.confidence
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        labels = prediction["labels"][keep]
        target_boxes = target["boxes"]
        target_labels = target["labels"]

        if boxes.numel() == 0:
            self.fn += len(target_boxes)
            return
        if target_boxes.numel() == 0:
            self.fp += len(boxes)
            return

        overlaps = box_iou(boxes, target_boxes)
        matched_targets: set[int] = set()
        true_positives = 0
        for prediction_index in scores.argsort(descending=True).tolist():
            candidates = [
                index
                for index in range(len(target_boxes))
                if index not in matched_targets
                and target_labels[index] == labels[prediction_index]
            ]
            if not candidates:
                continue
            candidate_ious = overlaps[prediction_index, candidates]
            best_position = int(candidate_ious.argmax())
            if candidate_ious[best_position] >= self.iou_threshold:
                matched_targets.add(candidates[best_position])
                true_positives += 1

        self.tp += true_positives
        self.fp += len(boxes) - true_positives
        self.fn += len(target_boxes) - true_positives

    def compute(self) -> dict[str, torch.Tensor]:
        precision = self.tp.float() / (self.tp + self.fp).clamp_min(1)
        recall = self.tp.float() / (self.tp + self.fn).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)
        return {"precision": precision, "recall": recall, "f1": f1}

class YOLOFineTuner(L.LightningModule):
    def __init__(
        self,
        model_cfg: str = "yolo26m.yaml",
        pretrained_weights: str = "yolo26m.pt",
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        validation_groups: tuple[str, ...] = VAL_GROUPS,
        metric_confidence: float = 0.25,
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
        self.validation_groups = validation_groups
        self.val_map = torch.nn.ModuleList(
            [MeanAveragePrecision(box_format="xyxy") for _ in validation_groups]
        )
        self.val_quality = torch.nn.ModuleList(
            [DetectionQuality(confidence=metric_confidence) for _ in validation_groups]
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
        self.log(
            f"{stage}/loss", total, prog_bar=True, batch_size=batch_size,
            add_dataloader_idx=False, on_step=False, on_epoch=True,
        )
        if hasattr(components, "items"):
            component_values = components.items()
        else:
            names = getattr(self.model, "loss_names", ())
            component_values = zip(names, components)
        self.log_dict(
            {f"{stage}/{name}": value for name, value in component_values},
            batch_size=batch_size, add_dataloader_idx=False, on_step=False, on_epoch=True,
        )
        return total, output

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")[0]

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        stage = f"val_{self.validation_groups[dataloader_idx]}"
        total, output = self._step(batch, stage)
        detections = nms.non_max_suppression(
            output,
            conf_thres=0.001,
            iou_thres=0.7,
            nc=self.hparams.num_classes,
            multi_label=True,
            max_det=300,
            end2end=getattr(self.model, "end2end", False),
        )
        self._update_metrics(batch, detections, dataloader_idx)
        return total

    def _update_metrics(self, batch: dict, detections: list[torch.Tensor], group: int) -> None:
        size = batch["img"].shape[-1]
        for i, kept in enumerate(detections):
            selected = batch["batch_idx"] == i
            boxes = batch["bboxes"][selected] * size
            half_w, half_h = boxes[:, 2] / 2, boxes[:, 3] / 2
            target_boxes = torch.stack(
                [boxes[:, 0] - half_w, boxes[:, 1] - half_h,
                 boxes[:, 0] + half_w, boxes[:, 1] + half_h], dim=1
            )

            prediction = {
                "boxes": kept[:, :4],
                "scores": kept[:, 4],
                "labels": kept[:, 5].long(),
            }
            target = {"boxes": target_boxes, "labels": batch["cls"][selected]}
            self.val_map[group].update([prediction], [target])
            self.val_quality[group].update(prediction, target)

    def on_validation_epoch_end(self) -> None:
        for name, map_metric, quality_metric in zip(
            self.validation_groups, self.val_map, self.val_quality
        ):
            map_result = map_metric.compute()
            quality = quality_metric.compute()
            self.log(f"val_map/{name}", map_result["map"], prog_bar=True, sync_dist=True)
            self.log(f"val_map50/{name}", map_result["map_50"], prog_bar=True, sync_dist=True)
            self.log(f"val_precision/{name}", quality["precision"], sync_dist=True)
            self.log(f"val_recall/{name}", quality["recall"], sync_dist=True)
            self.log(f"val_f1/{name}", quality["f1"], sync_dist=True)
            map_metric.reset()
            quality_metric.reset()

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
