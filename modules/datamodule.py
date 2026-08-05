"""Feeds the training loop, and splits validation by a chosen context class."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import DetectionDataset


class DetectionDataModule(L.LightningDataModule):
    """Validation is served as two loaders, split on whether the context class is present.

    For abandoned-object work that class is "person": an object with nobody
    nearby is the case that matters, and measuring it separately is the only way
    to see whether the model handles it.
    """

    def __init__(
        self,
        train_dataset: DetectionDataset,
        val_dataset: DetectionDataset,
        batch_size: int = 16,
        num_workers: int = 4,
        context_class: str | None = "person",
        balance_context: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_class = context_class
        self.balance_context = balance_context

    def _train_sampler(self) -> WeightedRandomSampler | None:
        if not self.balance_context or self.context_class is None:
            return None

        flags = self.train_dataset.has_class(self.context_class)
        with_context = sum(flags)
        without_context = len(flags) - with_context
        if not with_context or not without_context:
            return None

        # dividing by group size makes both groups weigh the same in total,
        # so a draw is equally likely to land in either one
        weights = [
            1.0 / with_context if flag else 1.0 / without_context for flag in flags
        ]
        return WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(flags),
            replacement=True,
        )

    def train_dataloader(self) -> DataLoader:
        sampler = self._train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def _val_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.context_class is None:
            return self._val_loader(self.val_dataset)

        flags = self.val_dataset.has_class(self.context_class)
        # index 0 is the case the whole project is about: an object with nobody nearby
        without_context = [i for i, flag in enumerate(flags) if not flag]
        with_context = [i for i, flag in enumerate(flags) if flag]
        return [
            self._val_loader(Subset(self.val_dataset, without_context)),
            self._val_loader(Subset(self.val_dataset, with_context)),
        ]

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        bboxes = [target[0] for target in targets]
        labels = [target[1] for target in targets]

        # boxes of every image are concatenated into one long tensor, and
        # batch_idx records which image each box came from
        counts = torch.tensor([len(box) for box in bboxes])
        batch_idx = torch.repeat_interleave(torch.arange(len(bboxes)), counts)

        return {
            "img": torch.stack(images),
            "bboxes": torch.cat(bboxes),
            "cls": torch.cat(labels),
            "batch_idx": batch_idx,
        }
