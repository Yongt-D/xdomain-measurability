"""Query-conditioned multi-support prototype GAPL-SegNet-v5."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

EPS = 1e-8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    geometry: bool
    prototype_mode: str
    geometry_weighting: bool


ABLATIONS = {
    "A": AblationConfig("A", False, "none", False),
    "B": AblationConfig("B", True, "none", False),
    "C": AblationConfig("C", True, "mean", False),
    "D": AblationConfig("D", True, "mean", True),
    "E": AblationConfig("E", True, "closed_loop", True),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(gpu_id: int | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_id = 0 if gpu_id is None else gpu_id
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} unavailable; found {torch.cuda.device_count()} GPUs")
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    return torch.device(f"cuda:{gpu_id}")


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)
    return (dilated - eroded).clamp(0.0, 1.0)


class BuildingDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(self, image_dir: Path, label_dir: Path, train: bool, image_size: int = 512) -> None:
        labels = {p.stem: p for p in Path(label_dir).iterdir() if p.is_file()}
        self.pairs = [
            (p, labels[p.stem])
            for p in sorted(Path(image_dir).iterdir())
            if p.suffix.lower() in self.IMAGE_EXTENSIONS and p.stem in labels
        ]
        if not self.pairs:
            raise FileNotFoundError(f"No image-label pairs in {image_dir} and {label_dir}")
        self.train, self.image_size = train, image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        label = TF.resize(label, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        if self.train:
            if random.random() < 0.5:
                image, label = TF.hflip(image), TF.hflip(label)
            if random.random() < 0.5:
                image, label = TF.vflip(image), TF.vflip(label)
            if random.random() < 0.5:
                angle = random.choice((90, 180, 270))
                image, label = TF.rotate(image, angle), TF.rotate(label, angle)
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        label = (TF.to_tensor(label) > 0.5).float()
        return image, label


class EpisodeDataset(Dataset):
    """K support samples + one query; query labels never construct prototypes."""

    def __init__(
        self,
        query_set: Dataset,
        support_set: Dataset,
        query_indices: list[int],
        support_indices: list[int],
        seed: int,
        require_distinct: bool,
        support_shots: int,
    ) -> None:
        self.query_set, self.support_set = query_set, support_set
        self.query_indices, self.support_indices = list(query_indices), list(support_indices)
        self.support_shots = support_shots
        minimum = support_shots + 1 if require_distinct else support_shots
        if len(self.support_indices) < minimum:
            raise ValueError("Insufficient support images for multi-support protocol")
        rng = np.random.default_rng(seed)
        self.support_order = [
            self.support_indices[i] for i in rng.permutation(len(self.support_indices))
        ]
        self.require_distinct = require_distinct

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, index: int):
        query_index = self.query_indices[index]
        selected = []
        cursor = index
        while len(selected) < self.support_shots:
            support_index = self.support_order[cursor % len(self.support_order)]
            cursor += 1
            if self.require_distinct and support_index == query_index:
                continue
            if support_index in selected:
                continue
            selected.append(support_index)
        query_image, query_mask = self.query_set[query_index]
        support_pairs = [self.support_set[support_index] for support_index in selected]
        support_images = torch.stack([pair[0] for pair in support_pairs], dim=0)
        support_masks = torch.stack([pair[1] for pair in support_pairs], dim=0)
        return query_image, query_mask, support_images, support_masks


class SegmentationMetrics:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred = (predictions.detach() > 0.5).long().view(-1).cpu()
        target = targets.detach().long().view(-1).cpu()
        self.tp += int(((pred == 1) & (target == 1)).sum())
        self.fp += int(((pred == 1) & (target == 0)).sum())
        self.fn += int(((pred == 0) & (target == 1)).sum())

    def compute(self) -> dict[str, float]:
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + EPS),
            "f1_score": 2 * self.tp / (2 * self.tp + self.fp + self.fn + EPS),
            "precision": self.tp / (self.tp + self.fp + EPS),
            "recall": self.tp / (self.tp + self.fn + EPS),
        }


class SingleResidualDGM(nn.Module):
    """Exactly one residual: F + sigmoid(a) * G(F)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = channels // 4
        self.one = nn.Conv2d(channels, reduced, 1)
        self.three = nn.Conv2d(channels, reduced, 3, padding=1)
        self.dilated = nn.Conv2d(channels, reduced, 3, padding=2, dilation=2)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, reduced, 1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
        self.strength_logit = nn.Parameter(torch.tensor(-1.1))

    def forward(self, features: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return features
        global_feature = F.interpolate(
            self.global_pool(features), features.shape[-2:], mode="bilinear", align_corners=False
        )
        geometry = self.fuse(
            torch.cat(
                [self.one(features), self.three(features), self.dilated(features), global_feature], 1
            )
        )
        return features + torch.sigmoid(self.strength_logit) * geometry


class PrototypeHead(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        geometry_channels = channels // 2
        self.geometry = nn.Sequential(
            nn.Conv2d(channels, geometry_channels, 3, padding=1),
            nn.BatchNorm2d(geometry_channels),
            nn.ReLU(inplace=True),
        )
        self.importance = nn.Sequential(
            nn.Conv2d(channels + geometry_channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.foreground_refiner = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.background_refiner = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.edge_head = nn.Conv2d(geometry_channels, 1, 1)
        self.strength_logit = nn.Parameter(torch.tensor(-1.1))
        self.geometry_gain_logit = nn.Parameter(torch.tensor(-1.1))
        self.loop_logit = nn.Parameter(torch.tensor(-1.1))
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.support_temperature = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def pool(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (features * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(EPS)

    @staticmethod
    def similarity(features: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1, eps=EPS)
        prototype = F.normalize(prototype, p=2, dim=1, eps=EPS).unsqueeze(-1).unsqueeze(-1)
        return (features * prototype).sum(1, keepdim=True)

    def logits_from_prototypes(
        self, query_features: torch.Tensor, foreground: torch.Tensor, background: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        foreground_similarity = self.similarity(query_features, foreground)
        background_similarity = self.similarity(query_features, background)
        prototype_logits = (
            foreground_similarity - background_similarity
        ) / self.temperature.abs().clamp_min(0.05)
        return prototype_logits, foreground_similarity, background_similarity

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        config: AblationConfig,
    ) -> dict[str, torch.Tensor]:
        query_geometry = self.geometry(query_features)
        query_importance = self.importance(torch.cat([query_features, query_geometry], 1))
        if support_features.dim() == 4:
            support_features = support_features.unsqueeze(1)
            support_mask = support_mask.unsqueeze(1)
        batch, shots, channels, height, width = support_features.shape
        support_flat = support_features.reshape(batch * shots, channels, height, width)
        geometry_flat = self.geometry(support_flat)
        importance_flat = self.importance(torch.cat([support_flat, geometry_flat], 1))
        support_importance = importance_flat.reshape(batch, shots, 1, height, width)
        mask_flat = support_mask.reshape(
            batch * shots, support_mask.shape[-3], support_mask.shape[-2], support_mask.shape[-1]
        )
        mask_flat = F.interpolate(mask_flat.float(), (height, width), mode="nearest")
        support_mask = mask_flat.reshape(batch, shots, 1, height, width)
        if config.geometry_weighting:
            geometry_gain = 0.5 * torch.sigmoid(self.geometry_gain_logit)
            support_weight = 1.0 + geometry_gain * (
                2.0 * support_importance - 1.0
            ).abs()
        else:
            geometry_gain = support_importance.new_zeros(())
            support_weight = torch.ones_like(support_importance)

        foreground_weight = support_mask * support_weight
        background_weight = (1.0 - support_mask) * support_weight
        shot_foreground = (support_features * foreground_weight).sum((3, 4)) / (
            foreground_weight.sum((3, 4)).clamp_min(EPS)
        )
        shot_background = (support_features * background_weight).sum((3, 4)) / (
            background_weight.sum((3, 4)).clamp_min(EPS)
        )

        query_global = F.normalize(query_features.mean((2, 3)), p=2, dim=1, eps=EPS)
        support_global = F.normalize(support_features.mean((3, 4)), p=2, dim=2, eps=EPS)
        support_scores = (
            support_global * query_global.unsqueeze(1)
        ).sum(2) / self.support_temperature.abs().clamp_min(0.05)
        support_attention = torch.softmax(support_scores, dim=1)
        foreground = (shot_foreground * support_attention.unsqueeze(2)).sum(1)
        background = (shot_background * support_attention.unsqueeze(2)).sum(1)

        foreground_unit = F.normalize(foreground, p=2, dim=1, eps=EPS).unsqueeze(1)
        background_unit = F.normalize(background, p=2, dim=1, eps=EPS).unsqueeze(1)
        foreground_consistency = 1.0 - (
            F.normalize(shot_foreground, p=2, dim=2, eps=EPS) * foreground_unit
        ).sum(2)
        background_consistency = 1.0 - (
            F.normalize(shot_background, p=2, dim=2, eps=EPS) * background_unit
        ).sum(2)
        prototype_consistency = 0.5 * (
            (foreground_consistency * support_attention).sum(1).mean()
            + (background_consistency * support_attention).sum(1).mean()
        )
        attention_entropy = -(
            support_attention * support_attention.clamp_min(EPS).log()
        ).sum(1).mean() / math.log(shots)
        prototype_logits, fg_similarity, bg_similarity = self.logits_from_prototypes(
            query_features, foreground, background
        )
        return {
            "prototype_logits": prototype_logits,
            "foreground_similarity": fg_similarity,
            "background_similarity": bg_similarity,
            "query_importance": query_importance,
            "support_importance": support_importance,
            "edge_logits": self.edge_head(query_geometry),
            "foreground_prototype": foreground,
            "background_prototype": background,
            "prototype_consistency": prototype_consistency,
            "support_attention": support_attention,
            "support_attention_entropy": attention_entropy,
            "strength": torch.sigmoid(self.strength_logit),
            "geometry_gain": geometry_gain,
            "loop_strength": prototype_logits.new_zeros(()),
        }

class GAPLSegNetV5(nn.Module):
    def __init__(
        self,
        config: AblationConfig,
        pretrained: bool = True,
        pretrained_weights: str | Path | None = None,
        fusion_mode: str = "plain",
    ) -> None:
        super().__init__()
        self.config = config
        self.fusion_mode = fusion_mode
        if pretrained and pretrained_weights:
            vgg = models.vgg16_bn(weights=None)
            vgg.load_state_dict(
                torch.load(pretrained_weights, map_location="cpu", weights_only=False)
            )
        else:
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = models.vgg16_bn(weights=weights)
        features = list(vgg.features.children())
        self.encoder1 = nn.Sequential(*features[0:7])
        self.encoder2 = nn.Sequential(*features[7:14])
        self.encoder3 = nn.Sequential(*features[14:24])
        self.encoder4 = nn.Sequential(*features[24:34])
        self.geometry2 = SingleResidualDGM(128)
        self.geometry3 = SingleResidualDGM(256)
        self.geometry4 = SingleResidualDGM(512)
        self.up4 = self.up(512, 256)
        self.dec4, self.res4 = self.decoder(512, 256), nn.Conv2d(512, 256, 1)
        self.up3 = self.up(256, 128)
        self.dec3, self.res3 = self.decoder(256, 128), nn.Conv2d(256, 128, 1)
        self.up2 = self.up(128, 64)
        self.dec2, self.res2 = self.decoder(128, 64), nn.Conv2d(128, 64, 1)
        self.up1 = self.up(64, 32)
        self.processor = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.prototype = PrototypeHead(32)
        self.prototype_weight = 0.0

    @staticmethod
    def up(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def decoder(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def resize(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, target.shape[-2:], mode="bilinear", align_corners=False)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(images)
        e2 = self.geometry2(self.encoder2(e1), self.config.geometry)
        e3 = self.geometry3(self.encoder3(e2), self.config.geometry)
        e4 = self.geometry4(self.encoder4(e3), self.config.geometry)
        d4_cat = torch.cat([self.resize(self.up4(e4), e3), e3], 1)
        d4 = self.dec4(d4_cat) + self.res4(d4_cat)
        d3_cat = torch.cat([self.resize(self.up3(d4), e2), e2], 1)
        d3 = self.dec3(d3_cat) + self.res3(d3_cat)
        d2_cat = torch.cat([self.resize(self.up2(d3), e1), e1], 1)
        d2 = self.dec2(d2_cat) + self.res2(d2_cat)
        return self.processor(self.up1(d2))

    def forward(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor | None = None,
        support_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output_size = query_images.shape[-2:]
        query_features = self.extract_features(query_images)
        base_logits = self.classifier(query_features)
        if self.config.prototype_mode != "none":
            if support_images is None or support_masks is None:
                raise ValueError("Ablations C/D/E require support_images and support_masks")
            if support_images.dim() == 5:
                batch, shots, channels, height, width = support_images.shape
                support_flat = support_images.reshape(batch * shots, channels, height, width)
                support_features = self.extract_features(support_flat)
                support_features = support_features.reshape(
                    batch, shots, support_features.shape[1], support_features.shape[2], support_features.shape[3]
                )
            else:
                support_features = self.extract_features(support_images)
            head = self.prototype(query_features, support_features, support_masks, self.config)
            if self.fusion_mode == "confidence":
                prototype_gate = (
                    0.25 + 0.75 * torch.tanh(head["prototype_logits"].abs() / 2.0)
                ).detach()
            else:
                prototype_gate = torch.ones_like(head["prototype_logits"])
            head["prototype_gate"] = prototype_gate
            logits = base_logits + head["strength"] * prototype_gate * head["prototype_logits"]
        else:
            query_geometry = self.prototype.geometry(query_features)
            zero_map = torch.zeros_like(base_logits)
            zero_prototype = torch.zeros(
                query_features.shape[0],
                query_features.shape[1],
                dtype=query_features.dtype,
                device=query_features.device,
            )
            head = {
                "prototype_logits": zero_map,
                "query_importance": torch.ones_like(base_logits),
                "support_importance": torch.ones_like(base_logits),
                "edge_logits": self.prototype.edge_head(query_geometry),
                "foreground_prototype": zero_prototype,
                "background_prototype": zero_prototype,
                "prototype_gate": torch.zeros_like(base_logits),
            }
            logits = base_logits
        result = {"logits": logits, "base_logits": base_logits, **head}
        for key in ("logits", "base_logits", "prototype_logits", "prototype_gate", "query_importance", "edge_logits"):
            if result[key].shape[-2:] != output_size:
                result[key] = F.interpolate(
                    result[key], output_size, mode="bilinear", align_corners=False
                )
        result["predictions"] = torch.sigmoid(result["logits"])
        return result

    def set_epoch_progress(self, epoch: int, total_epochs: int) -> None:
        progress = ((epoch - 1) / max(total_epochs - 1, 1)) / 0.20
        self.prototype_weight = 0.30 * min(progress, 1.0)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        query_targets: torch.Tensor,
        support_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if query_targets.dim() == 3:
            query_targets = query_targets.unsqueeze(1)
        query_targets = query_targets.float()
        segmentation = F.binary_cross_entropy_with_logits(outputs["logits"], query_targets)
        base = F.binary_cross_entropy_with_logits(outputs["base_logits"], query_targets)
        zero = segmentation.new_zeros(())
        if self.config.prototype_mode != "none":
            prototype_bce = F.binary_cross_entropy_with_logits(
                outputs["prototype_logits"], query_targets
            )
            prototype_probability = torch.sigmoid(outputs["prototype_logits"])
            prototype_intersection = (
                prototype_probability * query_targets
            ).sum((1, 2, 3))
            prototype_dice = 1.0 - (
                (2.0 * prototype_intersection + 1.0)
                / (
                    prototype_probability.sum((1, 2, 3))
                    + query_targets.sum((1, 2, 3))
                    + 1.0
                )
            ).mean()
            prototype = 0.5 * prototype_bce + 0.5 * prototype_dice
            consistency = outputs["prototype_consistency"]
        else:
            prototype = zero
            consistency = zero
        if self.config.geometry_weighting:
            support_targets = F.interpolate(
                support_targets.float(), outputs["support_importance"].shape[-2:], mode="nearest"
            )
            with autocast(enabled=False):
                probability = outputs["support_importance"].float().clamp(EPS, 1.0 - EPS)
                target = support_targets.float()
                bce = F.binary_cross_entropy(probability, target)
                intersection = (probability * target).sum((1, 2, 3))
                dice = 1.0 - (
                    (2.0 * intersection + 1.0)
                    / (
                        probability.sum((1, 2, 3))
                        + target.sum((1, 2, 3))
                        + 1.0
                    )
                ).mean()
                importance = 0.5 * bce + 0.5 * dice
        else:
            importance = zero
        if self.config.geometry:
            edge = F.binary_cross_entropy_with_logits(
                outputs["edge_logits"], boundary_target(query_targets)
            )
        else:
            edge = zero
        total = (
            segmentation
            + 0.25 * base
            + self.prototype_weight * prototype
            + 0.03 * importance
            + 0.15 * edge
        )
        parts = {
            "total": total,
            "segmentation": segmentation,
            "base": base,
            "prototype": prototype,
            "importance": importance,
            "edge": edge,
            "consistency": consistency,
            "prototype_weight": total.new_tensor(self.prototype_weight),
        }
        return total, {key: float(value.detach()) for key, value in parts.items()}


def create_dataloaders(
    dataset_path: str, sample_size: int, batch_size: int, sample_seed: int, workers: int,
    support_shots: int,
):
    root = Path(dataset_path)
    train_plain = BuildingDataset(
        root / "train" / "image", root / "train" / "label", train=False
    )
    if sample_size > len(train_plain):
        raise ValueError(f"sample_size={sample_size} exceeds {len(train_plain)} pairs")
    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(len(train_plain), sample_size, replace=False).tolist()
    rng.shuffle(selected)
    val_count = min(max(1, round(sample_size * 0.2)), sample_size - 2)
    val_indices, train_indices = selected[:val_count], selected[val_count:]
    train_aug = BuildingDataset(
        root / "train" / "image", root / "train" / "label", train=True
    )
    test_plain = BuildingDataset(
        root / "test" / "image", root / "test" / "label", train=False
    )
    train_episodes = EpisodeDataset(
        train_aug, train_aug, train_indices, train_indices, sample_seed + 11, True,
        support_shots,
    )
    val_episodes = EpisodeDataset(
        train_plain, train_plain, val_indices, train_indices, sample_seed + 17, False,
        support_shots,
    )
    test_episodes = EpisodeDataset(
        test_plain,
        train_plain,
        list(range(len(test_plain))),
        train_indices,
        sample_seed + 23,
        False,
        support_shots,
    )
    common = {"num_workers": workers, "pin_memory": True}
    if workers:
        common["persistent_workers"] = True
    return (
        DataLoader(train_episodes, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_episodes, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_episodes, batch_size=1, shuffle=False, **common),
        len(train_indices),
        len(val_indices),
    )


def set_lr(
    optimizer: optim.Optimizer,
    base_lrs: list[float],
    epoch_progress: float,
    total_epochs: int,
    warmup_epochs: int = 5,
) -> None:
    if epoch_progress < warmup_epochs:
        scale = max(epoch_progress / warmup_epochs, 1e-6)
    else:
        progress = min(
            (epoch_progress - warmup_epochs) / max(total_epochs - warmup_epochs, 1), 1.0
        )
        scale = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def run_epoch(
    model, loader, optimizer, base_lrs, device, epoch, epochs, scaler, amp, train: bool
):
    model.train(train)
    if train:
        model.set_epoch_progress(epoch, epochs)
    metrics, sums = SegmentationMetrics(), {}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for index, batch in enumerate(loader):
            query_images, query_masks, support_images, support_masks = [
                tensor.to(device, non_blocking=True) for tensor in batch
            ]
            if train:
                set_lr(
                    optimizer,
                    base_lrs,
                    epoch - 1 + (index + 1) / max(len(loader), 1),
                    epochs,
                )
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                outputs = model(query_images, support_images, support_masks)
                loss, parts = model.compute_loss(outputs, query_masks, support_masks)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            metrics.update(outputs["predictions"], query_masks)
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
    result = metrics.compute()
    result.update({key: value / max(len(loader), 1) for key, value in sums.items()})
    if train:
        result["lr"] = optimizer.param_groups[0]["lr"]
    return result


def self_check(device: torch.device) -> None:
    seed_everything(123)
    model = GAPLSegNetV5(
        ABLATIONS["C"], pretrained=False, fusion_mode="confidence"
    ).to(device).eval()
    query = torch.randn(2, 3, 64, 64, device=device)
    support = torch.randn(2, 4, 3, 64, 64, device=device)
    mask_a = torch.zeros(2, 4, 1, 64, 64, device=device)
    mask_b = torch.zeros_like(mask_a)
    mask_b[:, :, :, 16:48, 16:48] = 1
    outputs_a = model(query, support, mask_a)
    outputs_b = model(query, support, mask_b)
    logits_a, logits_b = outputs_a["logits"], outputs_b["logits"]
    repeat = model(query, support, mask_b)["logits"]
    support_mask_delta = float((logits_a - logits_b).detach().abs().max())
    enabled_repeat_delta = float((logits_b - repeat).detach().abs().max())
    geometry_gain = float(outputs_b["geometry_gain"].detach())
    loop_strength = float(outputs_b["loop_strength"].detach())
    targets = (torch.rand(2, 1, 64, 64, device=device) > 0.7).float()
    loss, _ = model.compute_loss(outputs_b, targets, mask_b)
    loss.backward()
    gradient = model.prototype.temperature.grad
    gradient_sum = 0.0 if gradient is None else float(gradient.abs().sum())
    selection_gradient = model.prototype.support_temperature.grad
    selection_gradient_sum = (
        0.0 if selection_gradient is None else float(selection_gradient.abs().sum())
    )
    attention_sum_error = float(
        (outputs_b["support_attention"].sum(1) - 1.0).abs().max().detach()
    )
    model.config = ABLATIONS["B"]
    disabled_1 = model(query, support, mask_b)["logits"]
    disabled_2 = model(query, support, mask_b)["logits"]
    disable_delta = float((logits_b.detach() - disabled_1.detach()).abs().max())
    disabled_repeat_delta = float((disabled_1.detach() - disabled_2.detach()).abs().max())
    checks = {
        "support_mask_changes_query_logits": support_mask_delta > 1e-6,
        "prototype_parameter_gradient_nonzero": gradient_sum > 0 and math.isfinite(gradient_sum),
        "support_selector_gradient_nonzero": (
            selection_gradient_sum > 0 and math.isfinite(selection_gradient_sum)
        ),
        "support_attention_normalized": attention_sum_error < 1e-6,
        "disabling_prototype_changes_output": disable_delta > 1e-6,
        "fixed_input_is_reproducible": (
            enabled_repeat_delta == 0.0 and disabled_repeat_delta == 0.0
        ),
        "geometry_gain_is_bounded": 0.0 <= geometry_gain <= 0.5,
        "loop_strength_is_bounded": 0.0 <= loop_strength <= 0.25,
    }
    report = {
        "checks": checks,
        "support_mask_logit_max_delta": support_mask_delta,
        "prototype_parameter_gradient_abs_sum": gradient_sum,
        "support_selector_gradient_abs_sum": selection_gradient_sum,
        "support_attention_sum_error": attention_sum_error,
        "prototype_disable_logit_max_delta": disable_delta,
        "enabled_repeat_max_delta": enabled_repeat_delta,
        "disabled_repeat_max_delta": disabled_repeat_delta,
        "geometry_gain": geometry_gain,
        "loop_strength": loop_strength,
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise RuntimeError(f"GAPL-SegNet-v5 self-check failed: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-support episodic GAPL-SegNet-v5")
    parser.add_argument("--dataset-path", default="datadir/whu_building")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--support-shots", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--pretrained-weights", default="weights/vgg16_bn-6c64b313.pth"
    )
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="C")
    parser.add_argument("--fusion-mode", choices=("plain", "confidence"), default="plain")
    parser.add_argument("--output-dir", default="outputs/v5_whu100_C_plain_seed0")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    device = setup_device(args.gpu)
    seed_everything(args.seed)
    if args.self_check:
        self_check(device)
        return

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders = create_dataloaders(
        args.dataset_path,
        args.sample_size,
        args.batch_size,
        args.sample_seed,
        args.num_workers,
        args.support_shots,
    )
    train_loader, val_loader, test_loader, train_count, val_count = loaders
    pretrained_weights = None
    if not args.no_pretrained and args.pretrained_weights:
        pretrained_weights = Path(args.pretrained_weights)
        if not pretrained_weights.is_absolute():
            pretrained_weights = Path(__file__).resolve().parent / pretrained_weights
        if not pretrained_weights.is_file():
            raise FileNotFoundError(
                f"Project-local pretrained weight missing: {pretrained_weights}"
            )
    config = ABLATIONS[args.ablation]
    model = GAPLSegNetV5(
        config,
        pretrained=not args.no_pretrained,
        pretrained_weights=pretrained_weights,
        fusion_mode=args.fusion_mode,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    amp = bool(args.enable_amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp)
    history, best_iou, best_epoch = [], -1.0, 0
    checkpoint = output / "best_gaplsegnet_v5.pth"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            base_lrs,
            device,
            epoch,
            args.epochs,
            scaler,
            amp,
            True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            base_lrs,
            device,
            epoch,
            args.epochs,
            scaler,
            amp,
            False,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["iou"] > best_iou:
            best_iou, best_epoch = val_metrics["iou"], epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_iou": best_iou,
                    "args": vars(args),
                    "ablation_config": asdict(config),
                    "version": "GAPLsegnetV5-query-selected",
                },
                checkpoint,
            )

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer,
        base_lrs,
        device,
        args.epochs,
        args.epochs,
        scaler,
        amp,
        False,
    )
    results = {
        "version": "GAPLsegnetV5-query-selected",
        "protocol": f"1-way {args.support_shots}-shot support/query",
        "ablation_config": asdict(config),
        "elapsed_seconds": time.time() - started,
        "train_samples": train_count,
        "validation_samples": val_count,
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "test_metrics": test_metrics,
        "args": vars(args),
        "history": history,
    }
    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

