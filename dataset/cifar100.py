"""CIFAR-100 data loading and reproducible synthetic smoke-test data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class Cutout:
    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        center_y = torch.randint(height, (1,)).item()
        center_x = torch.randint(width, (1,)).item()
        half = self.length // 2
        y1, y2 = max(0, center_y - half), min(height, center_y + half)
        x1, x2 = max(0, center_x - half), min(width, center_x + half)
        image = image.clone()
        image[:, y1:y2, x1:x2] = 0
        return image


def cifar100_transforms(autoaugment: bool = False, cutout: int = 0):
    train_steps: list[object] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if autoaugment:
        train_steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    if cutout > 0:
        train_steps.append(Cutout(cutout))
    test_steps = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
    return transforms.Compose(train_steps), transforms.Compose(test_steps)


@dataclass(frozen=True)
class CIFAR100Loaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def get_cifar100_dataloaders(
    data_folder: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    validation_fraction: float = 0.0,
    seed: int = 42,
    autoaugment: bool = False,
    cutout: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Compatibility loader returning train and test dataloaders."""
    loaders = build_cifar100_loaders(
        data_folder=data_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        validation_fraction=validation_fraction,
        seed=seed,
        autoaugment=autoaugment,
        cutout=cutout,
        pin_memory=pin_memory,
    )
    validation = loaders.validation if validation_fraction > 0 else loaders.test
    return loaders.train, validation


def build_cifar100_loaders(
    data_folder: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    validation_fraction: float = 0.0,
    seed: int = 42,
    autoaugment: bool = False,
    cutout: int = 0,
    pin_memory: bool | None = None,
) -> CIFAR100Loaders:
    root = Path(data_folder)
    train_transform, test_transform = cifar100_transforms(autoaugment, cutout)
    full_train = datasets.CIFAR100(root=root, train=True, download=download, transform=train_transform)
    test = datasets.CIFAR100(root=root, train=False, download=download, transform=test_transform)
    if validation_fraction > 0.0:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        val_size = int(round(len(full_train) * validation_fraction))
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(seed)
        train, validation_indices = random_split(full_train, [train_size, val_size], generator=generator)
        # Validation must use deterministic transforms, so recreate the dataset and retain indices.
        validation_base = datasets.CIFAR100(
            root=root, train=True, download=False, transform=test_transform
        )
        validation = Subset(validation_base, validation_indices.indices)
    else:
        train = full_train
        validation = test
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, drop_last=True, **common)
    validation_loader = DataLoader(validation, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test, shuffle=False, drop_last=False, **common)
    return CIFAR100Loaders(train_loader, validation_loader, test_loader)


def get_fake_cifar100_dataloaders(
    batch_size: int = 8,
    train_size: int = 32,
    test_size: int = 16,
    num_workers: int = 0,
    seed: int = 7,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    train = datasets.FakeData(
        size=train_size,
        image_size=(3, 32, 32),
        num_classes=100,
        transform=transform,
        random_offset=seed,
    )
    test = datasets.FakeData(
        size=test_size,
        image_size=(3, 32, 32),
        num_classes=100,
        transform=transform,
        random_offset=seed + train_size,
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
