from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

NUM_WORKERS = 2


def build_transforms() -> transforms.Compose:
    """Builds a composition of data transformations for training and testing datasets.
    Returns:
        transforms.Compose: A composition of data transformations.
    """
    return transforms.Compose([
        transforms.Resize(size=(232, 232)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
    ])


def create_datasets(root: str = "data") -> Tuple[Dataset, Dataset]:
    """Creates training and testing datasets for the Food101 dataset.
    Args:
        root (str): The root directory where the dataset will be stored or loaded from.
    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    transform = build_transforms()

    train_dataset = datasets.Food101(root=root, split="train", transform=transform, download=True)
    test_dataset = datasets.Food101(root=root, split="test", transform=transform, download=True)

    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    rank: int,
    world_size: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    """Creates DataLoaders for training and testing datasets with distributed sampling.
    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The testing dataset.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        batch_size (int): The batch size for the DataLoaders.
    Returns:
        Tuple[DataLoader, DataLoader, DistributedSampler]: A tuple containing the training DataLoader, testing DataLoader, and the training DistributedSampler.
    """
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        pin_memory=True, num_workers=NUM_WORKERS,
    )

    return train_loader, test_loader, train_sampler
