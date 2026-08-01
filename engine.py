from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Performs a single training step (epoch) for the model.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn: Loss function to use.
        optimizer: Optimizer to use.
        device: Device to run the training on.
    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch.
        example: (0.5, 0.8) -> 50% loss, 80% accuracy
    """
    model.train()
    train_loss, train_correct, train_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item() * len(y)
        train_correct += (y_pred.argmax(dim=1) == y).sum().item()
        train_samples += len(y)

        loss.backward()
        optimizer.step()

    metrics = torch.tensor([train_loss, train_correct, train_samples], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    return metrics[0].item() / metrics[2].item(), metrics[1].item() / metrics[2].item()


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Performs a single testing step (epoch) for the model.
    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): DataLoader for the testing data.
        loss_fn: Loss function to use.
        device: Device to run the testing on.
    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch.
        example: (0.5, 0.8) -> 50% loss, 80% accuracy
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)

            total_loss += loss.item() * len(y)
            total_correct += (test_pred.argmax(dim=1) == y).sum().item()
            total_samples += len(y)

        metrics = torch.tensor([total_loss, total_correct, total_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    return metrics[0].item() / metrics[2].item(), metrics[1].item() / metrics[2].item()


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_sampler: DistributedSampler,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    device: torch.device,
    rank: int = 0,
) -> Dict[str, List[float]]:
    """Trains the model for a given number of epochs and evaluates on the test set.
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the testing data.
        train_sampler (DistributedSampler): Sampler for the training data.
        loss_fn: Loss function to use.
        optimizer: Optimizer to use.
        scheduler: Learning rate scheduler to use.
        epochs (int): Number of epochs to train for.
        device: Device to run the training on.
        rank (int, optional): Rank of the current process. Defaults to 0.
    Returns:
        Dict[str, List]: Dictionary containing training and testing loss and accuracy for each epoch.
        example: {"train_loss": [...], "train_acc": [...], "test_loss": [...], "test_acc": [...]}
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs), disable=(rank != 0)):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if rank == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

    return results
