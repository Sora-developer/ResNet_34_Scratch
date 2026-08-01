from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """Saves the model's state_dict to the specified directory with the given model name."""
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target_dir_path / model_name)


def load_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Loads the model's state_dict from the specified checkpoint path and moves it to the given device."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)


def plot_curves(results: Dict[str, List[float]]) -> None:
    """Plots the training and testing loss and accuracy curves from the results dictionary.
    Args:
        results (Dict[str, List[float]]): A dictionary containing training and testing loss and accuracy.
            Expected keys: "train_loss", "test_loss", "train_acc", "test_acc".
    """
    epochs = range(len(results["train_loss"]))

    plt.figure()
    plt.plot(epochs, results["train_loss"], c="r", label="Train Loss")
    plt.plot(epochs, results["test_loss"], c="g", label="Test Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, results["train_acc"], c="r", label="Train Acc")
    plt.plot(epochs, results["test_acc"], c="g", label="Test Acc")
    plt.title("Accuracy Curves")
    plt.grid(True)
    plt.legend()
    plt.show()
