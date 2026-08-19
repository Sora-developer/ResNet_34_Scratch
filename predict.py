import argparse

import matplotlib.pyplot as plt
import torch

import data_setup
import model_builder
import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ResNet-34 predictions on Food101 test samples")
    parser.add_argument("--checkpoint-path", type=str, default="resnet34_food101.pth")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="If set, saves the prediction grid to this path instead of (or in addition to) showing it",
    )
    return parser.parse_args()


def visualize_predictions(
    model: torch.nn.Module,
    dataset,
    class_names: list,
    device: torch.device,
    rows: int = 4,
    cols: int = 4,
    save_path: str = None,
) -> None:
    """Visualizes predictions of a model on a dataset.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        dataset: The dataset to visualize predictions on. Should use a deterministic
            (non-augmented) transform - see data_setup.build_eval_transforms - so the
            displayed "Original" image actually matches what was scored.
        class_names (list): List of class names corresponding to the dataset.
        device (torch.device): The device to run the model on.
        rows (int): Number of rows in the visualization grid.
        cols (int): Number of columns in the visualization grid.
        save_path (str, optional): If provided, saves the figure here instead of/in
            addition to displaying it. Needed for headless (no-display) environments.
    """
    model.eval()
    fig = plt.figure(figsize=(15, 15))

    for i in range(1, rows * cols + 1):
        idx = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[idx]

        with torch.inference_mode():
            pred = model(img.to(device).unsqueeze(dim=0)).argmax(dim=1)

        plt.subplot(rows, cols, i)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Original: {class_names[label]}\nPredicted: {class_names[pred]}")
        plt.axis(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"--> Saved prediction grid to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_dataset = data_setup.create_datasets(root=args.data_root)
    class_names = test_dataset.classes

    model = model_builder.ResNet(in_channels=3, out_channels=101, blocks=[3, 4, 6, 3])
    model = utils.load_model(model, args.checkpoint_path, device)

    visualize_predictions(
        model, test_dataset, class_names, device,
        rows=args.rows, cols=args.cols, save_path=args.save_path,
    )
