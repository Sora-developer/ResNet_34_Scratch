# ResNet-34 from Scratch — Food101 (DDP)

ResNet-34 implemented from scratch and trained on Food101 with `DistributedDataParallel` across all
available GPUs on a single node.

## Structure

| File                   | Purpose                                                              |
|------------------------|-----------------------------------------------------------------------|
| `model_builder.py`     | `ResBlock` and `ResNet` architecture                                  |
| `data_setup.py`        | Food101 dataset/transform construction and distributed dataloaders    |
| `distributed_utils.py` | Process group setup/teardown, seeding                                 |
| `engine.py`            | `train_step`, `test_step`, and the epoch loop (DDP-aware metric sync) |
| `train.py`             | Entry point: launches training across GPUs via `mp.spawn`             |
| `predict.py`           | Loads a checkpoint and visualizes predictions on the test set         |
| `utils.py`             | Checkpoint save/load, loss and accuracy curve plotting                |

## Usage

```bash
pip install -r requirements.txt

# Train (spawns one process per visible GPU)
python train.py

# Train with custom hyperparameters
python train.py --epochs 30 --learning-rate 5e-4 --weight-decay 5e-5 --batch-size 64

# Plot results
python -c "import torch, utils; utils.plot_curves(torch.load('training_results.pth'))"

# Run inference on random test samples
python predict.py
```

Training produces `resnet34_food101.pth` (model weights) and `training_results.pth` (per-epoch
loss/accuracy history) in the working directory.

### `train.py` arguments

| Argument             | Default                    | Description                              |
|----------------------|-----------------------------|-------------------------------------------|
| `--epochs`            | `15`                        | Number of training epochs                 |
| `--learning-rate`     | `0.001`                     | Adam learning rate                        |
| `--weight-decay`      | `1e-4`                      | Adam weight decay                         |
| `--batch-size`        | `32`                        | Per-GPU batch size                        |
| `--label-smoothing`   | `0.1`                       | Label smoothing for `CrossEntropyLoss`    |
| `--eta-min`           | `1e-6`                      | Minimum LR for the cosine scheduler       |
| `--seed`              | `42`                        | Base random seed (offset by rank)         |
| `--data-root`         | `data`                      | Food101 download/cache directory          |
| `--checkpoint-path`   | `resnet34_food101.pth`      | Where to save model weights               |
| `--results-path`      | `training_results.pth`      | Where to save the metrics history         |

### `predict.py` arguments

| Argument             | Default                    | Description                    |
|----------------------|-----------------------------|----------------------------------|
| `--checkpoint-path`   | `resnet34_food101.pth`      | Model weights to load           |
| `--data-root`         | `data`                      | Food101 download/cache directory|
| `--rows`, `--cols`    | `4`, `4`                    | Prediction grid size             |

## Loss and Accuracy curves after running the model for 40 epochs
![Loss Curve][Loss_Curves_Final_40epochs.png]
![Acc Curve][Acc_Curves_Final_40epochs.png]
