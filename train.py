import argparse

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import data_setup
import distributed_utils
import engine
import model_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-34 on Food101 with DDP")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for the cosine scheduler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint-path", type=str, default="resnet34_food101.pth")
    parser.add_argument("--results-path", type=str, default="training_results.pth")
    return parser.parse_args()


def worker(rank: int, world_size: int, train_dataset, test_dataset, args: argparse.Namespace) -> None:
    """Worker function for distributed training.
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        train_dataset: Training dataset.
        test_dataset: Testing dataset.
    """
    distributed_utils.setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    distributed_utils.set_seed(args.seed + rank)

    train_loader, test_loader, train_sampler = data_setup.create_dataloaders(
        train_dataset, test_dataset, rank=rank, world_size=world_size, batch_size=args.batch_size
    )

    model = model_builder.ResNet(in_channels=3, out_channels=101, blocks=[3, 4, 6, 3]).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    results = engine.train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        rank=rank,
    )

    if rank == 0:
        torch.save(model.module.state_dict(), args.checkpoint_path)
        torch.save(results, args.results_path)
        print("--> Saved checkpoint and results successfully!")

    distributed_utils.cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()

    train_dataset, test_dataset = data_setup.create_datasets(root=args.data_root)
    world_size = torch.cuda.device_count()

    print(f"Starting DDP across {world_size} GPUs...")
    mp.spawn(worker, args=(world_size, train_dataset, test_dataset, args), nprocs=world_size, join=True)
