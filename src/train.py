# src/train.py
import argparse
import os
import csv
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_setup import get_dataloaders
from model import MNISTNet, count_params


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN more deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train MNIST CNN")
    p.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # Model
    model = MNISTNet(hidden_dim=args.hidden_dim, dropout_p=args.dropout).to(device)
    print(model)
    print(f"Trainable params: {count_params(model):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logging setup
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best.pt")

    # Train loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, best_path)

        # Current LR (for schedulers later; for Adam it stays constant unless you add one)
        current_lr = optimizer.param_groups[0]["lr"]

        # Console log
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"best_acc={best_acc:.4f} | "
            f"lr={current_lr:.2e} | {dt:.1f}s"
        )

        # CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.6f}"])

    print(f"\nTraining complete. Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
