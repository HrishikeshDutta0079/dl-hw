import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

# If using road_dataset's load_data:
from .datasets.road_dataset import load_data

from .models import Detector, load_model, save_model

def iou_loss(logits, targets, eps=1e-6):
    """
    Computes a soft IoU loss for multi-class segmentation.
    
    Args:
        logits: Tensor of shape (B, num_classes, H, W)
        targets: Tensor of shape (B, H, W) with class indices
        eps: small constant for numerical stability
    Returns:
        IoU loss (scalar tensor)
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    # Create one-hot encoding of targets: shape (B, num_classes, H, W)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Compute intersection and union along spatial dimensions
    intersection = (probs * targets_onehot).sum(dim=(2, 3))
    union = (probs + targets_onehot - probs * targets_onehot).sum(dim=(2, 3))
    iou = (intersection + eps) / (union + eps)
    
    # Return 1 - mean IoU across classes as loss
    return 1 - iou.mean()


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # If your dataset returns a dictionary { "image", "track", "depth", ... }:
    train_data = load_data(
        "drive_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2
    )
    val_data = load_data(
        "drive_data/val",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2
    )

    # create loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        # ------------------------------
        # 1) TRAIN PHASE
        # ------------------------------
        for batch in train_data:
            # batch is a dict, so extract your inputs/labels
            # e.g. "image" is your input, "track" is your segmentation label
            img = batch["image"].to(device)         # shape (B, 3, H, W)
            label = batch["track"].long().to(device)  # shape (B, H, W) for CrossEntropyLoss

            optimizer.zero_grad()

            # forward pass
            seg_logits, raw_depth = model(img)  # seg_logits -> (B, num_classes, H, W)

            # compute loss: combine cross-entropy and IoU loss
            ce_loss = loss_func(seg_logits, label)
            iou = iou_loss(seg_logits, label)
            loss = ce_loss + 0.5 * iou

            # backprop
            loss.backward()
            optimizer.step()

            # compute training accuracy for this mini-batch
            train_preds = seg_logits.argmax(dim=1)      # shape (B, H, W)
            train_acc = (train_preds == label).float().mean().item()
            metrics["train_acc"].append(train_acc)

            # log the training loss
            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # ------------------------------
        # 2) VALIDATION PHASE
        # ------------------------------
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                img = batch["image"].to(device)
                label = batch["track"].long().to(device)

                seg_logits, raw_depth = model(img)
                val_preds = seg_logits.argmax(dim=1)
                val_acc = (val_preds == label).float().mean().item()
                metrics["val_acc"].append(val_acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparameters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train()
    args = parser.parse_args()
    train(**vars(args))
