"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from .models import load_model, save_model
from .datasets.road_dataset import load_data   

# Copilot code
# --------------------------------------------------------------------------- #
# Loss & Metric helpers
# --------------------------------------------------------------------------- #

def waypoint_l1_loss(pred, tgt, mask=None):
    """
    Smooth‑L1 loss over (x,y) way‑points.  If a boolean mask (B, n_wp) is
    provided, only the "clean" way‑points contribute to the loss.
    """
    if mask is not None:
        mask = mask.unsqueeze(-1)          # (B, n_wp, 1)
        diff = F.smooth_l1_loss(pred * mask, tgt * mask, reduction="sum")
        denom = mask.sum().clamp(min=1.0)
        return diff / denom
    else:
        return F.smooth_l1_loss(pred, tgt)


@torch.no_grad()
def compute_offline_metrics(pred, tgt, mask=None):
    """Return mean longitudinal & lateral L1 errors."""
    if mask is not None:
        mask = mask.float()
    else:
        mask = torch.ones_like(tgt[..., 0])

    lon_err = torch.abs(pred[..., 0] - tgt[..., 0]) * mask
    lat_err = torch.abs(pred[..., 1] - tgt[..., 1]) * mask
    denom = mask.sum().clamp(min=1.0)
    return lon_err.sum() / denom, lat_err.sum() / denom


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train(
    exp_dir="logs",
    model_name="mlp_planner",
    num_epoch=50,
    lr=1e-3,
    batch_size=128,
    seed=2024,
    transform_pipeline="state_only",
    **model_kwargs,
):
    if model_name == "cnn_planner" and transform_pipeline == "state_only":
        # CNN needs RGB frames; the state‑only pipeline has no 'image' key
        transform_pipeline = "default"

    # ------------------------------------------------------------------ device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # -------------------------------------------------------- determinism‑ish
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------- logging paths
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # ---------------------------------------------------- data + model init
    model = load_model(model_name, **model_kwargs).to(device).train()

    train_loader = load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline=transform_pipeline,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    # ---------------------------------------------------------------- loop
    for epoch in range(num_epoch):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            target_wp = batch["waypoints"].to(device)              # (B, n_wp, 2)
            mask = batch.get("waypoints_mask", None)
            if mask is not None:
                mask = mask.to(device)

            # ------------------------------ choose correct inputs
            if model_name == "cnn_planner":
                pred_wp = model(image=batch["image"].float().to(device))
            else:
                pred_wp = model(
                    track_left=batch["track_left"].float().to(device),
                    track_right=batch["track_right"].float().to(device),
                )

            # ------------------------------ loss & back‑prop
            loss = waypoint_l1_loss(pred_wp, target_wp, mask)
            loss.backward()
            optimizer.step()

            # ------------------------------ offline metrics
            lon_err, lat_err = compute_offline_metrics(pred_wp.detach(), target_wp, mask)
            logger.add_scalar("train/lon_err", lon_err.item(), global_step)
            logger.add_scalar("train/lat_err", lat_err.item(), global_step)
            logger.add_scalar("train/loss", loss.item(), global_step)

            global_step += 1

        # ------------------------------------------------ validation epoch
        model.eval()
        val_lon, val_lat = 0.0, 0.0
        n_batches = 0
        with torch.inference_mode():
            for batch in val_loader:
                target_wp = batch["waypoints"].to(device)
                mask = batch.get("waypoints_mask", None)
                if mask is not None:
                    mask = mask.to(device)

                if model_name == "cnn_planner":
                    pred_wp = model(image=batch["image"].float().to(device))
                else:
                    pred_wp = model(
                        track_left=batch["track_left"].float().to(device),
                        track_right=batch["track_right"].float().to(device),
                    )

                lon_err, lat_err = compute_offline_metrics(pred_wp, target_wp, mask)
                val_lon += lon_err.item()
                val_lat += lat_err.item()
                n_batches += 1

        val_lon /= max(n_batches, 1)
        val_lat /= max(n_batches, 1)
        logger.add_scalar("val/lon_err", val_lon, epoch)
        logger.add_scalar("val/lat_err", val_lat, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:2d}/{num_epoch:2d}  "
                f"val_longitudinal={val_lon:.4f}  "
                f"val_lateral={val_lat:.4f}"
            )

    # --------------------------------------------------------------- save
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"✔  Model weights saved to {log_dir / f'{model_name}.th'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # Optional: extra hyper‑params for specific models
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)

    args = parser.parse_args()
    train(**vars(args))
