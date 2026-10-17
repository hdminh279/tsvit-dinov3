"""
Command-line training script for TSViT-DINOv3.
Loads configuration, trains on multi-temporal Sentinel-2 patches, and exports checkpoints.
"""

import os
import sys
import argparse
import yaml
import glob
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.tsvit_dinov3 import TSViT_DINOv3
from src.data.dataset_builder import S2CropDatasetBuilder
from src.data.dataloader import get_dataloaders
from src.data.constants import NUM_CLASSES
from src.utils.metrics import compute_segmentation_metrics
from src.utils.export import save_checkpoint, export_onnx


def main():
    parser = argparse.ArgumentParser(description="Train TSViT-DINOv3 on Sentinel-2 Time-Series")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to preprocessed patches")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    m_cfg = cfg.get("MODEL", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Initializing TSViT-DINOv3 on device: {device}")

    # Ensure dataset exists
    os.makedirs(args.data_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(args.data_dir, "*.pt")) + glob.glob(os.path.join(args.data_dir, "*.npz"))
    if not existing_files:
        raise FileNotFoundError(
            f"[Train] Không tìm thấy tệp dataset (.pt/.npz) nào trong thư mục '{args.data_dir}'. "
            f"Vui lòng thực hiện bước thu thập dữ liệu (Cách 2 / Cách 3 / Cách 4) trước khi huấn luyện!"
        )


    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2, val_split=0.2)
    print(f"[Train] Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # Initialize model
    model = TSViT_DINOv3(
        img_res=m_cfg.get("img_res", 24),
        patch_size=m_cfg.get("patch_size", 2),
        in_channels=m_cfg.get("in_channels", 10),
        num_classes=m_cfg.get("num_classes", NUM_CLASSES),
        dim=m_cfg.get("dim", 192),
        temporal_depth=m_cfg.get("temporal_depth", 4),
        spatial_depth=m_cfg.get("spatial_depth", 4),
        heads=m_cfg.get("heads", 4),
        dim_head=m_cfg.get("dim_head", 48),
        dropout=m_cfg.get("dropout", 0.1)
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0
    best_pt = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['inputs'].to(device)
            doy = batch['doy'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(x, doy)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['inputs'].to(device)
                doy = batch['doy'].to(device)
                y = batch['label'].to(device)
                logits = model(x, doy)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                all_preds.append(torch.argmax(logits, dim=1).cpu())
                all_targets.append(y.cpu())

        val_loss /= len(val_loader.dataset)
        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        metrics = compute_segmentation_metrics(preds_cat, targets_cat, num_classes=NUM_CLASSES, ignore_index=0)

        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | OA: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

        if metrics['macro_f1'] >= best_f1:
            best_f1 = metrics['macro_f1']
            save_checkpoint(model, best_pt, optimizer, epoch, metrics)

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, "tsvit_dinov3_crop.onnx")
    export_onnx(model.cpu(), onnx_path, num_timesteps=20, img_res=24)
    print(f"[Train] Training complete. Best checkpoint: {best_pt}, ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
