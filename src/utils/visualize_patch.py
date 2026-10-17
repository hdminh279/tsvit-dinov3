"""
Quick script to visualize a preprocessed .pt patch as a true-color (RGB) image.
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image

from src.data.constants import S2_MEAN_10B, S2_STD_10B


def visualize_patch(pt_file: str, output_png: str = "patch_rgb.png", t_idx: int = 0):
    if not os.path.exists(pt_file):
        print(f"Error: File {pt_file} not found.")
        return

    sample = torch.load(pt_file, weights_only=False)
    # Channels: 2=B04(Red), 1=B03(Green), 0=B02(Blue)
    rgb = sample['image'][t_idx, [2, 1, 0]].numpy()

    # Un-normalize
    mean = S2_MEAN_10B[[2, 1, 0], None, None]
    std = S2_STD_10B[[2, 1, 0], None, None]
    raw_rgb = rgb * std + mean

    # Contrast stretch for human eye: scale [0, 3000] to [0, 255]
    scaled = np.clip(raw_rgb / 3000.0 * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(scaled.transpose(1, 2, 0))
    # Resize 10x for easy viewing (24x24 -> 240x240)
    img_large = img.resize((240, 240), Image.NEAREST)
    img_large.save(output_png)
    print(f"[Visualize] Successfully saved bright RGB image to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_file", type=str, help="Path to .pt patch file")
    parser.add_argument("--output", type=str, default="patch_rgb.png", help="Output PNG path")
    parser.add_argument("--t", type=int, default=0, help="Timestamp index (default: 0)")
    args = parser.parse_args()
    visualize_patch(args.pt_file, args.output, args.t)
