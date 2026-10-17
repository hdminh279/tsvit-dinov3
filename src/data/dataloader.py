"""
PyTorch Dataset and DataLoader for Sentinel-2 time series crop classification.
Loads synchronized patches [T, 10, 24, 24] and labels [24, 24].
"""

from typing import Callable, Dict, List, Optional, Tuple
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class S2CropPatchDataset(Dataset):
    """
    Loads preprocessed 24x24 Sentinel-2 multi-temporal patches and crop labels.
    """
    def __init__(
        self,
        data_dir: str,
        is_training: bool = False,
        transform: Optional[Callable] = None
    ):
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = transform

        # Look for .pt files first, fallback to .npz
        self.pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if self.pt_files:
            self.file_list = self.pt_files
            self.file_type = "pt"
        elif self.npz_files:
            self.file_list = self.npz_files
            self.file_type = "npz"
        else:
            self.file_list = []
            self.file_type = "none"

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_list[idx]

        if self.file_type == "pt":
            sample = torch.load(file_path, weights_only=False)
            image = sample['image']   # [T, 10, 24, 24]
            doy = sample['doy']       # [T]
            label = sample['label']   # [24, 24]
        else:
            data = np.load(file_path)
            image = torch.from_numpy(data['image']).float()
            doy = torch.from_numpy(data['doy']).float()
            label = torch.from_numpy(data['label']).long()

        # Spatial data augmentations for remote sensing (flips, rotations)
        if self.is_training:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-1])
                label = torch.flip(label, dims=[-1])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-2])
                label = torch.flip(label, dims=[-2])

        if self.transform:
            image, doy, label = self.transform(image, doy, label)

        return {
            'inputs': image,   # [T, 10, 24, 24]
            'doy': doy,        # [T]
            'label': label     # [24, 24]
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders.
    """
    full_dataset = S2CropPatchDataset(data_dir=data_dir, is_training=True)
    n_total = len(full_dataset)
    if n_total == 0:
        raise ValueError(f"No valid patches found in directory: {data_dir}")

    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
