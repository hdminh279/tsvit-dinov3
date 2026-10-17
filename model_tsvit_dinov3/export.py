"""
Model checkpoint saving and ONNX export utility.
"""

from typing import Optional, Tuple
import os
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[dict] = None
) -> str:
    """Saves full training checkpoint (.pt)."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics or {}
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(state, save_path)
    print(f"[Export] Saved PyTorch checkpoint to: {save_path}")
    return save_path


def export_onnx(
    model: nn.Module,
    export_path: str,
    num_timesteps: int = 30,
    img_res: int = 24,
    device: str = "cpu"
) -> str:
    """
    Exports model to ONNX format with dynamic batch dimension.
    """
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    model.eval()
    model.to(device)

    dummy_x = torch.randn(1, num_timesteps, 10, img_res, img_res, device=device)
    dummy_doy = torch.linspace(0.0, 1.0, num_timesteps, device=device).unsqueeze(0)

    dynamic_axes = {
        'imagery': {0: 'batch_size', 1: 'num_timesteps'},
        'doy': {0: 'batch_size', 1: 'num_timesteps'},
        'logits': {0: 'batch_size'}
    }

    try:
        torch.onnx.export(
            model,
            (dummy_x, dummy_doy),
            export_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['imagery', 'doy'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
        print(f"[Export] Successfully exported ONNX model to: {export_path}")
    except TypeError:
        # For older PyTorch without dynamo parameter
        torch.onnx.export(
            model,
            (dummy_x, dummy_doy),
            export_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['imagery', 'doy'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes
        )
        print(f"[Export] Successfully exported ONNX model to: {export_path}")
    return export_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu"
) -> dict:
    """Loads model weights from checkpoint (.pt)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"[Export] Loaded checkpoint from: {checkpoint_path}")
    return checkpoint

