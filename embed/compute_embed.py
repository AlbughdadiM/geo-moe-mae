"""
A script to generate embeddings of a model from
aniterable dataloader.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

from typing import Dict, Any, Iterable, Tuple, Union
import torch
from tqdm import tqdm


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    return obj


def _extract_batch(batch: Union[Tuple, Dict[str, Any]], device: torch.device):
    """
    Returns (x, y, meta_kwargs) where meta_kwargs contains only the metadata
    that exists in the batch (any of meta_week/meta_hour/meta_lat/meta_lon).
    """
    meta_keys = ("meta_week", "meta_hour", "meta_lat", "meta_lon")
    meta_kwargs: Dict[str, torch.Tensor] = {}

    if isinstance(batch, dict):
        x = _to_device(batch["x"], device)
        y = _to_device(batch["y"], device)
        for k in meta_keys:
            if k in batch and batch[k] is not None:
                meta_kwargs[k] = _to_device(batch[k], device)
        return x, y, meta_kwargs

    # Tuple style
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Batch must be a dict or a tuple/list.")

    if len(batch) < 2:
        raise ValueError("Batch must at least contain (x, y).")

    x = _to_device(batch[0], device)
    y = _to_device(batch[1], device)

    # If metadata is present, accept up to 6 items total
    # (x, y, meta_week, meta_hour, meta_lat, meta_lon)
    names = ("meta_week", "meta_hour", "meta_lat", "meta_lon")
    for idx in range(2, min(len(batch), 6)):
        val = batch[idx]
        if val is not None:
            meta_kwargs[names[idx - 2]] = _to_device(val, device)

    return x, y, meta_kwargs


@torch.inference_mode()
def compute_geomoemae_embeddings(
    model,
    dataloader: Iterable,
    device: Union[str, torch.device] = "cuda",
    use_autocast: bool = True,
):
    """
    Works with or without metadata. If the dataloader supplies any of:
    meta_week/meta_hour/meta_lat/meta_lon, they are passed to the model via **kwargs.
    Otherwise the model is called with just (x).

    Supported batch formats:
      - dict: {"x": Tensor, "y": Tensor, "meta_week": Tensor, ...}
      - tuple: (x, y[, meta_week, meta_hour, meta_lat, meta_lon])
    """
    device = torch.device(device)
    model.eval().to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_list, y_list = [], []

    # choose autocast device_type dynamically
    amp_device_type = (
        "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    )

    for batch in tqdm(
        dataloader,
        total=getattr(dataloader, "__len__", lambda: None)(),
        desc="Computing embeddings",
    ):
        x, y, meta_kwargs = _extract_batch(batch, device)

        if use_autocast:
            with torch.autocast(device_type=amp_device_type, enabled=True):
                # Try calling with metadata; if the model doesn't accept them, call with x only.
                try:
                    out = model(x, **meta_kwargs)
                except TypeError:
                    out = model(x)
        else:
            try:
                out = model(x, **meta_kwargs)
            except TypeError:
                out = model(x)

        # Accept either (embeddings, logits) or just embeddings
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            embeddings = out[-1]
        else:
            embeddings = out

        x_list.append(embeddings.float().cpu())
        y_list.append(y.cpu())

    x_all = torch.cat(x_list, dim=0).numpy()
    y_all = torch.cat(y_list, dim=0).numpy()
    return x_all, y_all
