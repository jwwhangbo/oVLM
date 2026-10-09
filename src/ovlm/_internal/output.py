"""JSON transport for CLI inference results."""

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from ovlm._types.common import ImageInferenceResponse


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def serialize_result(image: str, output: ImageInferenceResponse) -> str:
    masks = _array(output.masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError("Expected masks with shape (N, H, W) or (N, 1, H, W)")
    encoded = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
        encoded.append({"size": rle["size"], "counts": rle["counts"].decode("ascii")})
    return json.dumps({
        "image": str(Path(image).resolve()),
        "masks": encoded,
        "boxes": _array(output.boxes).tolist(),
        "scores": _array(output.scores).tolist(),
    }, separators=(",", ":"))
