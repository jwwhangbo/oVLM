"""Render piped inference JSON as a PNG."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

app = typer.Typer()
COLORS = [(255, 99, 71), (56, 189, 248), (163, 230, 53), (232, 121, 249),
          (251, 191, 36), (45, 212, 191)]


def render_result(result: dict) -> Image.Image:
    with Image.open(result["image"]) as source:
        canvas = source.convert("RGBA")
    masks, boxes, scores = result["masks"], result["boxes"], result["scores"]
    if not len(masks) == len(boxes) == len(scores):
        raise ValueError("Masks, boxes, and scores must have matching lengths")
    for index, rle in enumerate(masks):
        mask = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
        if mask.shape != (canvas.height, canvas.width):
            raise ValueError("Mask dimensions must match the source image")
        overlay = Image.new("RGBA", canvas.size, COLORS[index % len(COLORS)] + (0,))
        overlay.putalpha(Image.fromarray((mask > 0).astype(np.uint8) * 80))
        canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    width = max(2, min(canvas.size) // 400)
    font = ImageFont.load_default(size=max(12, min(canvas.size) // 65))
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = COLORS[index % len(COLORS)]
        draw.rectangle(box, outline=color, width=width)
        label = f"{index + 1}: {score:.2f}"
        x, y = max(0, box[0]), max(0, box[1])
        label_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((x, y), label, fill="black", font=font)
    return canvas.convert("RGB")


@app.command()
def visualize(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Save PNG directly instead of writing stdout.")] = None,
):
    """Read ovlm inference JSON from stdin and write a mask/box overlay PNG."""
    try:
        result = json.load(sys.stdin)
        image = render_result(result)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise typer.BadParameter(f"Cannot visualize inference result: {exc}") from exc
    if output is not None:
        image.save(output, format="PNG")
    else:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        image.save(sys.stdout.buffer, format="PNG")


if __name__ == "__main__":
    app()
