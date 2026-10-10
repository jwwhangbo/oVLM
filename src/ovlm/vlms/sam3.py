import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

from ovlm._model._vlm import VLM
from ovlm._types.capabilities import ModelCapability
from ovlm._types.image import ImageInput, ImageOutput


class SAM3(VLM):
    name = "sam3"
    capabilities = frozenset({
        ModelCapability.IMAGE,
        ModelCapability.VIDEO,
        ModelCapability.IMAGE_BATCH,
    })

    def pull(self) -> str:
        """Fetch SAM3 assets into the Hugging Face cache without building the model."""
        self.weights_path = download_ckpt_from_hf(version="sam3")
        return self.weights_path

    def find_cached_weights(self) -> str | None:
        """Look up SAM3's checkpoint in the local Hugging Face cache."""
        try:
            return hf_hub_download(
                repo_id="facebook/sam3", filename="sam3.pt", local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return None

    def rm(self) -> str | None:
        """Unlink the local checkpoint without downloading or following symlinks."""
        weights_path = self.weights_path
        if weights_path is None:
            weights_path = self.find_cached_weights()
        if weights_path is None:
            return None

        try:
            Path(weights_path).unlink()
        except FileNotFoundError:
            self.weights_path = None
            return None

        self.weights_path = None
        return weights_path

    def _get_weights_path(self) -> str:
        if self.weights_path is not None and Path(self.weights_path).is_file():
            return self.weights_path

        cached_path = self.find_cached_weights()

        if cached_path is not None and Path(cached_path).is_file():
            self.weights_path = cached_path
            return cached_path

        print("No local weights are available for sam3. Downloading them now...", file=sys.stderr)
        return self.pull()

    def infer_image(self, request: ImageInput) -> ImageOutput:
        if request.prompt is None:
            raise ValueError("SAM3 requires a prompt")
        if not request.prompt.texts or len(request.prompt.texts) != 1:
            raise ValueError("SAM3 requires exactly one text prompt")

        model = build_sam3_image_model(
            checkpoint_path=self._get_weights_path(), load_from_HF=False,
        )
        device = next(model.parameters()).device
        processor = Sam3Processor(model, device=str(device))
        # SAM3's fused MLP emits bfloat16; autocast keeps subsequent layers compatible.
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            inference_state = processor.set_image(request.image)
            output = processor.set_text_prompt(
                state=inference_state, prompt=request.prompt.texts[0]
            )
        return ImageOutput(
            masks=output["masks"],
            boxes=output["boxes"],
            scores=output["scores"],
        )
