import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from ovlm._model._vlm import VLM
from ovlm._types.common import ImageInferenceRequest, ImageInferenceResponse


class SAM3(VLM):
    name = "sam3"

    def infer_image(self, request: ImageInferenceRequest) -> ImageInferenceResponse:
        if request.prompt is None:
            raise ValueError("SAM3 requires a prompt")
        if not request.prompt.texts or len(request.prompt.texts) != 1:
            raise ValueError("SAM3 requires exactly one text prompt")

        model = build_sam3_image_model()
        device = next(model.parameters()).device
        processor = Sam3Processor(model, device=str(device))
        # SAM3's fused MLP emits bfloat16; autocast keeps subsequent layers compatible.
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            inference_state = processor.set_image(request.image)
            output = processor.set_text_prompt(
                state=inference_state, prompt=request.prompt.texts[0]
            )
        return ImageInferenceResponse(
            masks=output["masks"],
            boxes=output["boxes"],
            scores=output["scores"],
        )
