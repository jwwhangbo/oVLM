from abc import ABC
from typing import Any, ClassVar

from ovlm._types.capabilities import ModelCapability
from ovlm._types.image import ImageInput, ImageOutput

class VLM(ABC):
    name: str
    capabilities: ClassVar[frozenset[ModelCapability]] = frozenset()

    def infer_image(self, request: ImageInput) -> ImageOutput:
        """Infer one image; override in models that support image inference."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image inference."
        )

    def infer_video(self, request: Any) -> Any:
        """Infer video; override in models that support video inference."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support video inference."
        )

    def infer_image_batch(self, requests: list[ImageInput]) -> list[ImageOutput]:
        """Infer an image batch; override in models that support batch inference."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image batch inference."
        )
