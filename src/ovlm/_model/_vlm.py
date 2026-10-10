from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ovlm._types.capabilities import ModelCapability
from ovlm._types.image import ImageInput, ImageOutput

class VLM(ABC):
    name: str
    capabilities: ClassVar[frozenset[ModelCapability]] = frozenset()
    weights_path: str | None

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

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

    @abstractmethod
    def pull(self) -> str:
        """Download model weights and return their local path."""
        pass

    @abstractmethod
    def find_cached_weights(self) -> str | None:
        """Return a cached checkpoint path without downloading or building a model."""
        pass

    @abstractmethod
    def rm(self) -> str | None:
        """Delete the local checkpoint path, returning it or None if absent."""
        pass
