from abc import ABC, abstractmethod
from ovlm._types.common import ImageInferenceRequest, ImageInferenceResponse

class VLM(ABC):
    name: str

    @abstractmethod
    def infer_image(self, request: ImageInferenceRequest) -> ImageInferenceResponse:
        pass