"""Model capabilities shared by VLM implementations and API schemas."""

from enum import StrEnum


class ModelCapability(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    IMAGE_BATCH = "image_batch"
