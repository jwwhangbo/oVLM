from functools import cache
from importlib import import_module
from inspect import getmembers, isabstract, isclass
from pathlib import Path
from pkgutil import iter_modules

from ovlm import vlms
from ovlm._model._vlm import VLM


@cache
def _discover_models() -> dict[str, type[VLM]]:
    """Discover concrete VLM classes by name once per process."""
    models: dict[str, type[VLM]] = {}

    for info in iter_modules(vlms.__path__, prefix=f"{vlms.__name__}."):
        module = import_module(info.name)

        for _, cls in getmembers(module, isclass):
            if (
                cls.__module__ != module.__name__
                or not issubclass(cls, VLM)
                or isabstract(cls)
            ):
                continue

            if cls.name in models:
                raise ValueError(f"Duplicate VLM name: {cls.name!r}")

            models[cls.name] = cls

    return models


def get_model(name: str) -> VLM:
    """Return a fresh instance of the VLM registered under name."""
    models = _discover_models()
    if name not in models:
        raise ValueError(
            f"Unknown VLM {name!r}. Available: {', '.join(sorted(models))}"
        )
    return models[name]()


def list_local_models() -> list[str]:
    """List supported models with local checkpoints, checking availability each call."""
    available = []
    for name, model_cls in _discover_models().items():
        path = model_cls().find_cached_weights()
        if path is not None and Path(path).is_file():
            available.append(name)
    return sorted(available)
