from fastapi import APIRouter

from ovlm._internal.registry import list_local_models

router = APIRouter(prefix="/api")


@router.get("/model")
def get_models() -> list[str]:
    """Return model names whose checkpoints are available locally."""
    return list_local_models()
