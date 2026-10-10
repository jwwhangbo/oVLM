from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError
from ovlm._internal.registry import get_model
from ovlm._internal.output import serialize_result
from ovlm._types.image import ImageInput, Prompt
from ovlm.api import app as fastapp
from PIL import Image

app = typer.Typer()

@app.command()
def serve(daemon: Annotated[bool, typer.Option(help="runs the api as a daemon process")]):
    uvicorn.run(fastapp, host="127.0.0.1")

@app.command()
def down():
    pass

@app.command()
def pull(model: Annotated[str, typer.Argument(help="name of model to download")]):
    """Download model weights for local inference."""
    try:
        vlm = get_model(model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="model") from exc

    try:
        weights_path = vlm.pull()
    except OSError as exc:
        typer.echo(f"Cannot pull {model}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Weights for {model} are available at {weights_path}")

@app.command()
def rm(model: Annotated[str, typer.Argument(help="name of model whose local checkpoint to remove")]):
    """Delete a model's local checkpoint path."""
    try:
        vlm = get_model(model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="model") from exc

    try:
        weights_path = vlm.rm()
    except OSError as exc:
        typer.echo(f"Cannot remove {model}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if weights_path is None:
        typer.echo(f"No local weights are available for {model}.")
    else:
        typer.echo(f"Removed checkpoint for {model}: {weights_path}")

@app.command()
def run(
        model: Annotated[str, typer.Argument(help="name of model to run inference")],
        image: Annotated[str, typer.Option(help="location of the image to run inference on")],
        prompt: Annotated[str, typer.Option(help="JSON object with Prompt fields to use for inference")]
    ):
    try:
        parsed_prompt = Prompt.model_validate_json(prompt)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--prompt") from exc

    vlm = get_model(model)
    with Image.open(image) as pil_image:
        output = vlm.infer_image(
            ImageInput(
                image=pil_image.convert("RGB"),
                prompt=parsed_prompt,
            )
        )
        print(serialize_result(image, output))


if __name__ == "__main__":
    app()
