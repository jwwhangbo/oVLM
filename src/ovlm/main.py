from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError
from ovlm._internal.registry import get_model
from ovlm._internal.output import serialize_result
from ovlm._types.common import ImageInferenceRequest, Prompt
from ovlm.api import app as fastapp
from PIL import Image

app = typer.Typer()

@app.command()
def serve(model: str):
    print(f"serving {model}")
    uvicorn.run(fastapp, host="127.0.0.1")

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
            ImageInferenceRequest(
                image=pil_image.convert("RGB"),
                prompt=parsed_prompt,
            )
        )
        print(serialize_result(image, output))


if __name__ == "__main__":
    app()
