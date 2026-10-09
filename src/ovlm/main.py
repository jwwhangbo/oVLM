import typer
import uvicorn
from ovlm.api import app as fastapp

app = typer.Typer()

@app.command()
def hello(name: str):
    print(f"Hello {name}")

@app.command()
def serve(model: str):
    print(f"serving {model}")
    uvicorn.run(fastapp, host="127.0.0.1")

if __name__ == "__main__":
    app()