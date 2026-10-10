# oVLM

oVLM is a project conceived to make vision-language model (VLM) inference easier to run and serve, with batch processing as a core workflow.

Serving a VLM involves more than accepting an image alongside a prompt. Model-specific input formats, preprocessing, multi-image inputs, and batching can make a complete inference workflow difficult to put together. oVLM aims to bring model management, inference, and API serving into a straightforward CLI while preserving the capabilities of supported models.

> **Status:** Early development. SAM3 checkpoint pulling and removal, image inference, and piped visualization are implemented; the other capabilities below describe the intended interface.

## A. Comparison with Ollama and vLLM

| Area | oVLM (planned) | Ollama | vLLM |
| --- | --- | --- | --- |
| Project focus | Simple VLM inference and batch workflows | Running and serving language and vision models | Inference and serving for language and multimodal models |
| Vision input | A consistent interface for supported VLMs | Images alongside text for supported vision models | Multimodal inputs for supported models |
| Batch processing | An explicit workflow for processing multiple independent inputs | Chat and generation APIs use individual requests; applications orchestrate bulk jobs | Offline VLM batch inference is documented |
| Multiple images per prompt | Planned where the underlying model supports it | Image arrays; capabilities depend on the model | Supported by compatible models with per-prompt limits configured |
| Model management | `list` and `pull` commands | CLI model management | Model selection through engine and serving configuration |
| API serving | `serve` starts a FastAPI server | HTTP API | OpenAI-compatible server |

Batch processing means running multiple independent inference inputs. Multi-image input means supplying several images to a single prompt. These are separate capabilities, and support depends on the model and runtime configuration.

Ollama and vLLM already support vision workloads. oVLM's goal is to make the end-to-end VLM workflow easier to use, with explicit batch processing and clear support for model-specific features.

Comparison references: [Ollama vision documentation](https://docs.ollama.com/capabilities/vision), [Ollama API documentation](https://docs.ollama.com/api), and [vLLM multimodal input documentation](https://docs.vllm.ai/en/stable/features/multimodal_inputs/). Capabilities may vary by version and model.

## B. Installation

**TODO:** Add installation instructions, supported platforms, GPU requirements, and dependency setup.

The current project metadata requires Python 3.12. Package installation and CLI setup will be documented once available.

## C. How to use

### Download SAM3 weights (implemented)

```sh
uv run ovlm pull sam3
```

This downloads SAM3's checkpoint and configuration into the Hugging Face cache
without building the model. Existing cached downloads are reused. SAM3 requires
access to the gated `facebook/sam3` repository; authenticate with `hf auth login`
after obtaining access.

Inference uses local weights when available. If they are missing, it prints a
notice to stderr, downloads them, and continues inference. You can also provide
a checkpoint in Python with `SAM3(weights_path="path/to/sam3.pt")`.

### Remove SAM3's local checkpoint (implemented)

```sh
uv run ovlm rm sam3
```

This deletes the local checkpoint path without downloading anything. If no local
checkpoint exists, it reports that and exits successfully. The next inference
call pulls the weights again automatically. In Python, `SAM3(weights_path="path/to/sam3.pt").rm()`
removes that specific checkpoint.

Removal leaves configuration files and other cached revisions in place. If the
checkpoint path is a symlink, only the link is removed; shared backing files
remain in the Hugging Face cache and may be reused on the next pull.

### Image inference and visualization (implemented)

`ovlm run` writes JSON to stdout containing the absolute source image path,
pixel-coordinate boxes, scores, and masks compressed as COCO RLE. The visualization
utility reads that JSON and writes a PNG with colored masks, boxes, and scores.
The source image must remain accessible to the renderer.

```powershell
# PowerShell 7.4+ preserves binary PNG data when using >.
$PSNativeCommandArgumentPassing = 'Legacy'
uv run ovlm run sam3 `
  --image 'C:\Users\jonny\Downloads\IMG_2094.heic_compressed.JPEG' `
  --prompt '{\"texts\": [\"flamingo\"]}' |
  uv run tools.visualize > output.png
```

On older PowerShell versions, replace the last line with
`uv run tools.visualize --output output.png` to save the PNG directly.
SAM3 currently accepts exactly one text phrase in `texts`.

**TODO:** Finalize the CLI interface and add working examples.

The planned CLI provides the following commands:

| Command | Intended behavior |
| --- | --- |
| `ovlm list` | List locally available models. |
| `ovlm pull <model>` | Download a model for local inference. |
| `ovlm rm <model>` | Remove the local checkpoint path. |
| `ovlm run <model>` | Run inference with a selected model. |
| `ovlm serve` | Start the FastAPI inference server. |

Illustrative workflow (`list` and inference API serving are still planned):

```sh
ovlm pull <model>
ovlm list
ovlm run <model>
ovlm serve
```

**TODO:** Document image and prompt arguments, batch input and output formats, model options, server configuration, and API endpoints.
