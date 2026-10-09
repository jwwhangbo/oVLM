# oVLM

oVLM is a project conceived to make vision-language model (VLM) inference easier to run and serve, with batch processing as a core workflow.

Serving a VLM involves more than accepting an image alongside a prompt. Model-specific input formats, preprocessing, multi-image inputs, and batching can make a complete inference workflow difficult to put together. oVLM aims to bring model management, inference, and API serving into a straightforward CLI while preserving the capabilities of supported models.

> **Status:** Early development. The oVLM capabilities and commands below describe the intended interface, not implemented functionality.

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

**TODO:** Finalize the CLI interface and add working examples.

The planned CLI provides the following commands:

| Command | Intended behavior |
| --- | --- |
| `ovlm list` | List locally available models. |
| `ovlm pull <model>` | Download a model for local inference. |
| `ovlm run <model>` | Run inference with a selected model. |
| `ovlm serve` | Start the FastAPI inference server. |

Illustrative workflow (commands are not yet implemented):

```sh
ovlm pull <model>
ovlm list
ovlm run <model>
ovlm serve
```

**TODO:** Document image and prompt arguments, batch input and output formats, model options, server configuration, and API endpoints.
