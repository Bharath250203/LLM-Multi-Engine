# LLM Multi Engine

A research prototype for serving multiple LoRA adapters on a shared Llama language model. The code combines Python request handling with Punica model implementations and CUDA kernels for batched inference.

The central idea is to keep one base model in GPU memory and select task-specific LoRA weights for each request. Requests using different adapters can be grouped into the same inference batch.

> **Status:** Work in progress. Core model and kernel code is included, but parts of the application, packaging, and benchmark setup need integration work. An end-to-end build and inference run have not been verified for this repository snapshot.

## What is included

- Llama model implementations with support for batched LoRA weights.
- A `MultiLora` prototype that loads a shared base model, manages generation requests, and batches prefill and decoding work.
- Paged key/value cache utilities for tracking attention state during generation.
- CUDA/C++ operations for batched and segmented matrix-vector multiplication, attention integration, and RMS normalization.
- A terminal UI demo and fine-tuning examples for math question answering, SQL generation, and structured-data-to-text tasks.
- LoRA weight conversion utilities and benchmark scripts.

The current Python package is named **`punica`** in the build configuration. The higher-level application code lives separately under `LLM Multi Engine/`.

## How it works

1. Load a Llama base model and tokenizer.
2. Load task-specific LoRA adapters into the shared inference environment.
3. Tokenize incoming prompts and allocate a key/value cache for each request.
4. Group requests by generation phase and adapter, then run a batched model step.
5. Sample output tokens, update request state, and release cache resources when generation finishes.

See [`LLM Multi Engine/entrypoints/llm.py`](LLM%20Multi%20Engine/entrypoints/llm.py) for the `MultiLora`, `LoraSpec`, and `TextGeneration` implementations.

## Repository layout

| Path | Purpose |
| --- | --- |
| `LLM Multi Engine/entrypoints/llm.py` | Multi-adapter generation prototype |
| `src/punica/models/` | Llama and LoRA model implementations |
| `src/punica/utils/` | Cache management, batching, and adapter conversion |
| `src/punica/ops/` | Python interface to compiled CUDA operations |
| `csrc/` | CUDA/C++ kernel source |
| `test/` | Demo programs and fine-tuning examples |
| `bench/` | Inference and kernel benchmark scripts |
| `third_party/` | Bundled CUTLASS, FlashInfer, and NVBench source |
| `licenses/` | Third-party license notices |
| `setup.py`, `pyproject.toml` | Python package and CUDA extension build configuration |

## Requirements

- Python **3.10 or newer**, as declared in `pyproject.toml`.
- An NVIDIA GPU, a compatible driver, and the CUDA toolkit, including `nvcc`.
- A CUDA-enabled PyTorch installation compatible with the local toolkit.
- A C++ build toolchain, setuptools, NumPy, Ninja, and wheel.
- Transformers, plus the optional development dependencies for demos and benchmarks.
- Base-model and adapter weights obtained separately. The examples reference `meta-llama/Llama-2-7b-hf`; access to that model may require approval from its publisher.

The current generation prototype uses `cuda:0` and FP16. CPU and Apple Metal/MPS execution paths are not implemented. GPU memory requirements depend on the base model, adapter count, batch size, and sequence lengths. No tested dependency-version matrix is provided yet.

## Development setup

These steps describe the intended build path on a machine with NVIDIA CUDA support. They are a starting point for development, not a verified installation recipe for the current snapshot.

```bash
git clone https://github.com/Bharath250203/LLM-Multi-Engine.git
cd LLM-Multi-Engine

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja numpy
```

Install a CUDA-enabled version of PyTorch appropriate for your GPU and CUDA toolkit, then check the environment:

```bash
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

With PyTorch and the CUDA toolkit available, attempt the extension build and editable installation:

```bash
python -m pip install --no-build-isolation -e '.[dev]'
```

The build generates the CUDA dispatch sources and `src/punica/_build_meta.py`. After a successful build, a basic import check is:

```bash
python -c "import punica; print(punica.__version__)"
```

An import check alone does not validate inference correctness or performance.

## Models, adapters, and examples

Model weights are not bundled with this repository. The examples use LoRA adapters for GSM8K, SQL generation, and ViGGO. See [`test/finetune/README.md`](test/finetune/README.md) for the existing training and conversion notes; some paths in those notes refer to an older repository layout and must be corrected before use.

Once the package is built, the included converter accepts a compatible PEFT PyTorch adapter file and writes Punica-format tensors:

```bash
python -m punica.utils.convert_lora_weight \
  model/adapter_model.bin \
  model/adapter.punica.pt
```

The converter expects the tensor names and layout used by its implementation; it does not directly load `.safetensors` files.

[`test/vlora-tui.py`](test/vlora-tui.py) contains the terminal UI demonstration. [`test/vlora.py`](test/vlora.py) illustrates how the higher-level generation interface is intended to be used, but currently needs the fixes described below.

## Current limitations

- **Application entry points:** `LLM Multi Engine/entrypoints/api_server.py` and `LLM Multi Engine/engine/vlora_engine.py` are empty placeholders. An HTTP serving interface is not implemented.
- **Application integration:** `test/vlora.py` imports a `vlora` package that is not present, omits a `pathlib` import, and calls `MultiLora` without its required `dtype` argument. The generation implementation also references an undefined `append_box` callback.
- **Configuration paths:** The pytest and Pyright settings refer to `tests/` and `examples/`, while this snapshot uses `test/`. The top-level CMake configuration refers to a missing `benchmarks/nvbench/` directory.
- **Benchmark dependencies:** Scripts in `bench/` reference helper modules such as `benchmark_utils` and `bench_textgen` that are absent from this snapshot.
- **Validation:** Dependency versions are unpinned, and no verified performance results or passing end-to-end test results are claimed here.

## Development priorities

1. Align package names, imports, and configuration paths with the actual source tree.
2. Connect the generation code to a defined output callback and complete an application entry point.
3. Restore the missing benchmark helpers and add meaningful inference tests.
4. Validate a reproducible CUDA/PyTorch environment and document measured results.

## License and acknowledgments

This repository includes an [Apache License 2.0](LICENSE). It incorporates Punica code and bundles CUTLASS, FlashInfer, and NVBench components. Consult the notices in [`licenses/`](licenses/) and the individual third-party directories for their applicable license terms and attribution. Model and dataset licenses apply separately.
