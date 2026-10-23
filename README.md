# pixtreme.infer

This repository owns the pure-Python distribution `pixtreme-infer` and its import package
`pixtreme.infer`. It provides the architecture-independent Frame boundary, shared inference
contracts, ONNX verification, recipe identity, and static dispatch used by architecture
distributions.

The wheel owns only the `pixtreme/infer/` subtree. The namespace root
`pixtreme/__init__.py` remains owned by
[pixtreme](https://github.com/sync-dev-org/pixtreme). Architecture implementations are
maintained and distributed separately as architecture distributions.

## Capabilities

The base dependency is `pixtreme>=1.3.0`. Optional capabilities are separated by extra:

- `ort`: ONNX Runtime verification.
- `verify`: device telemetry for the shared verification tools.

The core does not depend on TensorRT or torch. Architecture distributions build and run
TensorRT engines through tensoracle and declare TensorRT and tensoracle as their own
dependencies.

Core-only installation exposes the shared contracts and gives an actionable error when a
known architecture distribution is absent. The core does not eagerly import architecture
implementations.

## Development

Run development commands from the repository root:

```console
uv sync --all-extras
uv lock --check
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q
uv run --no-sync mypy
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build --wheel
```

The normal test suite does not require a GPU. Tests that require unavailable optional
dependencies skip with a reason.

## Version and release

Versions come from tags named `vX.Y.Z`. Each version is published to PyPI as a
`pixtreme-infer` wheel; there is no source distribution, GitHub Release, or Release asset.

## CUDA Graph components for module authors

`pixtreme.infer.cuda_graph` exports `CudaGraphSession` and `CudaGraphBuffers`.
The core captures backend operations that enqueue work using fixed GPU storage.
A module still validates every request, selects an operation, handles its temporal
history and reset, and builds output metadata on every call. The core does not
capture the module's public Frame call or import an inference backend.

A module can implement the optional
`CudaGraphFactoryProvider[StatelessInferenceFactory[ArtifactIdentity]]` or
`CudaGraphFactoryProvider[TemporalInferenceFactory[ArtifactIdentity]]` protocol
from `pixtreme.infer.contracts`. Its `graph_factory()` returns a separate factory
with the original inference form. Factory selection and `query_contract` allocate
no GPU memory. `open` repeats configuration validation and prepares all graphs
before returning an instance whose contract equals the query result. The normal
factory keeps its existing operations and accepted settings; there is no new
required setting on existing factories.

Open a graph instance inside its device and an explicit non-default CUDA stream.
One instance handles one fixed input configuration on that device and stream,
with sequential host calls. Modules can prepare several named operations, with
shared fixed storage and separate backend execution units where their backend
requires them. The following fragment shows the integration inside a module;
the backend callables, fixed Frames, GPU parameters, history, and execution units
are prepared and owned by that module:

```python
from pixtreme.infer.cuda_graph import CudaGraphBuffers, CudaGraphSession

buffers = CudaGraphBuffers(
    fixed_input_frame, fixed_output_frame, params=fixed_gpu_params,
)
graphs = CudaGraphSession(
    {"route-a": enqueue_a, "route-b": enqueue_b},
    resources=(buffers, *history, *execution_units),
    after_prepare=restore_initial_history_and_controls,
)
```

Each callable must enqueue on the current stream, use only fixed GPU addresses,
and be safe for CUDA stream capture. Preparation calls each callable once for
warm-up, waits for that stream, and captures it once. Keep every buffer and
backend workspace used inside capture alive for the whole session, using the
callable's owners or `resources`. Allocating an unretained temporary inside
capture is invalid. Preparation can write history; `after_prepare` restores both
GPU history and host controls to their open state without consuming requests.
If any operation or this callback fails, preparation releases all captured graphs
and retained references and propagates the error. The module's `open` must also
release its own partially prepared resources. No partial instance or automatic
normal-path fallback is returned.

For each request, the module first checks its Frame metadata, reset control, and
parameter contract. It calls `graphs.check_context()` **before** copying inputs
or resetting history. Then it can perform these steps:

```python
# Module request validation and graph selection have already completed.
graphs.check_context()
buffers.copy_inputs(frame, **gpu_params)
if reset:
    reset_history_on_current_stream()
graphs.replay(selected_name)
result = buffers.copy_output(
    channels=output_channels,
    colorspace=output_colorspace,
    gamma=output_gamma,
    matrix=output_matrix,
)
```

`copy_inputs` validates every source before the first copy and requires exact
shape, dtype, device, and auxiliary names. It imports Frame storage through
DLPack and accepts CUDA auxiliary arrays, including CAI v3 and CUDA DLPack
producers. Sources must already be ready on the current stream and remain alive
until its copies complete. Copying performs no casting, broadcasting, or host
transfer, and the graph retains no caller input address. Modules validate reset
before calling this method, so a structurally rejected request does not clear
history or modify fixed storage. Auxiliary inputs are distinct from module-owned
temporal history; reset and graph selection remain module responsibilities.

`copy_output` allocates independent GPU storage outside capture and uses the
metadata supplied for that request. Returned Frames remain unchanged by later
replays, resets, and close. Requests enqueue without a host wait. Callers use
CUDA events for another stream to consume results, and finish outstanding work
before the module closes `graphs`, `buffers`, and its backend resources. Both
component `close()` methods are idempotent and perform no synchronization; later
operations on closed components fail before GPU work. Changing settings, device,
or stream requires opening another instance; there is no automatic recapture or
graph cache.

For common verification, supply `CudaGraphContractSuite(module, replay_equivalence)`
to `verify_cuda_graph_contract` in `pixtreme.infer.verification`. `module` is the
existing `StatelessContractSuite` or `TemporalContractSuite`, populated from the
graph factory and its instances. This runs `FactoryContractCases` unchanged and,
for temporal modules, the existing reset, startup, instance isolation, and
persistent state footprint checks. `replay_equivalence` supplies `SequenceFacts`
from separate normal and graph instances with the same settings, including the
first call, continuation, changed input values and metadata, and resets. Missing
evidence stays unverified. Modules retain ownership of their fixtures, correctness
thresholds, support declarations, and evidence. Performance measurements do not
determine support status.

The opt-in synthetic GPU tests use no TensorRT or tensoracle. Run correctness with
`uv run --no-sync python -m pytest -m 'gpu and not performance' tests/test_cuda_graph_gpu.py`,
selecting the GPU with `CUDA_VISIBLE_DEVICES` when the machine has several.
The `performance` cases record video-sized copy latency with CUDA events and print
median and p90 values; they apply no speed threshold.

## License

The core distribution `pixtreme-infer` is licensed under the MIT License; see
[LICENSE](LICENSE). Architecture distributions are separate distributions, each with its own
license. The core license does not apply to them, and installing one does not change the core
license.
