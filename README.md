# pixtreme.infer

This repository owns the distribution `pixtreme-infer` and its import package
`pixtreme.infer`. It provides the architecture-independent Frame boundary, shared inference
contracts, shared verification, recipe identity, and static dispatch used by architecture
distributions.

The wheel owns only the `pixtreme/infer/` subtree. The namespace root
`pixtreme/__init__.py` remains owned by
[pixtreme](https://github.com/sync-dev-org/pixtreme). Architecture implementations are
maintained and distributed separately as architecture distributions.

## Capabilities

The base dependency is `pixtreme>=2.4.0`. Optional capabilities are separated by extra:

- `verify`: device telemetry for the shared verification tools.

The core does not depend on TensorRT or torch. Architecture distributions build and run
TensorRT engines through tensoracle and declare TensorRT and tensoracle as their own
dependencies.

Core-only installation exposes the shared contracts and gives an actionable error when a
known architecture distribution is absent. The core does not eagerly import architecture
implementations.

## Development

The standalone C++ SDK is in `cpp/`; the Python project and its environment are in
`python/` and `python/.venv`. Run Python development commands from `python/`:

```console
cd python
uv sync --all-extras
uv lock --check
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q
uv run --no-sync mypy
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build --wheel
```

The wheel includes the nanobind extension, C interface, header-only C++ wrapper and
nanobind adapter, CMake package and shared
library under `pixtreme/infer/`. Module builds find the CMake package with
`pixtreme.infer.get_cmake_dir()` and link `pixtreme_infer::pixtreme_infer`. The SDK
uses the shared library supplied by the installed `pixtreme` distribution. The wheel
contains one shared library, `libpixtreme_infer.so.1`. Importing
`pixtreme.infer` loads its extension and shared library without initializing CUDA.

A native consumer declares the SDK dependency in its `CMakeLists.txt`:

```cmake
find_package(pixtreme_infer CONFIG REQUIRED)
target_link_libraries(my_module PRIVATE pixtreme_infer::pixtreme_infer)
```

Configure the consumer from the Python environment containing the installed wheels:

```console
cmake -S . -B build -Dpixtreme_infer_DIR="$(python -c 'import pixtreme.infer; print(pixtreme.infer.get_cmake_dir())')"
cmake --build build
```

The wheel's config finds pixtreme in the same environment. An explicit
`-Dpixtreme_DIR=...` takes priority when selecting a different pixtreme SDK.

The normal test suite does not require a GPU. Tests that require unavailable optional
dependencies skip with a reason.

## Version and release

Versions come from tags named `vX.Y.Z`. Each version is published to PyPI as a
`pixtreme-infer` wheel; there is no source distribution. Install it from PyPI. The GitHub
Release of each version tag holds the same wheel and a `SHA256SUMS` file for verification.

## Native inference lifecycle

Architecture implementations derive from `pixtreme::infer::StatelessInference` or
`TemporalInference` in `<pixtreme/infer/core.hpp>`. A constructor inspects the artifact
and publishes its contract without initializing CUDA or allocating GPU memory.
`prepare(device=0, stream=0)` rechecks that contract and acquires backend resources.
Inference accepts one `pixtreme.core.Frame`, the declared CUDA parameters, an optional
`reset` for temporal inference, and an explicit `stream` handle. Omitting the stream
uses the legacy default stream. Calls before preparation, after a failed preparation,
and after `close()` fail before backend work. `close()` is idempotent.

A Python context manager prepares a newly constructed model with default arguments
and closes it on exit. For an explicit stream, prepare before entering the context:

```python
import pixtreme as px

stream = px.core.Stream(device=0, non_blocking=True)
stream_handle = stream.ptr
model = ModuleModel(artifact, precision="fp16", input_hw=(height, width), use_cuda_graph=True)
model.prepare(device=0, stream=stream_handle)
with model:
    result = model(frame, stream=stream_handle, **gpu_parameters)
    stream.synchronize()
```

`ModuleModel` is the class supplied by an architecture distribution. Its constructor
owns its artifact settings; core has no query/open factory. `model.contract` is an
immutable copy of the C instance contract, including precision, fixed input shape,
scale, CUDA Graph selection, and temporal state bytes where applicable.
`model.family_contract` exposes the same native family's Frame and parameter declarations.
Python contract construction uses the native C validation as well.

The native class interface requires architecture modules built for core 0.18 and
pixtreme 2.4. Older modules exposing query/open factories are incompatible. The core's
architecture table includes `spanv2` and `cda_vsr`; it does not recommend the older
`span` distribution.

## Binding an architecture module

The public `<pixtreme/infer/nanobind.h>` adapter binds a consumer-side class derived
from `pixtreme::infer::Model`. That consumer owns the C model handle. Construct the
implementation with `make_model_handle<Implementation>(...)`, which transfers ownership
to core; never bind the same implementation object as a second owning Python object.
For example, after defining the architecture's consumer class and constructor:

```cpp
#include <pixtreme/infer/nanobind.h>

NB_MODULE(_native, module) {
    nanobind::class_<ModuleModel> model(module, "Model");
    model.def(nanobind::init<ModuleSettings>());
    pixtreme::infer::python::bind_model(model);
}
```

The adapter uses only the C interface and public Python contract value types. Module
and core extensions may use different nanobind domains; Python inheritance from a
core class is not required. Frame arguments and owning results use pixtreme's public
`<pixtreme/nanobind.h>` type casters. Module constructors can use the adapter's
`pixtreme::infer::python::call` helper to apply the same error mapping: invalid arguments
become `ValueError`, invalid lifecycle states and backend failures become `RuntimeError`,
and allocation failures become `MemoryError`.

## Native CUDA Graphs and verification

The constructor's `use_cuda_graph` selects core's C++ graph implementation. Graph
preparation requires an explicit non-default stream. The module supplies fixed GPU
buffers, allocation-free backend enqueue hooks, route selection, and temporal reset.
Core warms up and captures every route during preparation, restores initial state,
and releases partial resources if preparation fails. Preparation consumes no requests.

Core validates request structure before copying inputs, resetting state, or replaying.
Input/output copies stay outside capture. Returned Frames own independent pixels and
are not changed by later requests or model cleanup. Native input pixels may be released
after submission because pixtreme records their reads; callers retain borrowed pixels
and auxiliary storage until the queued work finishes. Output writes are recorded on
the supplied stream. Use Frame `wait` before reading on another stream and
`record_stream` after its last queued use, or acquire through DLPack for its stream
handoff. Finish pending work before closing backend resources.

Graph outputs allocate their pixels on the inference stream passed to `prepare`.
Keep that stream alive until the last reference to every returned Frame, shallow
clone, and exported DLPack view has been released, including after the model is
closed. Prefer a stream owned by pixtreme (`px.core.Stream`) and pass its numeric
`stream.ptr` handle, as in the example above. Pixtreme retains this allocation
stream for the output pixels, so its Python object can be released earlier.
For an external stream, such as one created by CuPy, the caller must keep the
stream alive through those final references.

`pixtreme.infer.verification` remains Python. Supply `LifecycleContractCases` for
constructor allocations, contract equivalence before/after preparation, rejection
before preparation and after failure/close, and resource rollback after failed
preparation. `verify_module_contract` also checks Frame boundaries and temporal state.
`verify_cuda_graph_contract(CudaGraphContractSuite(...))` reuses those checks and compares
complete request sequences with independently prepared normal instances. Missing
observations remain unverified. Architecture modules own their fixtures, thresholds,
support declarations, and evidence.

Run the synthetic native GPU binding tests with
`uv run --no-sync python -m pytest -m gpu tests/test_native_binding.py` and select the
GPU through `CUDA_VISIBLE_DEVICES`.

## License

The core distribution `pixtreme-infer` is licensed under the MIT License; see
[LICENSE](LICENSE). The wheel also includes nanobind (BSD-3-Clause), robin-map (MIT),
and a CUDA runtime license notice in `dist-info/licenses/`. Architecture distributions are separate distributions, each with its own
license. The core license does not apply to them, and installing one does not change the core
license.
