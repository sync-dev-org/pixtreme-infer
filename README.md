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

The base dependency is `pixtreme>=2.14.0`. Optional capabilities are separated by extra:

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
uv sync --all-extras --no-install-project
uv pip install --no-deps --reinstall-package pixtreme-infer .
uv lock --check
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q
uv run --no-sync mypy
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build --wheel
```

The environment holds `pixtreme-infer` as an installed wheel, not an editable install,
because the `pixtreme` package itself comes from the `pixtreme` wheel. `uv sync` removes
that wheel, so run the first two commands again after `uv sync` and after changing the
Python or C++ source, and pass `--no-sync` to `uv run`.

The wheel includes the nanobind extension, C interface, header-only C++ wrapper and
nanobind adapter, CMake package and shared
library under `pixtreme/infer/`. Module builds find the CMake package with
`pixtreme.infer.get_cmake_dir()` and link `pixtreme_infer::pixtreme_infer`. The SDK
uses the shared library supplied by the installed `pixtreme` distribution. The wheel
contains one shared library, `libpixtreme_infer.so.3`. Importing
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
Inference accepts one `pixtreme.core.Frame`, named `pixtreme.NDArray` parameters, an optional
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

Auxiliary values use the names declared by the family. Pass `pixtreme.NDArray`
objects as keyword arguments; other values, including CuPy arrays, are rejected.
Wrap a DLPack producer explicitly with `pixtreme.NDArray.from_dlpack(array)`.
The C interface accepts name/`pixtreme_ndarray_t` pairs. C++ consumers can call
`model.run(frame, {{"motion", motion_array}}, stream)` with a `pixtreme::ndarray`.
Core and modules exchange CUDA pixels and arrays only as `pixtreme_frame_t` and
`pixtreme_ndarray_t`, and the public headers declare no DLPack types. The input
staging helper writes name/NDArray destinations, the output helper reads a
three-dimensional HWC NDArray, and a graph declaration lists its routes only. A module
chooses how its backend reads that storage; `pixtreme_ndarray_borrow` describes
module-allocated memory without copying it.
The C ABI is 3.1; modules built against ABI 2 or earlier must be rebuilt for SONAME 3.
Preparation, inference, `close()` and destruction run on the prepared device: core makes
it current for the call and returns with the calling thread's current CUDA device and
context unchanged, on success and on failure, even when the module selects its own device
inside. A thread whose current device differs can therefore run and release the model.

`model.family_contract` exposes the same native family's Frame and parameter declarations.
Python contract construction uses the native C validation as well.

The native class interface requires architecture modules built for core 0.18 or later.
Older modules exposing query/open factories are incompatible. Build modules against the
pixtreme headers of the installed pixtreme: pixtreme 2.14 rejects code built with headers
older than its C ABI 2.54. The core's
architecture table includes `spanv2` and `cda_vsr`; it does not recommend the older
`span` distribution.

## Creating a module by name from C and C++

C and C++ consumers can create a module's model from the module name and name/value
string options, without linking the module. The result is an ordinary model handle:
prepare, run, close and destroy it as above. Python keeps the module classes.

```c
pixtreme_infer_option_t weights = PIXTREME_INFER_OPTION_INIT, precision = PIXTREME_INFER_OPTION_INIT;
weights.name = "weights"; weights.value = "/models/spanv2.safetensors";
precision.name = "precision"; precision.value = "fp16";
/* ... input_height, input_width, and optionally use_cuda_graph ... */
const pixtreme_infer_option_t *options[] = {&weights, &precision /* , ... */};
pixtreme_infer_model_t *model = NULL;
pixtreme_infer_error_t *error = NULL;
pixtreme_infer_status_t status = pixtreme_infer_model_create_by_name(
    "spanv2", options, sizeof options / sizeof options[0], &model, &error);
```

```cpp
auto model = pixtreme::infer::Model::create_by_name("spanv2", {
    {"weights", "/models/spanv2.safetensors"}, {"precision", "fp16"},
    {"input_height", "720"}, {"input_width", "1280"}, {"use_cuda_graph", "true"}});
```

A module name matches `[a-z][a-z0-9_]*` and equals the module's Python root
`pixtreme.infer.<name>`. Option values are strings: integers are decimal, booleans are
`true` or `false`, rationals are `<numerator>/<denominator>`, and a hierarchical name
separates its levels with `.`. Empty values are never integers, booleans or rationals;
whether an empty text value is acceptable is up to each module. Every module accepts the
common options `weights` (a nonempty path; relative paths resolve against the process
working directory), `precision` (`fp32`,
`fp16`, `int8` or `fp8`), `input_height`, `input_width` and `use_cuda_graph`
(default `false`). The module rejects unknown, duplicate and missing options and values
of the wrong type or range by option name; errors never contain option values.

The status tells the failures apart: `PIXTREME_INFER_INVALID_ARGUMENT` for a malformed
name or rejected options, `PIXTREME_INFER_MODULE_NOT_FOUND` when the module is not
installed (the message names the distribution to install for a published module),
`PIXTREME_INFER_MODULE_LOAD_FAILED` when its entry library exists but cannot be
loaded (the message carries the loader's reason), and `PIXTREME_INFER_INCOMPATIBLE_ABI`
when its entry is missing or was built for an incompatible core ABI. Creation can be
called from several threads, and each call returns an independent model. To replace a
running model, close and destroy the old model and create the new one.

A module distribution makes itself available by name with one entry library:

- File: `pixtreme/infer/<name>/lib/libpixtreme_infer_<name>_entry.so` in the wheel. The
  core derives the path from the directory that contains its own `lib/` directory (the
  directory of `pixtreme/infer/lib/libpixtreme_infer.so.3`), without scanning
  directories, reading environment variables or reading registration files. The file name carries no module ABI version: the entry library
  depends on the module's own shared library (`NEEDED`, with RUNPATH `$ORIGIN`), so both
  the Python binding and creation by name use the same loaded module library.
- Entry function: the entry library exports
  `const pixtreme_infer_module_entry_t *pixtreme_infer_module_entry(void)` with C linkage.
  It returns a descriptor with static storage and does no CUDA, file or allocation work.
- ABI: the descriptor carries the core ABI the module was built against
  (`PIXTREME_INFER_MODULE_ENTRY_INIT` fills `PIXTREME_INFER_ABI_MAJOR` and `_MINOR`) and
  the module's creation function. The core calls that function only when the majors
  are equal and the module's minor is not newer than the core's.
- Lifetime: the core opens each name's entry library once per process and never closes
  it. Module state shared by instances must be thread-safe, and instances used on
  different threads must give the results each gives alone. A different module build
  takes effect in a new process.

Modules read their options with `pixtreme::infer::Settings` and
`pixtreme::infer::read_common_settings` from `<pixtreme/infer/core.hpp>`, which apply
these value forms and error rules in one place:

```cpp
pixtreme_infer_status_t create(const pixtreme_infer_option_t *const *options, uint64_t count,
                               pixtreme_infer_model_t **model, pixtreme_infer_error_t **error) {
    if (model) *model = nullptr;
    return pixtreme::infer::boundary(error, [&] {
        pixtreme::infer::Settings settings(options, count);
        auto common = pixtreme::infer::read_common_settings(settings);
        settings.finish(); // rejects every option no reader asked for
        *model = pixtreme::infer::make_model_handle<my_architecture::Implementation>(common);
    });
}
extern "C" __attribute__((visibility("default")))
const pixtreme_infer_module_entry_t *pixtreme_infer_module_entry(void) noexcept {
    static const pixtreme_infer_module_entry_t entry = {
        sizeof(pixtreme_infer_module_entry_t), PIXTREME_INFER_ABI_MAJOR, PIXTREME_INFER_ABI_MINOR, create};
    return &entry;
}
```

The core's static table of published module names and distributions (`spanv2`,
`cda_vsr`) only guides installation; an unlisted name is created when its entry library
is installed. `pixtreme_infer_module_distribution` reads the table. Creation by name is
implemented for the Linux wheel layout.

## Binding an architecture module

The public `<pixtreme/infer/nanobind.h>` adapter binds a consumer-side class derived
from `pixtreme::infer::Model`. That consumer owns the C model handle. Construct the
implementation with `make_model_handle<Implementation>(...)`, which transfers ownership
to core; never bind the same implementation object as a second owning Python object.
For example, after defining the architecture's consumer class and constructor:

```cpp
#include <pixtreme/infer/nanobind.h>

namespace my_architecture {
class Model : public pixtreme::infer::Model { /* ... */ };
}

NB_MODULE(_native, module) {
    nanobind::class_<my_architecture::Model> model(module, "Model");
    model.def(nanobind::init<ModuleSettings>());
    pixtreme::infer::python::bind_model(model);
}
```

Several architecture modules can be loaded in one Python process. nanobind matches
bound types across extensions by their C++ type name, so every C++ type a module binds
with `nanobind::class_` must have a name no other module uses: declare it in a namespace
named after the architecture, not in an anonymous namespace (an anonymous-namespace
`Model` has the same type name in every extension). Alternatively, give each extension
its own nanobind domain with `nanobind_add_module(... NB_DOMAIN <name> ...)`.

The adapter uses the header-only C++ wrapper and public Python contract value types. Module
and core extensions may use different nanobind domains; Python inheritance from a
core class is not required. Frame arguments and owning results use pixtreme's public
`<pixtreme/nanobind.h>` type casters. Module constructors can use the adapter's
`pixtreme::infer::python::call` helper to apply the same error mapping: invalid arguments
become `ValueError`, invalid lifecycle states and backend failures become `RuntimeError`,
and allocation failures become `MemoryError`.

`pixtreme::infer::Family::create(desc)` owns a standalone family and copies the input
declaration, strings, and arrays through the C interface. `Family` is move-only;
moving transfers ownership, and destruction or move assignment releases the previously
owned family. `Family::desc()` borrows its declaration and nested storage until that
family is destroyed or replaced. `pixtreme::infer::family_desc(model.contract().family)`
borrows a model's family declaration until the model is destroyed; closing the model
does not end that borrow. Never destroy a borrowed family. The adapter copies borrowed
declarations into Python values while their owner is alive.

## Native CUDA Graphs and verification

The constructor's `use_cuda_graph` selects core's C++ graph implementation. Graph
preparation requires an explicit stream created with `cudaStreamNonBlocking`
(`px.core.Stream(non_blocking=True)` in Python) and rejects other streams before
recording any graph, so work on the legacy default stream elsewhere does not disturb the
capture. While a model prepares, no thread may synchronize the whole device, because that
invalidates the capture. The module owns its fixed GPU
storage and supplies allocation-free backend enqueue hooks, route selection, and temporal reset.
Core warms up and captures every route during preparation, restores initial state,
and releases partial resources if preparation fails. Preparation consumes no requests.

Core validates request structure before copying inputs, resetting state, or replaying.
Input/output copies stay outside capture. Returned Frames own independent pixels and
are not changed by later requests or model cleanup. Native input pixels may be released
after submission because pixtreme records their reads. The same rule applies to
auxiliary NDArrays allocated by pixtreme: core waits for recorded writers and records
its reads, so callers may release the arrays as soon as inference returns. For borrowed
or DLPack-imported arrays and Frames, callers keep the external memory owner alive
until the inference stream finishes reading it. Output writes are recorded on
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
