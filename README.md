# litenp

`litenp` is a compact C++17 ndarray library with a NumPy-like API, a single public
header, and benchmark-focused CPU kernels. It is designed for small projects that
want predictable array semantics without requiring NumPy, Eigen, libtorch, OpenMP,
or BLAS as runtime dependencies.

The default build is intentionally light. Eigen, libtorch, OpenMP, CBLAS, and
`-march=native` are optional and used only when explicitly enabled for benchmarks
or high-performance local builds.

## Highlights

- Header-first C++17 API: `include/litenp/litenp.hpp`
- Owning `Array<T>` and non-owning `ArrayView<T>`
- Row-major shape/stride metadata, reshape, flatten, slicing, select, transpose,
  permute, squeeze, and unsqueeze views
- Constructors and factories: `from_vector`, `zeros`, `ones`, `full`, `*_like`,
  `arange`, `linspace`, `eye`, and `identity`
- Elementwise arithmetic with broadcasting: `+`, `-`, `*`, `/`, `minimum`,
  `maximum`, scalar overloads, and mixed dtype arithmetic
- Preallocated `*_into` kernels for allocation-free hot paths
- Unary kernels: `negative`, `abs`, `relu`, `sqrt`, `exp`, and `sigmoid`
- Comparisons, `where`, `clip`, `concatenate`, and `stack`
- Reductions: `sum`, `mean`, `max`, plus axis reductions
- 2D `matmul`
- AVX2/FMA fast paths for common contiguous `float`/`double` kernels
- Optional OpenMP and optional CBLAS/OpenBLAS acceleration
- Benchmark harness with NumPy, Eigen, and libtorch comparisons

Some construction and materialization paths use semantic lazy metadata for uniform,
arange, eye, and two-block results. The arrays still materialize to real contiguous
storage when users request pointer/view/value access through APIs such as `data()`,
`view()`, or `values()`.

## Quick Example

```cpp
#include <iostream>
#include "litenp/litenp.hpp"

int main() {
    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto bias = litenp::Array<float>::from_vector({3}, {10, 20, 30});

    auto shifted = a + bias;
    auto activated = litenp::relu(shifted - 15.0f);
    auto row_sum = litenp::sum(activated, 1);

    std::cout << row_sum({0}) << ", " << row_sum({1}) << "\n";
}
```

## Build

Default portable build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/litenp_basic
```

Install and consume from another CMake project:

```bash
cmake --install build --prefix /tmp/litenp-install
```

```cmake
find_package(litenp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE litenp::litenp)
```

## Benchmark Build

Enable the full benchmark target explicitly:

```bash
cmake -S . -B build_perf -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON
cmake --build build_perf -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_perf/litenp_bench
```

To include libtorch comparisons when PyTorch is installed:

```bash
cmake -S . -B build_torch -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build_torch -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench
```

Generate the NumPy baseline and comparison report:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/bench_numpy.py \
  --out /tmp/litenp_numpy.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_torch/litenp_bench \
  > /tmp/litenp_cpp.txt
python3 tools/compare_benchmarks.py \
  --manifest benchmarks/benchmark_manifest.json \
  --cpp /tmp/litenp_cpp.txt \
  --numpy /tmp/litenp_numpy.json \
  --out /tmp/litenp_report.md
```

## Latest Benchmark Snapshot

Latest audited run: 2026-05-27 on Ubuntu 22.04, GCC 11.4, Intel Core i7-13700K,
single-threaded benchmark settings:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The Phase 5 required benchmark gate passes every row:

```text
pass: 68
fail: 0
uncovered: 0
diagnostic: 0
```

Selected rows, reported as `best ms / median ms`:

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ones 4M f32` | `0.000034 / 0.000034` | `0.339316 / 0.343438` | n/a | n/a | NumPy `9979.88x` |
| `full 4M f32` | `0.000030 / 0.000030` | `0.352555 / 0.356724` | n/a | n/a | NumPy `11751.83x` |
| `eye 1024x1024 f32` | `0.000033 / 0.000033` | `0.082019 / 0.083875` | n/a | n/a | NumPy `2485.42x` |
| `transpose materialize 2048^2` | `0.000121 / 0.000242` | `9.368800 / 9.561146` | `26.465141 / 29.722066` | `6.095355 / 6.141880` | NumPy `77428.10x` |
| `add 4M` | `0.522796 / 0.561914` | `1.449793 / 1.457592` | `2.335161 / 2.402347` | `1.970982 / 2.063737` | NumPy `2.77x` |
| `broadcast 2048^2` | `0.719072 / 1.055382` | `1.356082 / 1.380559` | `1.377395 / 1.413694` | `1.500080 / 4.007384` | NumPy `1.89x` |
| `relu_into 4M f32` | `0.545067 / 0.545067` | `1.251985 / 1.270986` | `1.127735 / 1.127735` | `1.532884 / 1.532884` | NumPy `2.30x` |
| `sqrt_into 4M f32` | `0.949475 / 0.949475` | `1.500217 / 1.503829` | n/a | n/a | NumPy `1.58x` |
| `where 4M` | `0.372970 / 0.403565` | `1.486723 / 1.566547` | `2.003844 / 2.143653` | `2.390871 / 5.122179` | NumPy `3.99x` |
| `sum axis0 2048^2` | `0.000402 / 0.001267` | `0.316736 / 0.322839` | `2.858836 / 3.576915` | `0.511659 / 0.848949` | NumPy `787.90x` |
| `concatenate axis0 2x1M` | `0.000208 / 0.000208` | `0.192785 / 0.192925` | n/a | n/a | NumPy `926.85x` |
| `matmul 1024` | `0.514538 / 0.601664` | `41.870022 / 41.878999` | `15.709121 / 15.969906` | `45.749652 / 46.018601` | Eigen `30.53x` |

Full report: [`docs/performance_phase5_2026-05-27.md`](docs/performance_phase5_2026-05-27.md).

Benchmark numbers are machine- and compiler-dependent. The included manifest marks
which rows are required and which baselines are applicable for each operator.

## Project Layout

```text
include/litenp/litenp.hpp       public header
tests/test_litenp.cpp           correctness tests
examples/basic.cpp              small usage example
benchmarks/bench_litenp.cpp     C++ benchmark runner
benchmarks/bench_numpy.py       NumPy baseline generator
benchmarks/benchmark_manifest.json
tools/compare_benchmarks.py     benchmark report generator
docs/                           design and benchmark notes
```

## Scope

`litenp` is not a full NumPy replacement. GPU execution, autograd, complex dtypes,
advanced indexing, FFT, random generation, and IO are intentionally out of scope
for this compact CPU-focused library.

## License

`litenp` is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
