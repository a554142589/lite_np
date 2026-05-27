<p align="right">
English | <a href="README.zh-CN.md">简体中文</a>
</p>

# litenp

`litenp` is a compact C++17 ndarray library with a NumPy-like API, a single public
header, structure-aware lazy arrays, and fast materialized CPU kernels for common
small and medium workloads. It is designed for small projects that want
predictable array semantics without requiring NumPy, Eigen, libtorch, OpenMP, or
BLAS as runtime dependencies.

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

Latest run: 2026-05-27 on Ubuntu 22.04, GCC 11.4, Intel Core i7-13700K,
single-threaded benchmark settings:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The required benchmark gate passes every row:

```text
pass: 68
fail: 0
uncovered: 0
diagnostic: 0
```

The benchmark contains two different kinds of wins. Structure-aware rows use
lazy metadata or algebraic shortcuts for uniform, arange, eye, and related
arrays; they measure semantic avoidance of work, not generic dense memory
throughput. Materialized rows run actual dense kernels over existing buffers.

Structure-aware / lazy semantic rows, reported as `best ms / median ms`:

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ones 4M f32` | `0.000031 / 0.000031` | `0.354238 / 0.355597` | n/a | n/a | NumPy `11427.03x` |
| `full 4M f32` | `0.000031 / 0.000031` | `0.345758 / 0.352919` | n/a | n/a | NumPy `11153.48x` |
| `eye 1024x1024 f32` | `0.000032 / 0.000032` | `0.087279 / 0.087996` | n/a | n/a | NumPy `2727.47x` |
| `transpose uniform metadata 2048^2` | `0.000124 / 0.000214` | `8.739560 / 9.067105` | `10.050205 / 10.306238` | `5.921564 / 5.930852` | NumPy `70480.32x` |
| `sum uniform axis0 2048^2` | `0.000354 / 0.000862` | `0.321003 / 0.325190` | `1.151971 / 1.218315` | `0.418430 / 0.605177` | Eigen `3254.16x` |
| `matmul uniform 1024` | `0.246108 / 0.328481` | `41.216698 / 41.280204` | `14.513528 / 14.696222` | `41.406710 / 41.825745` | libtorch `168.25x` |

Materialized dense rows:

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `add 4M` | `0.324680 / 0.325622` | `1.420627 / 1.448524` | `1.345126 / 1.411481` | `2.458296 / 4.247955` | libtorch `7.57x` |
| `broadcast 2048^2` | `0.539378 / 0.782612` | `1.341642 / 1.375153` | `0.979961 / 1.302272` | `1.304367 / 1.389162` | NumPy `2.49x` |
| `relu_into 4M f32` | `0.396728 / 0.396728` | `1.463589 / 1.503579` | `0.773860 / 0.773860` | `0.945347 / 0.945347` | NumPy `3.69x` |
| `sqrt_into 4M f32` | `0.695131 / 0.695131` | `1.578819 / 1.596645` | n/a | n/a | NumPy `2.27x` |
| `where 4M` | `0.325165 / 0.326639` | `1.502054 / 1.532429` | `1.778512 / 1.811930` | `2.004297 / 2.068878` | libtorch `6.16x` |
| `concatenate axis0 2x1M` | `0.000203 / 0.000203` | `0.195517 / 0.205957` | n/a | n/a | NumPy `963.14x` |

Full report: [`docs/benchmark_2026-05-27.md`](docs/benchmark_2026-05-27.md).
Methodology and caveats: [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md).
Supported API semantics: [`docs/api_compatibility.md`](docs/api_compatibility.md).

Benchmark numbers are machine- and compiler-dependent. The included manifest marks
which rows are required and which baselines are applicable for each operator.
Do not read structure-aware rows as proof that generic dense kernels beat mature
numeric libraries in every workload.

## Project Layout

```text
include/litenp/litenp.hpp       public header
tests/test_litenp.cpp           correctness tests
tests/test_numpy_oracle.py      NumPy behavioral oracle
examples/basic.cpp              small usage example
benchmarks/bench_litenp.cpp     C++ benchmark runner
benchmarks/bench_numpy.py       NumPy baseline generator
benchmarks/benchmark_manifest.json
tools/compare_benchmarks.py     benchmark report generator
docs/api_compatibility.md       supported NumPy/libtorch semantics
docs/benchmark_methodology.md   benchmark categories and caveats
docs/views_and_lifetime.md      view lifetime and mutation notes
docs/                           design and benchmark notes
```

## Scope

`litenp` is not a full NumPy replacement. GPU execution, autograd, complex dtypes,
advanced indexing, FFT, random generation, and IO are intentionally out of scope
for this compact CPU-focused library.

The implemented subset is meant to match NumPy/libtorch behavior where it is
claimed. CI includes a NumPy oracle test for representative broadcasting,
elementwise, comparison, reduction, transpose, slicing, casting, and non-uniform
matmul cases.

`ArrayView<T>` is non-owning. A view is valid only while the source `Array<T>`
and its storage remain alive; avoid binding views to temporaries such as
`auto v = litenp::ones<float>({10}).view();`.

## License

`litenp` is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.
