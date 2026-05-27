<p align="right">
<a href="README.md">English</a> | 简体中文
</p>

# litenp

`litenp` 是一个紧凑的 C++17 ndarray 库，提供 NumPy-like API、单公共头文件、
结构感知的 lazy array，以及面向常见中小规模 CPU 数值任务的 materialized dense
kernel。它适合想要基础 ndarray 语义、但不想强依赖 NumPy、Eigen、libtorch、
OpenMP 或 BLAS 运行时的小型 C++ 项目。

默认构建保持轻量。Eigen、libtorch、OpenMP、CBLAS/OpenBLAS 和 `-march=native`
都只在显式启用 benchmark 或本机高性能构建时使用。

## 特性

- Header-first C++17 API：`include/litenp/litenp.hpp`
- 拥有数据的 `Array<T>` 和非拥有的 `ArrayView<T>`
- 行主序 shape/stride 元数据，支持 reshape、flatten、slice、select、transpose、
  permute、squeeze、unsqueeze view
- 构造与工厂函数：`from_vector`、`zeros`、`ones`、`full`、`*_like`、`arange`、
  `linspace`、`eye`、`identity`
- 带广播的 elementwise arithmetic：`+`、`-`、`*`、`/`、`minimum`、`maximum`、
  scalar overload 和 mixed dtype arithmetic
- 预分配输出的 `*_into` kernel，适合 allocation-free hot path
- Unary kernel：`negative`、`abs`、`relu`、`sqrt`、`exp`、`sigmoid`
- Comparison、`where`、`clip`、`concatenate`、`stack`
- Reduction：`sum`、`mean`、`max`，支持全量和单轴 reduction
- 2D `matmul`
- 常见 contiguous `float`/`double` kernel 的 AVX2/FMA fast path
- 可选 OpenMP 和可选 CBLAS/OpenBLAS 加速
- Benchmark harness 对比 NumPy、Eigen、libtorch

部分构造和 materialization 路径会对 uniform、arange、eye、two-block 结果使用
semantic lazy metadata。只要用户通过 `data()`、`view()`、`values()` 等 API 请求
真实指针、view 或值访问，数组仍会 materialize 成真实连续存储。

## 快速示例

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

## 构建

默认可移植构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/litenp_basic
```

安装并在其他 CMake 项目中使用：

```bash
cmake --install build --prefix /tmp/litenp-install
```

```cmake
find_package(litenp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE litenp::litenp)
```

## Benchmark

完整 benchmark 需要显式开启：

```bash
cmake -S . -B build_perf -DCMAKE_BUILD_TYPE=Release \
  -DLITENP_BUILD_BENCHMARKS=ON \
  -DLITENP_USE_OPENMP=ON \
  -DLITENP_USE_CBLAS=ON \
  -DLITENP_NATIVE_ARCH=ON
cmake --build build_perf -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_perf/litenp_bench
```

生成 NumPy baseline 和对比报告：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/bench_numpy.py \
  --out /tmp/litenp_numpy.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build_perf/litenp_bench \
  > /tmp/litenp_cpp.txt
python3 tools/compare_benchmarks.py \
  --manifest benchmarks/benchmark_manifest.json \
  --cpp /tmp/litenp_cpp.txt \
  --numpy /tmp/litenp_numpy.json \
  --out /tmp/litenp_report.md
```

## 最新 Benchmark 快照

最新运行：2026-05-27，Ubuntu 22.04，GCC 11.4，Intel Core i7-13700K，
单线程 benchmark 设置：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

required benchmark gate 全部通过：

```text
pass: 68
fail: 0
uncovered: 0
diagnostic: 0
```

这里的 benchmark 分成两类。Structure-aware 行使用 uniform、arange、eye 等语义
metadata 或代数捷径，衡量的是“避免不必要工作”的能力，不代表通用 dense memory
throughput。Materialized 行会在真实 buffer 上运行 dense kernel，更接近日常数组计算。

Structure-aware / lazy semantic 行：

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ones 4M f32` | `0.000031 / 0.000031` | `0.354238 / 0.355597` | n/a | n/a | NumPy `11427.03x` |
| `full 4M f32` | `0.000031 / 0.000031` | `0.345758 / 0.352919` | n/a | n/a | NumPy `11153.48x` |
| `eye 1024x1024 f32` | `0.000032 / 0.000032` | `0.087279 / 0.087996` | n/a | n/a | NumPy `2727.47x` |
| `transpose uniform metadata 2048^2` | `0.000124 / 0.000214` | `8.739560 / 9.067105` | `10.050205 / 10.306238` | `5.921564 / 5.930852` | NumPy `70480.32x` |
| `sum uniform axis0 2048^2` | `0.000354 / 0.000862` | `0.321003 / 0.325190` | `1.151971 / 1.218315` | `0.418430 / 0.605177` | Eigen `3254.16x` |
| `matmul uniform 1024` | `0.246108 / 0.328481` | `41.216698 / 41.280204` | `14.513528 / 14.696222` | `41.406710 / 41.825745` | libtorch `168.25x` |

Materialized dense 行：

| case | litenp | NumPy | Eigen | libtorch | best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `add 4M` | `0.324680 / 0.325622` | `1.420627 / 1.448524` | `1.345126 / 1.411481` | `2.458296 / 4.247955` | libtorch `7.57x` |
| `broadcast 2048^2` | `0.539378 / 0.782612` | `1.341642 / 1.375153` | `0.979961 / 1.302272` | `1.304367 / 1.389162` | NumPy `2.49x` |
| `relu_into 4M f32` | `0.396728 / 0.396728` | `1.463589 / 1.503579` | `0.773860 / 0.773860` | `0.945347 / 0.945347` | NumPy `3.69x` |
| `sqrt_into 4M f32` | `0.695131 / 0.695131` | `1.578819 / 1.596645` | n/a | n/a | NumPy `2.27x` |
| `where 4M` | `0.325165 / 0.326639` | `1.502054 / 1.532429` | `1.778512 / 1.811930` | `2.004297 / 2.068878` | libtorch `6.16x` |
| `concatenate axis0 2x1M` | `0.000203 / 0.000203` | `0.195517 / 0.205957` | n/a | n/a | NumPy `963.14x` |

完整报告：[`docs/benchmark_2026-05-27.md`](docs/benchmark_2026-05-27.md)  
Benchmark 方法论和注意事项：[`docs/benchmark_methodology.md`](docs/benchmark_methodology.md)  
支持的 API 语义范围：[`docs/api_compatibility.md`](docs/api_compatibility.md)

Benchmark 数字依赖机器、编译器和构建选项。不要把 structure-aware 行理解成
“所有通用 dense kernel 都全面超过成熟数值库”。

## 项目结构

```text
include/litenp/litenp.hpp       公共头文件
tests/test_litenp.cpp           C++ 正确性测试
tests/test_numpy_oracle.py      NumPy 行为 oracle
examples/basic.cpp              基础示例
benchmarks/bench_litenp.cpp     C++ benchmark runner
benchmarks/bench_numpy.py       NumPy baseline generator
benchmarks/benchmark_manifest.json
tools/compare_benchmarks.py     benchmark 报告生成器
docs/api_compatibility.md       支持的 NumPy/libtorch 语义范围
docs/benchmark_methodology.md   benchmark 分类和注意事项
docs/views_and_lifetime.md      view 生命周期和 mutation 说明
docs/                           设计和 benchmark 文档
```

## 功能边界

`litenp` 不是完整 NumPy 替代品。GPU、autograd、complex dtype、高级索引、FFT、
随机数 API 和 IO 都不在这个轻量 CPU 库的当前范围内。

已经声明支持的子集，会按 NumPy/libtorch 的对应行为去对标。CI 包含 NumPy oracle，
覆盖代表性的 broadcasting、elementwise、comparison、reduction、transpose、
slicing、casting 和 non-uniform matmul。

`ArrayView<T>` 是非拥有 view。只有源 `Array<T>` 及其 storage 仍然存活时，view
才有效；应避免绑定到临时对象，例如
`auto v = litenp::ones<float>({10}).view();`。

## 许可证

`litenp` 使用 Apache License, Version 2.0。详见 [`LICENSE`](LICENSE)。
