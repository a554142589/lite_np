#!/usr/bin/env python3
"""Compare representative litenp behavior against NumPy."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


CPP_SOURCE = r"""
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "litenp/litenp.hpp"

template <typename T>
void print_array(const std::string& name, const litenp::Array<T>& array) {
    std::cout << name;
    for (const auto& value : array.to_vector()) {
        std::cout << ' ' << static_cast<double>(value);
    }
    std::cout << '\n';
}

template <typename T>
void print_scalar(const std::string& name, T value) {
    std::cout << name << ' ' << static_cast<double>(value) << '\n';
}

int main() {
    std::cout << std::setprecision(10);

    auto a = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto row = litenp::Array<float>::from_vector({3}, {10, 20, 30});
    print_array("add_broadcast", a + row);
    print_array("scalar_left", 2.0f + a);
    print_array("scalar_right", a - 2.0f);

    print_array("relu", litenp::relu(a - 4.0f));
    auto sqrt_input = litenp::Array<float>::from_vector({3}, {0, 4, 9});
    print_array("sqrt", litenp::sqrt(sqrt_input));

    auto cmp_rhs = litenp::Array<float>::from_vector({2, 3}, {0, 2, 4, 3, 7, 1});
    auto mask = litenp::greater(a, cmp_rhs);
    print_array("greater", mask);
    auto fallback = litenp::full<float>({2, 3}, -1.0f);
    print_array("where", litenp::where<float>(mask.view(), a.view(), fallback.view()));

    print_array("clip", litenp::clip(a, 2.0f, 5.0f));
    print_scalar("sum_all", litenp::sum(a));
    print_array("sum_axis0", litenp::sum(a, 0));
    print_array("sum_axis1", litenp::sum(a, 1));
    print_array("mean_axis0", litenp::mean(a, 0));
    print_array("max_axis1", litenp::max(a, 1));

    auto left = litenp::Array<float>::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
    auto right = litenp::Array<float>::from_vector({3, 2}, {7, 8, 9, 10, 11, 12});
    print_array("matmul", litenp::matmul(left, right));

    print_array("transpose_contiguous", litenp::as_contiguous<float>(a.transpose()));
    print_array("slice_step", litenp::as_contiguous<float>(a.view().slice(1, 0, 3, 2)));
    print_array("astype_i32", litenp::astype<std::int32_t>(a));

    return 0;
}
"""


def parse_output(text: str) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        rows[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float64)
    return rows


def assert_close(rows: dict[str, np.ndarray], name: str, expected: np.ndarray, atol: float = 1e-5) -> None:
    got = rows[name]
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=1e-5):
        raise AssertionError(f"{name}: got {got}, expected {expected}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    compiler = os.environ.get("CXX", "c++")

    with tempfile.TemporaryDirectory(prefix="litenp_numpy_oracle_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "oracle.cpp"
        binary = tmp_path / "oracle"
        source.write_text(CPP_SOURCE)
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-I", str(repo / "include"), str(source), "-o", str(binary)],
            check=True,
        )
        output = subprocess.check_output([str(binary)], text=True)

    rows = parse_output(output)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    row = np.array([10, 20, 30], dtype=np.float32)
    cmp_rhs = np.array([[0, 2, 4], [3, 7, 1]], dtype=np.float32)
    mask = a > cmp_rhs

    assert_close(rows, "add_broadcast", a + row)
    assert_close(rows, "scalar_left", 2.0 + a)
    assert_close(rows, "scalar_right", a - 2.0)
    assert_close(rows, "relu", np.maximum(a - 4.0, 0.0))
    assert_close(rows, "sqrt", np.sqrt(np.array([0, 4, 9], dtype=np.float32)))
    assert_close(rows, "greater", mask.astype(np.uint8))
    assert_close(rows, "where", np.where(mask, a, -1.0))
    assert_close(rows, "clip", np.clip(a, 2.0, 5.0))
    assert_close(rows, "sum_all", np.array([a.sum()]))
    assert_close(rows, "sum_axis0", a.sum(axis=0))
    assert_close(rows, "sum_axis1", a.sum(axis=1))
    assert_close(rows, "mean_axis0", a.mean(axis=0))
    assert_close(rows, "max_axis1", a.max(axis=1))
    assert_close(rows, "matmul", a @ np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32))
    assert_close(rows, "transpose_contiguous", np.ascontiguousarray(a.T))
    assert_close(rows, "slice_step", a[:, 0:3:2])
    assert_close(rows, "astype_i32", a.astype(np.int32))

    print("numpy oracle passed")


if __name__ == "__main__":
    main()

