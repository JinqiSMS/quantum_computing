"""复现 concentration.pdf 中 Figure 3 的数值实验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, norm


TIME = 2.0
RUN_COUNT = 50
FIXED_GATE_COUNT = 160
RANDOM_SEED = 20260620
GATE_COUNTS = [20, 40, 80, 120, 160, 240, 320]
SYSTEM_SIZES = [4, 5, 6, 7]
REFERENCE_SYSTEM_SIZE = 4
FIGURE_DIR = Path("figures")


@dataclass(frozen=True)
class HeisenbergData:
    """保存给定系统尺寸下的哈密顿量和 qDRIFT 简单项。"""

    hamiltonian: np.ndarray
    terms: list[np.ndarray]
    simple_gate_cache: dict[int, list[np.ndarray]]


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    """计算多个单量子比特算符的 Kronecker 积。"""
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def make_two_site_term(pauli: np.ndarray, site_index: int, qubit_count: int) -> np.ndarray:
    """构造归一化的一维相邻两体 Pauli 相互作用项。"""
    identity = np.eye(2, dtype=complex)
    operators = [identity for _ in range(qubit_count)]
    operators[site_index] = pauli
    operators[site_index + 1] = pauli
    return kron_all(operators) / (qubit_count - 1)


def build_heisenberg_data(qubit_count: int) -> HeisenbergData:
    """构造论文数值实验使用的一维 Heisenberg 链数据。"""
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    terms: list[np.ndarray] = []
    for site_index in range(qubit_count - 1):
        for pauli in (pauli_x, pauli_y, pauli_z):
            terms.append(make_two_site_term(pauli, site_index, qubit_count))

    hamiltonian = np.sum(terms, axis=0)
    return HeisenbergData(hamiltonian=hamiltonian, terms=terms, simple_gate_cache={})


def get_qdrift_gates(data: HeisenbergData, gate_count: int) -> list[np.ndarray]:
    """获取指定门数下每个简单项对应的 qDRIFT 单步酉矩阵。"""
    if gate_count not in data.simple_gate_cache:
        lambda_value = sum(norm(term, 2) for term in data.terms)
        step_scale = TIME * lambda_value / gate_count
        data.simple_gate_cache[gate_count] = [expm(-1j * step_scale * term / norm(term, 2)) for term in data.terms]
    return data.simple_gate_cache[gate_count]


def sample_product_formula(data: HeisenbergData, gate_count: int, rng: np.random.Generator) -> np.ndarray:
    """按 qDRIFT 规则抽样随机乘积公式。"""
    gates = get_qdrift_gates(data, gate_count)
    dimension = data.hamiltonian.shape[0]
    product = np.eye(dimension, dtype=complex)
    sampled_indices = rng.integers(0, len(gates), size=gate_count)
    for sampled_index in sampled_indices:
        product = gates[sampled_index] @ product
    return product


def sample_product_state(qubit_count: int, rng: np.random.Generator) -> np.ndarray:
    """抽样单量子比特 Haar 随机态的张量积态。"""
    state = np.array([1.0 + 0.0j])
    for _ in range(qubit_count):
        vector = rng.normal(size=2) + 1j * rng.normal(size=2)
        vector = vector / norm(vector)
        state = np.kron(state, vector)
    return state


def estimate_errors(
    qubit_count: int,
    gate_count: int,
    run_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """估计所有输入态误差和固定输入态误差。"""
    data = build_heisenberg_data(qubit_count)
    target = expm(-1j * TIME * data.hamiltonian)
    fixed_state = sample_product_state(qubit_count, rng)

    all_state_errors = np.empty(run_count)
    fixed_state_errors = np.empty(run_count)
    for run_index in range(run_count):
        product = sample_product_formula(data, gate_count, rng)
        difference = target - product
        all_state_errors[run_index] = norm(difference, 2)
        fixed_state_errors[run_index] = norm(difference @ fixed_state)
    return all_state_errors, fixed_state_errors


def summarize(values: np.ndarray) -> tuple[float, float]:
    """返回样本均值和样本标准差。"""
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_gate_count_sweep(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """运行固定系统尺寸下不同门数的误差实验。"""
    all_means = []
    all_stds = []
    fixed_means = []
    fixed_stds = []
    for gate_count in GATE_COUNTS:
        all_errors, fixed_errors = estimate_errors(REFERENCE_SYSTEM_SIZE, gate_count, RUN_COUNT, rng)
        all_mean, all_std = summarize(all_errors)
        fixed_mean, fixed_std = summarize(fixed_errors)
        all_means.append(all_mean)
        all_stds.append(all_std)
        fixed_means.append(fixed_mean)
        fixed_stds.append(fixed_std)
        print(f"n={REFERENCE_SYSTEM_SIZE}, N={gate_count}: all={all_mean:.4g}, fixed={fixed_mean:.4g}")

    return {
        "gate_counts": np.array(GATE_COUNTS),
        "all_means": np.array(all_means),
        "all_stds": np.array(all_stds),
        "fixed_means": np.array(fixed_means),
        "fixed_stds": np.array(fixed_stds),
    }


def run_system_size_sweep(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """运行固定门数下不同系统尺寸的归一化误差实验。"""
    all_means = []
    all_stds = []
    fixed_means = []
    fixed_stds = []
    for qubit_count in SYSTEM_SIZES:
        all_errors, fixed_errors = estimate_errors(qubit_count, FIXED_GATE_COUNT, RUN_COUNT, rng)
        all_mean, all_std = summarize(all_errors)
        fixed_mean, fixed_std = summarize(fixed_errors)
        all_means.append(all_mean)
        all_stds.append(all_std)
        fixed_means.append(fixed_mean)
        fixed_stds.append(fixed_std)
        print(f"n={qubit_count}, N={FIXED_GATE_COUNT}: all={all_mean:.4g}, fixed={fixed_mean:.4g}")

    all_means_array = np.array(all_means)
    fixed_means_array = np.array(fixed_means)
    return {
        "system_sizes": np.array(SYSTEM_SIZES),
        "all_relative_means": all_means_array / all_means_array[0],
        "all_relative_stds": np.array(all_stds) / all_means_array[0],
        "fixed_relative_means": fixed_means_array / fixed_means_array[0],
        "fixed_relative_stds": np.array(fixed_stds) / fixed_means_array[0],
    }


def save_individual_figures(gate_data: dict[str, np.ndarray], size_data: dict[str, np.ndarray]) -> None:
    """分别保存四个子图到 figures 目录。"""
    FIGURE_DIR.mkdir(exist_ok=True)

    plot_error_vs_gate_count(
        gate_data["gate_counts"],
        gate_data["all_means"],
        gate_data["all_stds"],
        "All input states",
        FIGURE_DIR / "all_input_states_error_vs_gate_count.png",
    )
    plot_error_vs_gate_count(
        gate_data["gate_counts"],
        gate_data["fixed_means"],
        gate_data["fixed_stds"],
        "Fixed input state",
        FIGURE_DIR / "fixed_input_state_error_vs_gate_count.png",
    )
    plot_relative_error_vs_size(
        size_data["system_sizes"],
        size_data["all_relative_means"],
        size_data["all_relative_stds"],
        size_data["system_sizes"] / REFERENCE_SYSTEM_SIZE,
        "All input states",
        FIGURE_DIR / "all_input_states_relative_error_vs_system_size.png",
    )
    plot_relative_error_vs_size(
        size_data["system_sizes"],
        size_data["fixed_relative_means"],
        size_data["fixed_relative_stds"],
        np.ones_like(size_data["system_sizes"], dtype=float),
        "Fixed input state",
        FIGURE_DIR / "fixed_input_state_relative_error_vs_system_size.png",
    )


def plot_error_vs_gate_count(
    gate_counts: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    output_path: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    """绘制误差随 qDRIFT 门数变化的曲线。"""
    own_figure = axis is None
    if own_figure:
        _, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    assert axis is not None

    axis.plot(gate_counts, means, marker="o", color="#1f77b4")
    axis.fill_between(gate_counts, means - stds, means + stds, color="#1f77b4", alpha=0.2)
    axis.set_title(title)
    axis.set_xlabel("Gate count N")
    axis.set_ylabel("Error")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)

    if own_figure and output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_relative_error_vs_size(
    system_sizes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    theory_line: np.ndarray,
    title: str,
    output_path: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    """绘制固定门数下归一化误差随系统尺寸变化的曲线。"""
    own_figure = axis is None
    if own_figure:
        _, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    assert axis is not None

    axis.plot(system_sizes, means, marker="o", color="#d62728", label="simulation")
    axis.fill_between(system_sizes, means - stds, means + stds, color="#d62728", alpha=0.2)
    axis.plot(system_sizes, theory_line, linestyle=":", color="black", label="theory")
    axis.set_title(title)
    axis.set_xlabel("Number of qubits n")
    axis.set_ylabel(r"Relative error $\epsilon_n / \epsilon_4$")
    axis.set_xticks(system_sizes)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)

    if own_figure and output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()


def save_combined_figure(gate_data: dict[str, np.ndarray], size_data: dict[str, np.ndarray]) -> None:
    """保存与论文 Figure 3 对应的合并图。"""
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    plot_error_vs_gate_count(
        gate_data["gate_counts"],
        gate_data["all_means"],
        gate_data["all_stds"],
        "All input states",
        axis=axes[0, 0],
    )
    plot_error_vs_gate_count(
        gate_data["gate_counts"],
        gate_data["fixed_means"],
        gate_data["fixed_stds"],
        "Fixed input state",
        axis=axes[0, 1],
    )
    plot_relative_error_vs_size(
        size_data["system_sizes"],
        size_data["all_relative_means"],
        size_data["all_relative_stds"],
        size_data["system_sizes"] / REFERENCE_SYSTEM_SIZE,
        f"All input states, N={FIXED_GATE_COUNT}",
        axis=axes[1, 0],
    )
    plot_relative_error_vs_size(
        size_data["system_sizes"],
        size_data["fixed_relative_means"],
        size_data["fixed_relative_stds"],
        np.ones_like(size_data["system_sizes"], dtype=float),
        f"Fixed input state, N={FIXED_GATE_COUNT}",
        axis=axes[1, 1],
    )

    figure.suptitle("Numerical experiments for 1D Heisenberg model", fontsize=14)
    figure.savefig(FIGURE_DIR / "figure3_reproduction.png", dpi=300)
    plt.close(figure)


def main() -> None:
    """运行全部复现实验并保存图片。"""
    rng = np.random.default_rng(RANDOM_SEED)
    gate_data = run_gate_count_sweep(rng)
    size_data = run_system_size_sweep(rng)
    save_combined_figure(gate_data, size_data)
    save_individual_figures(gate_data, size_data)


if __name__ == "__main__":
    main()
