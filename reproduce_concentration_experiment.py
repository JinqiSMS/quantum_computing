"""复现 concentration.pdf 中 Figure 3 的数值实验，并比较多个系统规模。"""

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
SYSTEM_SIZES = [4, 5, 6, 7, 8]
REFERENCE_SYSTEM_SIZE = 4
FIGURE_DIR = Path("figures")


@dataclass(frozen=True)
class LocalTerm:
    """保存一个相邻两量子比特 Heisenberg 简单项的信息。"""

    site_index: int
    pauli_index: int


@dataclass
class HeisenbergData:
    """保存给定系统尺寸下的哈密顿量、目标演化和局域项。"""

    qubit_count: int
    hamiltonian: np.ndarray
    target: np.ndarray
    local_terms: list[LocalTerm]
    gate_cache: dict[int, list[np.ndarray]]


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
    pauli_matrices = make_pauli_matrices()
    local_terms: list[LocalTerm] = []
    dense_terms: list[np.ndarray] = []

    for site_index in range(qubit_count - 1):
        for pauli_index, pauli in enumerate(pauli_matrices):
            local_terms.append(LocalTerm(site_index=site_index, pauli_index=pauli_index))
            dense_terms.append(make_two_site_term(pauli, site_index, qubit_count))

    hamiltonian = np.sum(dense_terms, axis=0)
    target = expm(-1j * TIME * hamiltonian)
    return HeisenbergData(
        qubit_count=qubit_count,
        hamiltonian=hamiltonian,
        target=target,
        local_terms=local_terms,
        gate_cache={},
    )


def make_pauli_matrices() -> list[np.ndarray]:
    """返回 Pauli X、Y、Z 矩阵。"""
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [pauli_x, pauli_y, pauli_z]


def get_local_qdrift_gates(data: HeisenbergData, gate_count: int) -> list[np.ndarray]:
    """获取指定门数下 X、Y、Z 相邻两体 qDRIFT 单步门。"""
    if gate_count not in data.gate_cache:
        lambda_value = 3.0
        step_scale = TIME * lambda_value / gate_count
        data.gate_cache[gate_count] = [
            expm(-1j * step_scale * np.kron(pauli, pauli)) for pauli in make_pauli_matrices()
        ]
    return data.gate_cache[gate_count]


def apply_two_qubit_gate_to_matrix(
    matrix: np.ndarray,
    gate: np.ndarray,
    site_index: int,
    qubit_count: int,
) -> np.ndarray:
    """把两量子比特门左乘到完整酉矩阵上。"""
    dimension = 2**qubit_count
    tensor = matrix.reshape([2] * qubit_count + [dimension])
    tensor = np.moveaxis(tensor, [site_index, site_index + 1], [0, 1])
    updated = gate @ tensor.reshape(4, -1)
    updated = updated.reshape([2, 2] + [2] * (qubit_count - 2) + [dimension])
    updated = np.moveaxis(updated, [0, 1], [site_index, site_index + 1])
    return updated.reshape(dimension, dimension)


def apply_two_qubit_gate_to_state(
    state: np.ndarray,
    gate: np.ndarray,
    site_index: int,
    qubit_count: int,
) -> np.ndarray:
    """把两量子比特门作用到完整态向量上。"""
    tensor = state.reshape([2] * qubit_count)
    tensor = np.moveaxis(tensor, [site_index, site_index + 1], [0, 1])
    updated = gate @ tensor.reshape(4, -1)
    updated = updated.reshape([2, 2] + [2] * (qubit_count - 2))
    updated = np.moveaxis(updated, [0, 1], [site_index, site_index + 1])
    return updated.reshape(2**qubit_count)


def sample_product_formula(
    data: HeisenbergData,
    gate_count: int,
    sampled_terms: np.ndarray,
) -> np.ndarray:
    """按 qDRIFT 抽样结果构造随机乘积公式的完整矩阵。"""
    gates = get_local_qdrift_gates(data, gate_count)
    dimension = data.target.shape[0]
    product = np.eye(dimension, dtype=complex)
    for sampled_term in sampled_terms:
        local_term = data.local_terms[int(sampled_term)]
        product = apply_two_qubit_gate_to_matrix(
            product,
            gates[local_term.pauli_index],
            local_term.site_index,
            data.qubit_count,
        )
    return product


def apply_product_formula_to_state(
    data: HeisenbergData,
    gate_count: int,
    sampled_terms: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """按 qDRIFT 抽样结果把随机乘积公式作用到态向量。"""
    gates = get_local_qdrift_gates(data, gate_count)
    evolved_state = state.copy()
    for sampled_term in sampled_terms:
        local_term = data.local_terms[int(sampled_term)]
        evolved_state = apply_two_qubit_gate_to_state(
            evolved_state,
            gates[local_term.pauli_index],
            local_term.site_index,
            data.qubit_count,
        )
    return evolved_state


def sample_product_state(qubit_count: int, rng: np.random.Generator) -> np.ndarray:
    """抽样单量子比特 Haar 随机态的张量积态。"""
    state = np.array([1.0 + 0.0j])
    for _ in range(qubit_count):
        vector = rng.normal(size=2) + 1j * rng.normal(size=2)
        vector = vector / norm(vector)
        state = np.kron(state, vector)
    return state


def estimate_errors(
    data: HeisenbergData,
    gate_count: int,
    run_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """估计所有输入态误差和固定输入态误差。"""
    fixed_state = sample_product_state(data.qubit_count, rng)
    target_state = data.target @ fixed_state
    all_state_errors = np.empty(run_count)
    fixed_state_errors = np.empty(run_count)

    for run_index in range(run_count):
        sampled_terms = rng.integers(0, len(data.local_terms), size=gate_count)
        product = sample_product_formula(data, gate_count, sampled_terms)
        product_state = apply_product_formula_to_state(data, gate_count, sampled_terms, fixed_state)
        all_state_errors[run_index] = norm(data.target - product, 2)
        fixed_state_errors[run_index] = norm(target_state - product_state)
    return all_state_errors, fixed_state_errors


def summarize(values: np.ndarray) -> tuple[float, float]:
    """返回样本均值和样本标准差。"""
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_experiments(rng: np.random.Generator) -> dict[int, dict[str, np.ndarray]]:
    """对所有系统尺寸和所有门数运行复现实验。"""
    results: dict[int, dict[str, np.ndarray]] = {}
    for qubit_count in SYSTEM_SIZES:
        data = build_heisenberg_data(qubit_count)
        all_means = []
        all_stds = []
        fixed_means = []
        fixed_stds = []
        for gate_count in GATE_COUNTS:
            all_errors, fixed_errors = estimate_errors(data, gate_count, RUN_COUNT, rng)
            all_mean, all_std = summarize(all_errors)
            fixed_mean, fixed_std = summarize(fixed_errors)
            all_means.append(all_mean)
            all_stds.append(all_std)
            fixed_means.append(fixed_mean)
            fixed_stds.append(fixed_std)
            print(f"n={qubit_count}, N={gate_count}: all={all_mean:.4g}, fixed={fixed_mean:.4g}")

        results[qubit_count] = {
            "gate_counts": np.array(GATE_COUNTS),
            "all_means": np.array(all_means),
            "all_stds": np.array(all_stds),
            "fixed_means": np.array(fixed_means),
            "fixed_stds": np.array(fixed_stds),
        }
    return results


def extract_size_sweep(results: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """从全部实验结果中提取固定门数下的系统规模扫描。"""
    fixed_index = GATE_COUNTS.index(FIXED_GATE_COUNT)
    all_means = np.array([results[n]["all_means"][fixed_index] for n in SYSTEM_SIZES])
    all_stds = np.array([results[n]["all_stds"][fixed_index] for n in SYSTEM_SIZES])
    fixed_means = np.array([results[n]["fixed_means"][fixed_index] for n in SYSTEM_SIZES])
    fixed_stds = np.array([results[n]["fixed_stds"][fixed_index] for n in SYSTEM_SIZES])
    return {
        "system_sizes": np.array(SYSTEM_SIZES),
        "all_relative_means": all_means / all_means[0],
        "all_relative_stds": all_stds / all_means[0],
        "fixed_relative_means": fixed_means / fixed_means[0],
        "fixed_relative_stds": fixed_stds / fixed_means[0],
    }


def save_individual_figures(
    results: dict[int, dict[str, np.ndarray]],
    size_data: dict[str, np.ndarray],
) -> None:
    """分别保存各类对比图到 figures 目录。"""
    FIGURE_DIR.mkdir(exist_ok=True)
    plot_error_vs_gate_count(
        results,
        "all",
        "All input states",
        FIGURE_DIR / "all_input_states_error_vs_gate_count.png",
    )
    plot_error_vs_gate_count(
        results,
        "fixed",
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
    results: dict[int, dict[str, np.ndarray]],
    error_kind: str,
    title: str,
    output_path: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    """绘制不同系统规模下误差随 qDRIFT 门数变化的曲线。"""
    own_figure = axis is None
    if own_figure:
        _, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    assert axis is not None

    for qubit_count in SYSTEM_SIZES:
        data = results[qubit_count]
        means = data[f"{error_kind}_means"]
        stds = data[f"{error_kind}_stds"]
        gate_counts = data["gate_counts"]
        axis.plot(gate_counts, means, marker="o", label=f"n={qubit_count}")
        axis.fill_between(gate_counts, means - stds, means + stds, alpha=0.12)

    axis.set_title(title)
    axis.set_xlabel("Gate count N")
    axis.set_ylabel("Error")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=False, ncol=2)

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
        _, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
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


def save_combined_figure(
    results: dict[int, dict[str, np.ndarray]],
    size_data: dict[str, np.ndarray],
) -> None:
    """保存与论文 Figure 3 对应的合并对比图。"""
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)

    plot_error_vs_gate_count(results, "all", "All input states", axis=axes[0, 0])
    plot_error_vs_gate_count(results, "fixed", "Fixed input state", axis=axes[0, 1])
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
    results = run_experiments(rng)
    size_data = extract_size_sweep(results)
    save_combined_figure(results, size_data)
    save_individual_figures(results, size_data)


if __name__ == "__main__":
    main()
