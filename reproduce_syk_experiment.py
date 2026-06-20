"""按照标准 SYK setting 复现 qDRIFT 随机乘积公式实验。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, norm


TIME = 2.0
RUN_COUNT = 50
FIXED_GATE_COUNT = 160
RANDOM_SEED = 20260620
COUPLING_J = 1.0
TARGET_LAMBDA = 3.0
QUBIT_COUNTS = [4, 5, 6, 7, 8]
MAJORANA_COUNTS = [2 * qubit_count for qubit_count in QUBIT_COUNTS]
GATE_COUNTS = [20, 40, 80, 120, 160, 240, 320]
REFERENCE_MAJORANA_COUNT = MAJORANA_COUNTS[0]
FIGURE_DIR = Path("figures")


@dataclass(frozen=True)
class PauliString:
    """保存 Pauli 字符串的二进制表示。"""

    x_mask: int
    z_mask: int
    coefficient: complex


@dataclass(frozen=True)
class SykTerm:
    """保存一个四 Majorana 相互作用项。"""

    x_mask: int
    z_mask: int
    coefficient: complex
    coupling: float
    sign: float


@dataclass
class SykData:
    """保存一个 SYK 系统的哈密顿量、目标演化和 qDRIFT 抽样项。"""

    qubit_count: int
    majorana_count: int
    hamiltonian: np.ndarray
    target: np.ndarray
    terms: list[SykTerm]
    lambda_value: float


def popcount(value: int) -> int:
    """计算整数二进制展开中 1 的个数。"""
    return int(value.bit_count())


def bit_position(qubit_index: int, qubit_count: int) -> int:
    """把量子比特编号转换为矩阵基态中的二进制位编号。"""
    return qubit_count - 1 - qubit_index


def multiply_pauli_strings(left: PauliString, right: PauliString) -> PauliString:
    """计算两个 Pauli 字符串的乘积。"""
    sign = -1 if popcount(left.z_mask & right.x_mask) % 2 else 1
    return PauliString(
        x_mask=left.x_mask ^ right.x_mask,
        z_mask=left.z_mask ^ right.z_mask,
        coefficient=left.coefficient * right.coefficient * sign,
    )


def make_majorana_strings(qubit_count: int) -> list[PauliString]:
    """构造带 Jordan-Wigner 字符串的 Majorana 算符，省略公共系数。"""
    majoranas: list[PauliString] = []
    for site_index in range(qubit_count):
        z_prefix = 0
        for prefix_index in range(site_index):
            z_prefix ^= 1 << bit_position(prefix_index, qubit_count)

        current_bit = 1 << bit_position(site_index, qubit_count)
        majoranas.append(PauliString(x_mask=current_bit, z_mask=z_prefix, coefficient=1.0 + 0.0j))
        majoranas.append(PauliString(x_mask=current_bit, z_mask=z_prefix ^ current_bit, coefficient=1.0j))
    return majoranas


def make_syk_terms(qubit_count: int, rng: np.random.Generator) -> list[SykTerm]:
    """按照标准 SYK 高斯分布生成所有四 Majorana 相互作用项。"""
    majorana_count = 2 * qubit_count
    coupling_std = np.sqrt(6.0 * COUPLING_J**2 / majorana_count**3)
    majoranas = make_majorana_strings(qubit_count)
    terms: list[SykTerm] = []

    for indices in combinations(range(majorana_count), 4):
        pauli_string = PauliString(x_mask=0, z_mask=0, coefficient=1.0 + 0.0j)
        for majorana_index in indices:
            pauli_string = multiply_pauli_strings(pauli_string, majoranas[majorana_index])
        coupling = float(rng.normal(loc=0.0, scale=coupling_std))
        sign = 1.0 if coupling >= 0 else -1.0
        terms.append(
            SykTerm(
                x_mask=pauli_string.x_mask,
                z_mask=pauli_string.z_mask,
                coefficient=pauli_string.coefficient,
                coupling=coupling,
                sign=sign,
            )
        )
    return terms


def normalize_terms_to_target_lambda(terms: list[SykTerm]) -> list[SykTerm]:
    """把 SYK 哈密顿量整体缩放到固定的 qDRIFT 强度。"""
    raw_lambda = sum(abs(term.coupling) / 4.0 for term in terms)
    if raw_lambda == 0:
        raise ValueError("SYK 随机耦合的强度为零，无法归一化。")
    scale = TARGET_LAMBDA / raw_lambda
    return [
        SykTerm(
            x_mask=term.x_mask,
            z_mask=term.z_mask,
            coefficient=term.coefficient,
            coupling=term.coupling * scale,
            sign=term.sign,
        )
        for term in terms
    ]


def pauli_row_phase(term: SykTerm, dimension: int) -> np.ndarray:
    """计算 Pauli 字符串左乘矩阵或作用态向量时的行相位。"""
    indices = np.arange(dimension, dtype=np.int64)
    source_indices = indices ^ term.x_mask
    parities = np.fromiter(
        (popcount(int(term.z_mask & source_index)) % 2 for source_index in source_indices),
        dtype=np.int8,
        count=dimension,
    )
    signs = 1 - 2 * parities
    return term.coefficient * signs


def apply_pauli_to_matrix(matrix: np.ndarray, term: SykTerm, row_phase: np.ndarray) -> np.ndarray:
    """把归一化 Pauli 字符串左乘到完整矩阵上。"""
    source_indices = np.arange(matrix.shape[0], dtype=np.int64) ^ term.x_mask
    return row_phase[:, None] * matrix[source_indices, :]


def apply_pauli_to_state(state: np.ndarray, term: SykTerm, row_phase: np.ndarray) -> np.ndarray:
    """把归一化 Pauli 字符串作用到态向量上。"""
    source_indices = np.arange(state.shape[0], dtype=np.int64) ^ term.x_mask
    return row_phase * state[source_indices]


def build_syk_data(qubit_count: int, rng: np.random.Generator) -> SykData:
    """构造一个给定量子比特数的 SYK 哈密顿量和目标演化。"""
    terms = normalize_terms_to_target_lambda(make_syk_terms(qubit_count, rng))
    dimension = 2**qubit_count
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    identity = np.eye(dimension, dtype=complex)

    for term in terms:
        row_phase = pauli_row_phase(term, dimension)
        normalized_pauli = apply_pauli_to_matrix(identity, term, row_phase)
        hamiltonian += (term.coupling / 4.0) * normalized_pauli

    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2.0
    target = expm(-1j * TIME * hamiltonian)
    lambda_value = sum(abs(term.coupling) / 4.0 for term in terms)
    return SykData(
        qubit_count=qubit_count,
        majorana_count=2 * qubit_count,
        hamiltonian=hamiltonian,
        target=target,
        terms=terms,
        lambda_value=lambda_value,
    )


def sample_product_formula(
    data: SykData,
    gate_count: int,
    sampled_indices: np.ndarray,
    row_phases: list[np.ndarray],
) -> np.ndarray:
    """按 qDRIFT 规则构造随机乘积公式的完整矩阵。"""
    dimension = data.target.shape[0]
    product = np.eye(dimension, dtype=complex)
    angle = TIME * data.lambda_value / gate_count
    cosine = np.cos(angle)
    sine = np.sin(angle)

    for sampled_index in sampled_indices:
        term = data.terms[int(sampled_index)]
        pauli_product = apply_pauli_to_matrix(product, term, row_phases[int(sampled_index)])
        product = cosine * product - 1j * term.sign * sine * pauli_product
    return product


def apply_product_formula_to_state(
    data: SykData,
    gate_count: int,
    sampled_indices: np.ndarray,
    row_phases: list[np.ndarray],
    state: np.ndarray,
) -> np.ndarray:
    """按 qDRIFT 规则把随机乘积公式作用到固定态向量。"""
    evolved_state = state.copy()
    angle = TIME * data.lambda_value / gate_count
    cosine = np.cos(angle)
    sine = np.sin(angle)

    for sampled_index in sampled_indices:
        term = data.terms[int(sampled_index)]
        pauli_state = apply_pauli_to_state(evolved_state, term, row_phases[int(sampled_index)])
        evolved_state = cosine * evolved_state - 1j * term.sign * sine * pauli_state
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
    data: SykData,
    gate_count: int,
    run_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """估计所有输入态误差和固定输入态误差。"""
    dimension = data.target.shape[0]
    row_phases = [pauli_row_phase(term, dimension) for term in data.terms]
    probabilities = np.array([abs(term.coupling) / 4.0 for term in data.terms]) / data.lambda_value
    fixed_state = sample_product_state(data.qubit_count, rng)
    target_state = data.target @ fixed_state
    all_state_errors = np.empty(run_count)
    fixed_state_errors = np.empty(run_count)

    for run_index in range(run_count):
        sampled_indices = rng.choice(len(data.terms), size=gate_count, p=probabilities)
        product = sample_product_formula(data, gate_count, sampled_indices, row_phases)
        product_state = apply_product_formula_to_state(data, gate_count, sampled_indices, row_phases, fixed_state)
        all_state_errors[run_index] = norm(data.target - product, 2)
        fixed_state_errors[run_index] = norm(target_state - product_state)
    return all_state_errors, fixed_state_errors


def summarize(values: np.ndarray) -> tuple[float, float]:
    """返回样本均值和样本标准差。"""
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_experiments(rng: np.random.Generator) -> dict[int, dict[str, np.ndarray]]:
    """对所有系统尺寸和所有门数运行 SYK qDRIFT 实验。"""
    results: dict[int, dict[str, np.ndarray]] = {}
    for qubit_count in QUBIT_COUNTS:
        data = build_syk_data(qubit_count, rng)
        all_means = []
        all_stds = []
        fixed_means = []
        fixed_stds = []
        print(
            f"Majorana n={data.majorana_count}, qubits={qubit_count}, "
            f"terms={len(data.terms)}, normalized lambda={data.lambda_value:.4g}"
        )
        for gate_count in GATE_COUNTS:
            all_errors, fixed_errors = estimate_errors(data, gate_count, RUN_COUNT, rng)
            all_mean, all_std = summarize(all_errors)
            fixed_mean, fixed_std = summarize(fixed_errors)
            all_means.append(all_mean)
            all_stds.append(all_std)
            fixed_means.append(fixed_mean)
            fixed_stds.append(fixed_std)
            print(
                f"Majorana n={data.majorana_count}, N={gate_count}: "
                f"all={all_mean:.4g}, fixed={fixed_mean:.4g}"
            )

        results[data.majorana_count] = {
            "gate_counts": np.array(GATE_COUNTS),
            "all_means": np.array(all_means),
            "all_stds": np.array(all_stds),
            "fixed_means": np.array(fixed_means),
            "fixed_stds": np.array(fixed_stds),
            "lambda_value": np.array([data.lambda_value]),
        }
    return results


def extract_size_sweep(results: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """从全部实验结果中提取固定门数下的系统规模扫描。"""
    fixed_index = GATE_COUNTS.index(FIXED_GATE_COUNT)
    all_means = np.array([results[n]["all_means"][fixed_index] for n in MAJORANA_COUNTS])
    all_stds = np.array([results[n]["all_stds"][fixed_index] for n in MAJORANA_COUNTS])
    fixed_means = np.array([results[n]["fixed_means"][fixed_index] for n in MAJORANA_COUNTS])
    fixed_stds = np.array([results[n]["fixed_stds"][fixed_index] for n in MAJORANA_COUNTS])
    return {
        "majorana_counts": np.array(MAJORANA_COUNTS),
        "all_relative_means": all_means / all_means[0],
        "all_relative_stds": all_stds / all_means[0],
        "fixed_relative_means": fixed_means / fixed_means[0],
        "fixed_relative_stds": fixed_stds / fixed_means[0],
    }


def plot_error_vs_gate_count(
    results: dict[int, dict[str, np.ndarray]],
    error_kind: str,
    title: str,
    output_path: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    """绘制不同 Majorana 数下误差随 qDRIFT 门数变化的曲线。"""
    own_figure = axis is None
    if own_figure:
        _, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    assert axis is not None

    for majorana_count in MAJORANA_COUNTS:
        data = results[majorana_count]
        means = data[f"{error_kind}_means"]
        stds = data[f"{error_kind}_stds"]
        gate_counts = data["gate_counts"]
        axis.plot(gate_counts, means, marker="o", label=f"n={majorana_count}")
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
    majorana_counts: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    theory_line: np.ndarray,
    title: str,
    output_path: Path | None = None,
    axis: plt.Axes | None = None,
) -> None:
    """绘制固定门数下归一化误差随 Majorana 数变化的曲线。"""
    own_figure = axis is None
    if own_figure:
        _, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    assert axis is not None

    axis.plot(majorana_counts, means, marker="o", color="#d62728", label="simulation")
    axis.fill_between(majorana_counts, means - stds, means + stds, color="#d62728", alpha=0.2)
    if "all input" in title.lower():
        fit_coefficients = np.polyfit(majorana_counts, means, deg=1)
        reference_line = np.polyval(fit_coefficients, majorana_counts)
        reference_label = "empirical linear fit"
    else:
        reference_line = np.full_like(majorana_counts, np.mean(means), dtype=float)
        reference_label = "empirical mean"
    axis.plot(majorana_counts, reference_line, linestyle=":", color="black", label=reference_label)
    axis.set_title(title)
    axis.set_xlabel("Number of Majoranas n")
    axis.set_ylabel(r"Relative error $\epsilon_n / \epsilon_8$")
    axis.set_xticks(majorana_counts)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)

    if own_figure and output_path is not None:
        plt.savefig(output_path, dpi=300)
        plt.close()


def save_figures(results: dict[int, dict[str, np.ndarray]], size_data: dict[str, np.ndarray]) -> None:
    """保存 SYK 实验的合并图和单独子图。"""
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)

    plot_error_vs_gate_count(results, "all", "SYK: all input states", axis=axes[0, 0])
    plot_error_vs_gate_count(results, "fixed", "SYK: fixed input state", axis=axes[0, 1])
    plot_relative_error_vs_size(
        size_data["majorana_counts"],
        size_data["all_relative_means"],
        size_data["all_relative_stds"],
        size_data["majorana_counts"] / REFERENCE_MAJORANA_COUNT,
        f"SYK: all input states, N={FIXED_GATE_COUNT}",
        axis=axes[1, 0],
    )
    plot_relative_error_vs_size(
        size_data["majorana_counts"],
        size_data["fixed_relative_means"],
        size_data["fixed_relative_stds"],
        np.ones_like(size_data["majorana_counts"], dtype=float),
        f"SYK: fixed input state, N={FIXED_GATE_COUNT}",
        axis=axes[1, 1],
    )

    figure.suptitle("qDRIFT numerical experiments for the SYK model", fontsize=14)
    figure.savefig(FIGURE_DIR / "syk_figure3_style_reproduction.png", dpi=300)
    plt.close(figure)

    plot_error_vs_gate_count(
        results,
        "all",
        "SYK: all input states",
        FIGURE_DIR / "syk_all_input_states_error_vs_gate_count.png",
    )
    plot_error_vs_gate_count(
        results,
        "fixed",
        "SYK: fixed input state",
        FIGURE_DIR / "syk_fixed_input_state_error_vs_gate_count.png",
    )
    plot_relative_error_vs_size(
        size_data["majorana_counts"],
        size_data["all_relative_means"],
        size_data["all_relative_stds"],
        size_data["majorana_counts"] / REFERENCE_MAJORANA_COUNT,
        "SYK: all input states",
        FIGURE_DIR / "syk_all_input_states_relative_error_vs_majorana_count.png",
    )
    plot_relative_error_vs_size(
        size_data["majorana_counts"],
        size_data["fixed_relative_means"],
        size_data["fixed_relative_stds"],
        np.ones_like(size_data["majorana_counts"], dtype=float),
        "SYK: fixed input state",
        FIGURE_DIR / "syk_fixed_input_state_relative_error_vs_majorana_count.png",
    )


def main() -> None:
    """运行 SYK qDRIFT 实验并保存图片。"""
    rng = np.random.default_rng(RANDOM_SEED)
    results = run_experiments(rng)
    size_data = extract_size_sweep(results)
    save_figures(results, size_data)


if __name__ == "__main__":
    main()




