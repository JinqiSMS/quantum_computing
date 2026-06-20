# Quantum Computing Product Formula Experiments

本项目围绕论文 `concentration.pdf`（Chen--Huang--Kueng--Tropp, *Concentration for random product formulas*）整理理论报告，并复现、扩展其中的 qDRIFT 数值实验。项目仓库地址：`https://github.com/JinqiSMS/quantum_computing.git`。

## 文件说明

- `report_final.tex`：最终课程报告源码，包含理论回顾、数值实验、Appendix、AI 使用报告和参考文献。
- `report_final.pdf`：由 `report_final.tex` 编译得到的最终报告。
- `concentration.pdf`：参考论文原文。
- `reproduce_concentration_experiment.py`：复现论文 Figure 3 风格的一维 Heisenberg 链实验。
- `reproduce_syk_experiment.py`：将同样的 qDRIFT 实验框架应用到归一化 SYK 模型。
- `figures/`：保存实验图片和原论文 Figure 3 的裁剪图。
- `AGENTS.md`：项目协作与代码风格要求。

## 报告内容

`report_final.tex` 当前包括：

- qDRIFT 理论背景与主要定理。
- 薛定谔方程和 qDRIFT 算法流程。
- Heisenberg 数值实验，并与原论文 Figure 3 对比。
- SYK 数值实验。
- Appendix：关于原论文 causality 条件表述的讨论、小组分工和 AI 使用报告。

## 实验设置

Heisenberg 实验使用归一化一维链哈密顿量

$$
H=\frac{1}{n-1}\sum_{i=1}^{n-1}(X_iX_{i+1}+Y_iY_{i+1}+Z_iZ_{i+1}),
$$

其中系统规模为 $n=4,5,6,7,8$ 个量子比特，演化时间为 $t=2$，并且 $\lambda=3$。

SYK 实验先按照标准高斯耦合生成

$$
H=\sum_{1\le i<j<k<l\le n}J_{ijkl}\chi_i\chi_j\chi_k\chi_l,
$$

其中

$$
\overline{J_{ijkl}}=0,\qquad \overline{J_{ijkl}^2}=\frac{3!J^2}{n^3}.
$$

随后对每个随机 SYK 实例整体缩放，使 qDRIFT 使用的强度统一为 $\lambda=3$。SYK 图中的 $n=8,10,12,14,16$ 表示 Majorana 数，对应 $4,5,6,7,8$ 个量子比特。

## 运行实验

```bash
python reproduce_concentration_experiment.py
python reproduce_syk_experiment.py
```

Python 依赖包括：

```bash
numpy
scipy
matplotlib
Pillow
```

## 编译报告

```bash
pdflatex -interaction=nonstopmode -halt-on-error report_final.tex
pdflatex -interaction=nonstopmode -halt-on-error report_final.tex
```

报告使用的 LaTeX 宏包包括 `amsmath`、`amssymb`、`mathtools`、`graphicx`、`subcaption`、`float`、`algorithm`、`algpseudocode` 和 `hyperref`。

## 输出图片

Heisenberg 实验输出：

- `figures/original_paper_figure3.png`：从原论文裁剪出的 Figure 3。
- `figures/figure3_reproduction.png`：本项目复现和扩展的 Heisenberg 合并图。
- `figures/all_input_states_error_vs_gate_count.png`
- `figures/fixed_input_state_error_vs_gate_count.png`
- `figures/all_input_states_relative_error_vs_system_size.png`
- `figures/fixed_input_state_relative_error_vs_system_size.png`

SYK 实验输出：

- `figures/syk_figure3_style_reproduction.png`
- `figures/syk_all_input_states_error_vs_gate_count.png`
- `figures/syk_fixed_input_state_error_vs_gate_count.png`
- `figures/syk_all_input_states_relative_error_vs_majorana_count.png`
- `figures/syk_fixed_input_state_relative_error_vs_majorana_count.png`

相对误差图中的虚线不是理论等式预测。`all input states` 使用实验数据的线性拟合作为趋势参考，`fixed input state` 使用实验均值作为水平参考。
