# Quantum Computing Product Formula Experiments

本项目复现并扩展论文 `concentration.pdf` 中随机乘积公式的数值实验，重点比较 qDRIFT 随机乘积公式在所有输入态和固定输入态两种误差度量下的表现。

## 文件说明

- `concentration.pdf`：参考论文。
- `reproduce_concentration_experiment.py`：复现论文 Figure 3 风格的一维 Heisenberg 链实验。
- `reproduce_syk_experiment.py`：将同样的 qDRIFT 实验框架应用到 SYK 模型。
- `figures/`：保存所有生成的实验图片。
- `AGENTS.md`：项目协作与代码风格要求。

## 实验设置

Heisenberg 实验使用一维链哈密顿量

$$
H=\frac{1}{n-1}\sum_{i=1}^{n-1}(X_iX_{i+1}+Y_iY_{i+1}+Z_iZ_{i+1}),
$$

其中系统规模为 $n=4,5,6,7,8$ 个量子比特，演化时间为 $t=2$，qDRIFT 强度归一化为 $\lambda=3$。

SYK 实验先按照标准高斯耦合生成

$$
H=\sum_{1\le i<j<k<l\le n}J_{ijkl}\chi_i\chi_j\chi_k\chi_l,
$$

其中

$$
\overline{J_{ijkl}}=0,\qquad \overline{J_{ijkl}^2}=\frac{3!J^2}{n^3}.
$$

随后对每个随机 SYK 实例整体缩放，使 qDRIFT 使用的强度统一为 $\lambda=3$。SYK 图中的 $n=8,10,12,14,16$ 表示 Majorana 数，对应 $4,5,6,7,8$ 个量子比特。

## 运行方式

安装依赖后运行：

```bash
python reproduce_concentration_experiment.py
python reproduce_syk_experiment.py
```

依赖包括：

```bash
numpy
scipy
matplotlib
Pillow
```

## 输出图片

Heisenberg 实验主要输出：

- `figures/figure3_reproduction.png`
- `figures/all_input_states_error_vs_gate_count.png`
- `figures/fixed_input_state_error_vs_gate_count.png`
- `figures/all_input_states_relative_error_vs_system_size.png`
- `figures/fixed_input_state_relative_error_vs_system_size.png`

SYK 实验主要输出：

- `figures/syk_figure3_style_reproduction.png`
- `figures/syk_all_input_states_error_vs_gate_count.png`
- `figures/syk_fixed_input_state_error_vs_gate_count.png`
- `figures/syk_all_input_states_relative_error_vs_majorana_count.png`
- `figures/syk_fixed_input_state_relative_error_vs_majorana_count.png`

相对误差图中的虚线不是理论等式预测。`all input states` 使用实验数据的线性拟合作为趋势参考，`fixed input state` 使用实验均值作为水平参考。
