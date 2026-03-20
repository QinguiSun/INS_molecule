# Real-time interpretation of neutron vibrational spectra with symmetry-equivariant Hessian matrix prediction

Predict the Hessian and INS of organic molecules, implemented with JAX.

# Installation
To use the package, conda environment is recommended. The suggested environment file, env_ins_mol.yml, is included in the package. To create the environment, you can use the following command:

```bash
git clone https://github.com/maplewen4/INS_molecule.git
cd INS_molecule
# conda env create -f env_ins_mol.yml
```

`env_ins_mol.yml`已经更新。确保**完全锁死 + CUDA12 GPU 可复现 + NequIP-JAX 稳定版**。

设计原则：

* ✅ Python 3.11（与你现有一致）
* ✅ JAX 0.4.28 + CUDA12（当前最稳定区间）
* ✅ Flax / Optax / Chex 版本严格匹配
* ✅ RDKit 走 conda-forge（避免 ABI 问题）
* ✅ 只保留 JAX GPU（不混 PyTorch GPU）
* ✅ 不在 yml 里写 pip flags（镜像放 pip.conf）
* ✅ NVIDIA runtime 显式锁版本（避免未来 wheel 变动）


### ⚙️ 使用方式

#### 1️⃣ 先配置 pip 镜像（非常重要）

创建：

```bash
mkdir -p ~/.pip
nano ~/.pip/pip.conf
```

写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 1000
retries = 20
```

不要把 `-i` 写进 yml。

---

#### 2️⃣ 创建环境

```bash
conda env create -f env_ins_mol.yml
conda activate nequipjax
```

---

### 🧪 验证 GPU

```bash
nvidia-smi
```

然后：

```bash
python -c "import jax; print(jax.devices())"
```

应该看到：

```
[GpuDevice(id=0)]
```

---

### 🔬 为什么这个版本是“稳定区间”

| 组件     | 版本     |
| ------ | ------ |
| jax    | 0.4.28 |
| jaxlib | 0.4.28 |
| flax   | 0.8.0  |
| optax  | 0.1.8  |
| chex   | 0.1.85 |

这是当前（2024–2025）：

> 最后一个广泛兼容 + 不触发 API break 的 JAX 区间

0.4.29+ 开始有内部结构变化
0.4.30+ flax 有时不兼容

---

### 🧠 如果你未来要：

* 做大规模 DFT surrogate
* 训练 NequIP-JAX
* 多 GPU
* NCCL 分布式

这个版本可以稳定运行 6–12 个月。

---



# Usage

To train and test the machine learning model, you may use the associated dataset, https://doi.org/10.5281/zenodo.14796532. To build the training dataset, you can run `load_data.py` with proper path to the raw files. To train a model, you could run `train_hessian.py`. The architecture of the model can be modified within `Nequip.py`. To use the pre-trained model to predict the Hessian matrices and inelastic neutron scattering spectra, please use `predict_hessian_and_ins_from_strucure.py`. The pretrained model is also included in the Zenodo link.

## Citation

```
@misc{han_real-time_2025,
	title = {Real-time interpretation of neutron vibrational spectra with symmetry-equivariant {Hessian} matrix prediction},
	url = {http://arxiv.org/abs/2502.13070},
	doi = {10.48550/arXiv.2502.13070},
	publisher = {arXiv},
	author = {Han, Bowen and Zhang, Pei and Mehta, Kshitij and Pasini, Massimiliano Lupo and Li, Mingda and Cheng, Yongqiang},
	month = feb,
	year = {2025},
	note = {arXiv:2502.13070 [physics]},
	keywords = {Physics - Chemical Physics, Physics - Computational Physics},
}
```
