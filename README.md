# MPC + LeWorldModel + Planning-Residual RL（论文实验级工程）

本工程实现三个可区分的研究组件：

1. **Ensemble World Model + CEM-MPC**：用于 MuJoCo 状态空间控制的可训练基线。
2. **Planning-level Residual SAC**：RL 不从零生成动作，而对 MPC 的短时 action chunk 做残差修正；可选世界模型不确定性自适应门控与 MPC 一致性正则。
3. **LeWorldModel 复现路径**：提供 LeWM 的两项核心训练目标（next-embedding prediction + SIGReg）及 latent CEM-MPC 适配代码；另附官方仓库对齐说明。

> 说明：ZPRL 原论文建立在预训练生成式机器人策略（flow-matching policy）上。本工程的 `zprl_style` 是用于 MuJoCo 公平对比的**瓶颈潜空间扰动代理基线**，不是对 ZPRL 机器人实验的等价复现。这样可避免把“世界模型潜空间”和“冻结基础策略瓶颈潜空间”混为一谈。

## 目录

```text
MPC_RL_ZPRL_RTX5090/
├─ configs/                   # 环境/训练/消融配置
├─ mpcrl/
│  ├─ world_model.py          # Ensemble dynamics + reward model
│  ├─ cem.py                  # 真正的 CEM-MPC
│  ├─ sac.py                  # Residual SAC
│  ├─ gate.py                 # 动态归一化不确定性门控 + Residual Ramp
  ├─ plan_cache.py           # next-state MPC 规划缓存
│  ├─ zprl_style.py           # 瓶颈潜空间扰动代理基线
│  ├─ lewm.py                 # LeWM core + SIGReg + reward probe
│  ├─ lewm_planner.py         # latent CEM planner
│  ├─ replay.py               # replay buffers
│  ├─ metrics.py              # 论文指标
│  └─ plotting.py             # 论文绘图
├─ experiments/
│  ├─ train.py                # MPC / action residual / planning residual
│  ├─ train_zprl_style.py     # ZPRL-style proxy
│  ├─ train_lewm.py           # LeWM objective-level reproduction
│  ├─ run_suite.py            # 多任务多 seed 批量实验
│  ├─ quick_diagnostic.py     # 10k 三方法快速诊断
│  ├─ ablation.py             # 消融实验矩阵
│  └─ aggregate.py            # CSV 聚合
├─ scripts/
│  ├─ smoke_test.sh
│  ├─ run_diagnostic_10k.sh
│  ├─ run_main_suite.sh
│  └─ run_ablation.sh
├─ tests/
└─ docs/
```

## RTX 5090 环境配置

本分支提供 `configs/rtx5090/` 专用配置。RTX 5090 属于 NVIDIA Blackwell（SM 12.0），建议使用 **PyTorch CUDA 12.8+** 官方构建。专用配置默认启用 BF16 自动混合精度、TF32、cuDNN benchmark，并针对 32 GB 显存增大 world model、SAC batch 和 CEM 并行候选数。

### 1. 创建 Conda 环境

```bash
conda create -n mpc_rl_zprl_5090 python=3.11 -y
conda activate mpc_rl_zprl_5090
```

也可以使用工程中的 `environment.yml` 创建基础环境：

```bash
conda env create -f environment.yml
conda activate mpc_rl_zprl_5090
```

### 2. 安装 RTX 5090 可用的 PyTorch

优先安装 CUDA 12.8 官方 wheel：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

随后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

> 不建议给 RTX 5090 安装旧的 `cu121/cu124/cu126` PyTorch wheel。工程要求 `torch>=2.7`，并以 CUDA 12.8+ 构建为目标。

### 3. 检查 RTX 5090 是否正确工作

```bash
nvidia-smi
python scripts/check_rtx5090.py
```

正常情况下应看到类似：

```text
GPU: NVIDIA GeForce RTX 5090
Compute capability: 12.0
VRAM GiB: 31.x
BF16 supported: True
RTX 5090 environment check: PASS
```

### 4. 一键安装（Ubuntu / bash）

若本机已经安装 Conda，可直接执行：

```bash
bash scripts/install_rtx5090.sh
conda activate mpc_rl_zprl_5090
```

该脚本依次创建 Conda 环境、安装 CUDA 12.8 PyTorch、安装工程依赖并执行 GPU 自检。

## 最小训练

```bash
python -m experiments.train --config configs/halfcheetah.yaml --method mpc_only --steps 30000
python -m experiments.train --config configs/halfcheetah.yaml --method action_residual --steps 30000
python -m experiments.train --config configs/halfcheetah.yaml --method planning_residual --steps 30000
# 对保存的 run 做确定性评估：
python -m experiments.evaluate --run-dir runs/<run_name> --episodes 10
```

## RTX 5090 推荐运行顺序

先进行最小 GPU 验证：

```bash
conda activate mpc_rl_zprl_5090
python scripts/check_rtx5090.py
python -m unittest discover -s tests -v
python -m experiments.train \
  --config configs/rtx5090/halfcheetah.yaml \
  --method planning_residual \
  --steps 2000 \
  --run-name rtx5090_smoke
```

确认无报错后，运行单个正式实验：

```bash
python -m experiments.train \
  --config configs/rtx5090/halfcheetah.yaml \
  --method planning_residual \
  --steps 100000 \
  --seed 0
```

四种方法分别为：

```text
mpc_only
action_residual
zprl_style
planning_residual
```

RTX 5090 主配置的关键参数：

- `precision: bf16`
- World Model：7-member ensemble，hidden=512
- SAC：hidden=512，batch=1024
- Hopper/Walker2d：CEM candidates=2048
- HalfCheetah：CEM candidates=4096
- TF32 与 cuDNN benchmark 默认开启

这些配置用于提高单卡吞吐。若显存被其他进程占用，可优先将 `mpc.candidates` 降至 2048，再将 `train.batch_size` 降至 512。

### RTX 5090 稳定训练改进

当前 `configs/rtx5090/*.yaml` 已针对 2000 步 smoke test 中发现的 Gate 饱和问题进一步调整：

- `gate_mode: running_zscore`：ensemble uncertainty 先按运行中的 EMA 均值/方差标准化，再进入 Sigmoid，不再依赖固定绝对阈值。
- `gate_min/gate_max: 0.05/0.95`：避免门控长期完全饱和到 0 或 1。
- `ramp_steps: 5000`：RL 开始更新后，Residual 修正强度由 0 线性增加至完整强度，避免随机初始策略立即破坏 MPC。
- `residual_scale: 0.10`：RTX 5090 主配置先采用保守残差幅度，后续根据 10k 诊断结果再调。
- `PlanCache`：Residual SAC 为 next-state context 计算出的 MPC plan 会在下一环境步复用，避免对同一状态重复规划。

训练日志新增：

- `adaptive_gate`：仅由标准化 uncertainty 得到的门控值；
- `residual_ramp`：当前 Residual Ramp 系数；
- `gate`：实际生效的 `adaptive_gate × residual_ramp`；
- `gate_z`：当前 uncertainty 的标准化 z-score；
- `effective_residual_norm`：真正加到 MPC action chunk 上的有效残差范数；
- `mpc_cache_hit_rate`：MPC next-plan 缓存命中率。

## 10k 三方法快速诊断

在正式 100k 实验前，建议先运行 10k 快速诊断：

```bash
conda activate mpc_rl_zprl_5090
python -m experiments.quick_diagnostic \
  --config configs/rtx5090/halfcheetah.yaml \
  --steps 10000 \
  --seeds 0
```

也可以直接执行：

```bash
bash scripts/run_diagnostic_10k.sh
```

该命令依次运行：

```text
mpc_only
action_residual
planning_residual
```

结果保存在：

```text
results/diagnostic_10k/<时间戳>/
├─ episodes_all.csv
├─ summary_per_seed.csv
├─ summary_mean_std.csv
└─ figures/
   ├─ episode_return.png
   ├─ mpc_ms.png
   ├─ prediction_mse.png
   ├─ action_d1.png
   ├─ effective_residual_norm.png
   └─ gate.png
```

正式 100k 实验前重点检查：Planning Residual 的回报是否开始超过 MPC；`gate` 是否脱离长期接近 1 的饱和状态；`effective_residual_norm` 是否随 Ramp 平稳增加；`mpc_cache_hit_rate` 在非终止步应接近 1；动作一阶/二阶平滑度不应明显恶化。

## 论文主实验

RTX 5090 推荐：

```bash
python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml
# 如需同时运行计算量更大的 LeWorldModel-MPC：
python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml --include-lewm
python -m experiments.aggregate --root runs --out results/summary.csv
python -m experiments.plot_results --summary results/summary.csv --outdir results/figures
```

原始保守配置仍保留在 `configs/*.yaml`，用于硬件无关的对照和消融复现。
