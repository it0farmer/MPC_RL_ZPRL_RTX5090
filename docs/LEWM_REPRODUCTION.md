# LeWorldModel 复现说明

本工程包含两层复现：

## A. 方法核心复现（本仓库可直接运行）

`mpcrl/lewm.py` 实现：

- raw RGB frame -> latent encoder
- action-conditioned next-latent predictor
- next-embedding MSE
- SIGReg（随机投影 + Epps-Pulley/经验特征函数式高斯匹配）
- 无 EMA teacher
- 无 stop-gradient target
- latent CEM planner

训练目标严格保持两项结构：

`L = L_pred + lambda * SIGReg`

MuJoCo locomotion 没有论文中那种明确 goal image，因此 `experiments/train_lewm.py` 在冻结 LeWM 后单独训练 reward probe，再用 CEM 最大化预测回报。该 reward probe 属于 MuJoCo 适配，不应被写成 LeWM 原论文训练目标。

## B. 严格论文数值复现（建议官方仓库）

LeWorldModel 官方仓库（2026）公开了数据、checkpoints 和基于 `stable-worldmodel` 的评估路径。若论文需要声称“严格复现原文数值”，应使用官方环境（如 two-room / Push-T / cube / reacher）、官方数据和 checkpoint，而不是用本仓库的 MuJoCo locomotion 数值替代原文表格。

本工程应作为“与提出方法统一接口下的工程复现/适配实验”，官方仓库用于“原论文 benchmark 数值核验”。
