# SPAVR

SPAVR 是一个用于具身机器人候选动作评估的研究原型。当前模型基于 action-conditioned C-JEPA：输入近期观测、机器人状态和一段候选末端动作，在对象隐空间中预测未来，并输出整段动作的回报分位数 `q10 / q50 / q90`。

> 项目仍在开发中。模型主体、failure 数据预处理、VideoSAUR 预训练、slot 提取和从 frozen slots 训练 SPAVR 的入口已经存在，真机闭环尚未完成。

## 模型概览

```text
pixels 或预抽取 object_slots
              │
object slots + proprio + candidate actions
              │
action-conditioned C-JEPA rollout
              │
rollout_trace (B,R,P,S+2,D)
              │
full-trace Transformer quantile head
              │
        q10 / q50 / q90
```

- `q50`：候选动作的典型整体回报，作为默认评分。
- `q10`：低回报尾部，用于风险敏感决策。
- `q90-q10`：条件回报分布宽度。

模型对整段动作只输出一组分位数，不输出逐时刻奖励，也不学习成功 / 失败分类标签。

### C-JEPA rollout

模型支持两个视觉入口：`pixels` 经过已经预训练并冻结的 VideoSAUR；`object_slots` 跳过视觉前端，用于预抽取 slots 的高效训练。像素入口必须提供 VideoSAUR checkpoint，代码不再允许用随机 Slot Attention 直接正式训练。

每个时间步的 token 为：

```text
[object_1, ..., object_S, proprio, action]
```

设历史长度为 `H`、动作长度为 `L`、每轮预测 `P` 步：

```text
R = ceil((L-H)/P) + 1
rollout_trace = (B,R,P,S+2,D)
```

后续 action block 在一轮预测后写入 future action slot，从下一轮开始影响预测；动作耗尽后再预测一次。

### 分位数头

当前代码读取所有轮次的 object 和 predicted proprio tokens，加入 type/time/round embeddings，经 full-trace Transformer 和三个 learned queries 输出分位数。预测 action slot 不进入输出头。

```text
q50 = median_head(h50)
q10 = q50 - softplus(lower_gap_head(h10))
q90 = q50 + softplus(upper_gap_head(h90))
```

因此始终满足 `q10 <= q50 <= q90`。

## 数据契约

机械臂数据统一定义为：

```text
proprio_t = [pose_t^ee, gripper_(t-1)]
action_t  = [Δpose_t^ee, gripper_command_t]
```

`proprio` 和 `action` 均为 7 维：前 6 维使用一致的坐标、单位和 rotation-vector 姿态参数化，第 7 维为夹爪。预处理会读取所选外部视频视角的逐帧 `T_cw`，将二者统一到对应的 OpenCV 相机坐标系；原始 `tcp_pose` 四元数按数据集的 `wxyz` 顺序解析。

```python
batch = {
    # 二选一
    "pixels":       ...,  # (B,T,3,H_img,W_img)
    "object_slots": ...,  # (B,T,S,D)
    "proprio":      ...,  # (B,T,7)
    "action":        ...,  # (B,L,7)
    "return_to_go": ...,  # (B,)，仅训练需要
}
```

- 普通评分要求 `T>=H`、`L>=H`。
- 后果损失训练要求 action horizon 满足 `L>=H+P`。
- 首轮训练配置：`H=3, P=2, L=5, S=6, D=128`。
- 输出形状：`(B,3)`，顺序为 `[q10,q50,q90]`。

处理后 episode 的视觉、proprio、action、reward 和 timestamp 具有相同时间长度。Dataset 为长度为 `L` 的候选计划计算整体标签：

```text
return_to_go = G_t^L
              = Σ(k=0...L-1) γ^k r_(t+k) / Σ(k=0...L-1) γ^k
```

不需要人工标注三个分位数。数据应覆盖高、中、低回报轨迹，并尽量包含相似状态下的不同候选动作。

### 数据预处理

```bash
conda activate spavr
python -m pip install -e .

spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /data7/yuqingchi/Code/SPAVR/processed_spavr_test \
  --ambiguous-view-policy skip \
  --workers 4
```

脚本会生成时间长度完全一致的外部视频、7 维 proprio、7 维 action、reward 和 timestamp，并默认只保留处理后帧数位于第 1 到第 99 百分位的轨迹。完整格式见 [`docs/SPAVR数据集预处理与训练接口.md`](docs/SPAVR数据集预处理与训练接口.md)。

## 三阶段训练

### 1. 分阶段训练 VideoSAUR 对象编码器

DINOv2 保持冻结，只训练投影层、Slot Attention、时序处理器和特征 decoder。损失采用 C-JEPA 官方 VideoSAUR 配置中的 DINO 特征重建和时间相似性监督。

正式路线从第一阶段起统一使用 6 slots，共 45k steps，分成三个独立阶段：

```text
stage 1: Push-T checkpoint -> 6-slot SPAVR 外观域适配  20k steps, lr=5e-5,   temperature=0.25
stage 2: stage-1 best.pt    -> temporal target 逐步锐化  15k steps, lr=2.5e-5, temperature=0.10
stage 3: stage-2 best.pt    -> 低学习率最终精修       10k steps, lr=1e-5,   temperature=0.05
```

每个阶段从上一阶段的 `best.pt` 加载模型权重，但重新创建 optimizer 和 scheduler。阶段之间不要使用 `--resume`；`--resume` 只用于恢复当前阶段的中断训练。

```bash
# 先验证第一阶段的完整训练链路
spavr-train-videosaur --config configs/videosaur_stages/01_domain_adapt_t025.yaml --smoke-test

# 按顺序执行三个 VideoSAUR 阶段
spavr-train-videosaur --config configs/videosaur_stages/01_domain_adapt_t025.yaml
spavr-train-videosaur --config configs/videosaur_stages/02_temporal_sharpen_t010.yaml
spavr-train-videosaur --config configs/videosaur_stages/03_final_refine_t005.yaml

# 示例：恢复被中断的第二阶段
spavr-train-videosaur \
  --config configs/videosaur_stages/02_temporal_sharpen_t010.yaml \
  --resume outputs/videosaur_multistage/stage2_temporal_sharpen_s6_t010/last.pt
```

训练曲线、slot mask 和 checkpoint 按阶段保存在 `outputs/videosaur_multistage/`。后续 slot 提取默认使用最终 6-slot 阶段的 `best.pt`。

### 2. 冻结并提取 object slots

```bash
spavr-extract-slots --config configs/extract_slots.yaml
```

默认输出到：

```text
processed_spavr_test/object_slots/videosaur_dinov2_s14_s6/
├── metadata.json
├── index.jsonl
├── files/<episode_id>.npy   # (T,6,128), float16
└── tensorboard/             # 提取样例的 slot 可视化
```

提取脚本按 episode 处理，DINO 帧可分批计算，但时序 Slot Attention 始终在完整 episode 上连续运行。

### 3. 从 frozen slots 训练 C-JEPA/SPAVR

这一阶段再拆成 3A 动力学预训练、3B 分位数头训练和 3C 低学习率联合微调。所有 Dataset 都先按 episode 划分 train/val/test，再在单条 episode 内以 `stride=1` 构造窗口；DataLoader 在全体窗口上 shuffle。

```text
3A：冻结 slots 和 quantile head
    训练 predictor 与 proprio/action encoder
    loss = 1.0 object + 0.5 proprio + 0.1 masked-history

3B：加载 3A best.pt，冻结完整 dynamics
    只训练 quantile head
    loss = quantile

3C：加载 3B best.pt，解冻 dynamics
    head lr=1e-4，dynamics lr=1e-5
    loss = 1.0 quantile + 0.1 object + 0.05 proprio + 0.01 masked-history
```

对长度为 `T` 的 episode，3A 使用全部 `T-H-P+1` 个 `H→P` 窗口；3B/3C 只使用还能提供完整回报 horizon 的窗口。阶段之间从上一阶段 `best.pt` 初始化，但重新创建 optimizer 和 scheduler。

```bash
# 必须先完成 slot 提取；分别验证三个子阶段
spavr-train --config configs/spavr_stages/01_dynamics.yaml --smoke-test
spavr-train --config configs/spavr_stages/02_quantile_head.yaml --smoke-test
spavr-train --config configs/spavr_stages/03_joint_finetune.yaml --smoke-test

# 按顺序正式训练
spavr-train --config configs/spavr_stages/01_dynamics.yaml
spavr-train --config configs/spavr_stages/02_quantile_head.yaml
spavr-train --config configs/spavr_stages/03_joint_finetune.yaml

# 示例：只恢复被中断的 3C；不要用 --resume 跨阶段加载
spavr-train --config configs/spavr_stages/03_joint_finetune.yaml \
  --resume outputs/spavr_multistage/stage3c_joint_h3_p2_l5_s6/last.pt
```

3A 的最佳 checkpoint 按 future object/proprio validation loss 选择；3B/3C 按 validation quantile loss 选择。TensorBoard 同时记录各项 loss、分参数组学习率、梯度范数、显存、吞吐量以及回报阶段的分位数指标。`onset` 只用于验证分组，不输入模型、不参与 loss。

### TensorBoard

三个阶段统一使用 TensorBoard，不依赖 W&B：

```bash
tensorboard --logdir outputs --bind_all --port 6006
```

在远程服务器上可用 SSH 将服务器的 `6006` 端口转发到本机浏览器。所有关键指标还会保存在各输出目录的 `metrics.jsonl` 中。

## 目录

```text
spavr/              # VideoSAUR、C-JEPA backbone、分位数头和 RewardRiskModel
preprocessing/      # 原始数据转换与时间对齐
utils/              # 预处理数学函数
train/              # VideoSAUR/slot 提取/SPAVR Dataset 与训练入口
configs/            # 训练配置
docs/               # 技术与设计文档
third_party/cjepa/   # C-JEPA git submodule
```

## 环境

```bash
git submodule update --init --recursive
conda activate spavr
python -m pip install -e .
```

项目使用 Python 3.10，依赖定义在 [`pyproject.toml`](pyproject.toml)：

- `python -m pip install -e .`：只安装数据预处理所需的轻量依赖。
- `python -m pip install -e '.[model]'`：补齐 SPAVR/C-JEPA backbone 直接需要的依赖。
- `VIRTUAL_ENV="$CONDA_PREFIX" uv sync --active --inexact --extra train --extra dev`：按 `pyproject.toml` 中固定的 CUDA 12.4 PyTorch 源补齐训练与测试环境。显式设置 `VIRTUAL_ENV` 是为了让 `uv sync` 使用 conda 环境，而不是新建项目 `.venv`。

`ffmpeg`/`ffprobe` 是 Python 包管理之外的系统依赖；当前 `spavr` 环境已安装。构造 backbone 时会加载 Hugging Face `facebook/dinov2-small`，首次运行需要网络或本地缓存。

## 当前缺口

- 正式训练前需要完成 VideoSAUR 收敛检查和 slot 可视化人工检查。
- 处理后数据和相机坐标系归一化统计已经生成；VideoSAUR 第一阶段已通过 CPU 1-step smoke test，正式 GPU 训练尚未开始。
- goal encoder、候选动作生成、真机接口和跨调用长程记忆尚未实现。

## 文档

- [`方案 A：分位数奖励 / 风险头`](docs/方案A-分位数奖励风险头-技术文档.md)
- [`未来方案：真机长程记忆`](docs/未来方案-真机长程记忆模型架构.md)
- [`SPAVR Agent 范式`](docs/SPAVR：Specify–Predict–Act–Verify–Recover.md)

实现细节以当前 `spavr/` 代码为准；主方案和未来设计以对应文档为准。
