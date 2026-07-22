# SPAVR

SPAVR 是一个用于具身机器人候选动作评估的研究原型。当前模型基于 action-conditioned C-JEPA：输入近期观测、机器人状态和一段候选末端动作，在对象隐空间中预测未来，并输出整段动作的回报分位数 `q10 / q50 / q90`。

> 项目仍在开发中。模型主体和训练脚手架已经存在，真实奖励数据接入、统一数据接口和真机闭环尚未完成。

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

模型支持两个视觉入口：`pixels` 在线经过 DINOv2 和 Slot Attention；`object_slots` 跳过视觉前端，用于预抽取 slots 的高效训练。两条路径共用后续网络。

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
state_t  = pose_t^ee             # 当前实际末端位姿
action_t = Δpose_t^ee,command    # 期望末端位姿增量
```

state 不是增量，action 不是关节空间动作。两者必须使用一致的坐标、控制周期、单位和姿态参数化。当前代码读取 `proprio`，因此末端实际位姿应包含在 `proprio` 中。

```python
batch = {
    # 二选一
    "pixels":       ...,  # (B,T,3,H_img,W_img)
    "object_slots": ...,  # (B,T,S,D)
    "proprio":      ...,  # (B,T,proprio_dim)
    "action":        ...,  # (B,L,frameskip*action_dim)
    "return_to_go": ...,  # (B,)，仅训练需要
}
```

- 普通评分要求 `T>=H`、`L>=H`。
- anchor loss 训练要求状态和 action 至少覆盖 `H+P` 步。
- 默认配置：`H=5, P=3, S=4, D=128`。
- 输出形状：`(B,3)`，顺序为 `[q10,q50,q90]`。

原始 episode 需要逐时刻的视觉、proprio、末端实际位姿、末端动作增量和 reward。Dataset 为长度为 `L` 的候选计划计算一个整体标签：

```text
return_to_go = G_t^L = Σ(k=0...L-1) γ^k r_(t+k)
```

不需要人工标注三个分位数。数据应覆盖高、中、低回报轨迹，并尽量包含相似状态下的不同候选动作。

## 训练

```text
loss = pinball_loss + λ_anchor × anchor_loss
```

anchor loss 监督 masked history objects、future objects 和 future proprio，不监督 action slot。

- Frozen：冻结 C-JEPA backbone，只训练分位数头，配置见 `configs/frozen.yaml`。
- Finetune Predictor：冻结视觉前端，训练 predictor、action/proprio encoder 和分位数头，配置见 `configs/finetune_predictor.yaml`。

```bash
python -m train.train_frozen data.embedding_dir=/path/slots.pkl \
  data.action_dir=/path/actions.pkl data.proprio_dir=/path/proprio.pkl \
  cjepa_model_object=/path/cjepa_model.ckpt

python -m train.train_finetune_predictor data.embedding_dir=/path/slots.pkl \
  data.action_dir=/path/actions.pkl data.proprio_dir=/path/proprio.pkl \
  cjepa_model_object=/path/cjepa_model.ckpt
```

## 目录

```text
spavr/              # backbone、分位数头和 RewardRiskModel
train/              # Stable-Pretraining 训练入口
configs/            # 训练配置
docs/               # 技术与设计文档
third_party/cjepa/   # C-JEPA git submodule
```

## 环境

```bash
git submodule update --init --recursive
```

建议使用 Python 3.10，并按 [`third_party/cjepa/docs/ENV.md`](third_party/cjepa/docs/ENV.md) 安装依赖。构造 backbone 时会加载 Hugging Face `facebook/dinov2-small`，首次运行需要网络或本地缓存。项目目前没有顶层依赖锁文件。

## 当前缺口

- `PushTSlotDataset` 仍使用旧字段 `pixels_embed`，backbone 当前读取 `object_slots`。
- 当前 Dataset 不读取 reward，也不生成训练需要的 `return_to_go`。
- 当前没有自动化测试。
- raw-pixel 训练 DataModule、回报归一化和可复现环境尚未完成。
- 当前代码的 full-trace Transformer 头与主技术文档中的目标头仍需统一。
- goal encoder、候选动作生成、真机接口和跨调用长程记忆尚未实现。

## 文档

- [`方案 A：分位数奖励 / 风险头`](docs/方案A-分位数奖励风险头-技术文档.md)
- [`未来方案：真机长程记忆`](docs/未来方案-真机长程记忆模型架构.md)
- [`SPAVR Agent 范式`](docs/SPAVR：Specify–Predict–Act–Verify–Recover.md)

实现细节以当前 `spavr/` 代码为准；主方案和未来设计以对应文档为准。
