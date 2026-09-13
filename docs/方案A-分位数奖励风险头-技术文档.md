# 方案 A：分位数奖励 / 风险头技术文档

## 1. 目标

SPAVR 使用 C-JEPA 预测候选动作可能产生的未来对象隐状态，再从完整预测轨迹中估计任务回报分布：

```text
历史状态 + 任务目标 + 候选未来动作
                    ↓
        action-conditioned C-JEPA rollout
                    ↓
              rollout trace
                    ↓
          分位数奖励 / 风险输出头
                    ↓
             q10 / q50 / q90
```

- `q50`：候选动作的典型回报。
- `q10`：低回报尾部，用于风险敏感决策。
- `q90-q10`：相同条件下结果分布的宽度。

模型对一整段候选动作只输出一组 `[q10,q50,q90]`，不输出每个未来时刻的奖励。每个训练样本提供一个由逐时刻奖励聚合得到的整体回报标量，不需要人工标注三个分位数。模型通过分位数回归学习 `q10/q50/q90`。

本方案不使用成功/失败分类标签。数据集只定义奖励：期望行为和状态赋高奖励，不期望行为和状态赋低奖励；模型只学习回报的高低及其分布。

## 2. 网络主体

### 2.1 总体结构

```text
pixels 或 object_slots（二选一）
        │
        ├─ pixels：encoder → initializer → slot attention
        └─ object_slots：直接读取预抽取对象表示
        │
object slots[:H] + proprio_encoder(proprio[:H])
                 + action_encoder(action[:H])
        │
        ▼
predict P 个 future embeddings
        │
        ├─ 收集本轮预测 ─────────────────┐
        │                                │
action 下一段 → 替换 future action slot   │
        │                                │
滚动最后 H 步并重复 predictor             │
        │                                │
action 耗尽后再预测并收集 ─────────────────┘
                                         │
                              目标 rollout_trace
                                (B,R,P,S+2,D)
                                       │
                         object + predicted proprio tokens
                                       │
                    per-round Transformer token mixer
                                       │
                             attention pooling
                                       │
                          GRU over rollout rounds
                                       │
                       quantile-specific attention queries
                                       │
                       monotonic q10 / q50 / q90
```

符号定义：

| 符号 | 含义 |
| --- | --- |
| `H` | 历史观测长度 |
| `L` | 输入 action 序列长度 |
| `P` | predictor 每轮并行预测的未来步数 |
| `R` | rollout 轮数 |
| `S` | 对象 slot 数量 |
| `D` | slot 维度 |

### 2.2 历史状态编码

模型保留两个互斥的视觉接口，两者在 object slots 处汇合：

```python
# 在线视觉路径：正常推理使用，也可用于训练
pixels:       (B,T,3,H_img,W_img)

# 训练加速路径：视频提前经过 VideoSAUR 后缓存
object_slots: (B,T,S,D)
```

`pixels` 路径在线运行视觉 encoder、initializer 和 slot attention；`object_slots` 路径跳过这些模块，直接把预抽取结果送入 predictor，以减少训练计算量。两条路径共用相同的 proprio/action encoder、predictor 和分位数头。

评分时 `T=H`；计算 anchor loss 时训练 batch 需要 `T=H+P`。预抽取 slots 必须来自与模型一致的 VideoSAUR 权重和 slot 配置。当前代码在 batch 含 `object_slots` 时优先使用该路径，否则读取 `pixels`。

模型中的当前数值状态是 7 维 proprio：

```text
proprio_t = [pose_t^ee, gripper_(t-1)]
```

前 6 维是末端实际位置和 rotation vector，第 7 维是动作执行前的夹爪状态。不使用关节角、关节速度或关节力矩。视觉 pixels / slots 表示外部场景观测。

当前代码没有独立的 goal / instruction encoder，任务目标只能通过图像或 slots 进入模型。

### 2.3 动作条件化 rollout

```python
action: (B,L,7)  # 6 维末端位姿增量 + 1 维夹爪命令
```

每个 action 对应一个重采样后的时间步，经过 `action_encoder` 映射到维度 `D`。

```python
history = concat(
    object_slots[:H],
    proprio_encoder(proprio[:H]),
    action_encoder(action[:H]),
)

current = H
while current < L:
    future = world_model.predict(history)[0][:, H:H+P]
    n = min(P, L-current)
    future[:, :n, action_slot] = action_encoder(action[:, current:current+n])
    history = last_H(concat_time(history, future[:, :n]))
    current += n

final_future = world_model.predict(history)[0][:, H:H+P]
```

`action[:H]` 直接进入初始 history；剩余 action 在一次预测完成后写入 action slot，因此影响下一次 predictor 调用，不改变本轮已经生成的 object/proprio。要求 `L>=H`，调用次数为 `R=ceil((L-H)/P)+1`。

```text
当前代码：z_future     (B,P,S+2,D)    # 仅最后一轮
目标接口：rollout_trace (B,R,P,S+2,D) # 收集全部轮次
```

最后一轮中的 action slot 是 predictor 输出，不是 raw action；分位数头不读取它。

## 3. Full-trace 分位数头

### 3.1 输入 token

输出头默认读取每轮预测中的：

```text
object slots + predicted proprio slot
```

raw action 已经用于生成预测轨迹，不直接把预测 action slot 当作真实计划动作。

### 3.2 轨迹聚合

```text
rollout_trace
  │
  ├─ 选择 object + predicted proprio
  ├─ 输入投影
  ├─ slot-type embedding
  ├─ within-round time embedding
  ├─ rollout-round embedding
  │
  ├─ 每轮 Transformer token mixer
  ├─ 每轮 attention pooling
  ├─ GRU 聚合 rollout rounds
  ├─ q10 / q50 / q90 learned queries 分别读取轨迹
  │
  └─ 单调分位数 readout
```

Transformer 负责建模同一轮内的对象关系和时间关系，GRU 负责建模多轮 rollout 的先后顺序。三个分位数查询使用不同的注意力分布，但共享主干特征。

### 3.3 单调分位数输出

```text
q50 = median_head(h50)
q10 = q50 - softplus(lower_gap_head(h10))
q90 = q50 + softplus(upper_gap_head(h90))
```

该参数化始终保证：

```text
q10 <= q50 <= q90
```

## 4. 数据与标签

### 4.1 Episode 数据

原始数据按完整 episode 保存：

```python
episode = {
    "pixels_or_slots": ...,
    "proprio": ...,       # 6 维末端位姿 + 上一时刻夹爪状态
    "action": ...,        # 6 维末端位姿增量 + 夹爪命令
    "reward": ...,        # 每个控制时刻的奖励
    "goal": ...,
}
```

必须统一时间语义：`proprio[t]` 和外部图像是动作执行前的观测，`action[t]` 是基于该观测下发的 7 维动作，`reward[t]` 是执行该动作后得到的回报。三者时间长度相同。

### 4.2 回报标签

候选动作评分固定使用与输入动作计划严格对齐的有限时域累计回报。对于从时刻 `t` 开始、长度为 `L` 的候选动作：

```text
G_t^L = Σ(k=0...L-1) γ^k r_(t+k)
```

原始 episode 必须记录每个控制时刻的奖励：

```text
r_t, r_(t+1), ..., r_(t+L-1)
```

Dataset 根据窗口起点和动作长度自动计算一个标量：

```python
return_to_go = G_t^L
```

输出头使用该标量训练，最终输出：

```text
q10：整段候选动作的保守回报评分
q50：整段候选动作的典型整体回报
q90：整段候选动作结果较好时的整体回报
```

推理和动作选择默认使用 `q50` 作为整体奖励评分；需要风险敏感决策时使用 `q10`。训练前使用训练集统计量对 `G_t^L` 归一化，验证和测试沿用同一组统计量。

### 4.3 数据覆盖

训练数据应覆盖从高奖励到低奖励的不同轨迹和局部片段。模拟环境中优先从同一历史状态分支执行多条候选动作，使模型真正学习动作对回报的影响。

如果环境完全确定，同一状态和动作只产生唯一结果，分位数间距自然会缩小。要学习条件结果分布，需要执行噪声、环境变化或其他未完全观测因素。

所有由同一初始状态产生的分支必须放在同一个数据划分中，避免训练与验证之间泄漏。

## 5. 训练目标

总损失为：

```text
loss = loss_quantile
     + λ_anchor × loss_anchor
```

### 5.1 分位数损失

```text
u = return_to_go - qτ
ρτ(u) = max(τu,(τ-1)u)
```

默认使用 pinball loss；回报噪声较大时可使用 Huberized pinball loss。

### 5.2 Anchor 监督

```text
history = embedding[:, :H]
target  = stop_gradient(embedding[:, H:H+P])

loss_masked_history =
    MSE(pred_history[:, :, mask_indices],
        stop_gradient(history[:, :, mask_indices]))

loss_future_object =
    MSE(pred_future[:, :, :S],
        target[:, :, :S])

loss_future_proprio =
    MSE(pred_future[:, :, S:S+1],
        target[:, :, S:S+1])

loss_anchor = loss_masked_history
            + loss_future_object
            + loss_future_proprio
```

该项严格对应当前代码的单轮 `H→P` anchor loss：监督历史中被 mask 的对象 slots、未来对象 slots 和未来 proprio slot，不监督 action slot。只有 `anchor_weight > 0` 时才计算并加入总损失。

## 6. 训练策略

### 3A Dynamics

```text
冻结 VideoSAUR、预抽取 object slots 和 quantile head
训练 predictor、action encoder 和 proprio encoder
```

每个 episode 内以 `stride=1` 使用全部 `H→P` 合法窗口。损失权重为 future-object `1.0`、future-proprio `0.5`、masked-history `0.1`，最佳 checkpoint 按前两项加权后的 validation loss 选择。

### 3B Quantile Head

```text
加载 3A best.pt
冻结完整 dynamics
只训练 full-trace quantile head
```

仅使用 quantile loss，最佳 checkpoint 按 validation quantile loss 选择。

### 3C Joint Finetune

```text
加载 3B best.pt
继续冻结 VideoSAUR 和 object slots
联合训练 dynamics 与 quantile head
```

head 使用 `1e-4`，dynamics 使用 `1e-5`。损失权重为 quantile `1.0`、future-object `0.1`、future-proprio `0.05`、masked-history `0.01`。阶段切换重新创建 optimizer/scheduler，`--resume` 只用于同一阶段的中断恢复。
