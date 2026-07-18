# 方案 A：分位数奖励 / 风险头技术文档

## 0. 目标

SPAVR 在 C-JEPA 的未来对象隐状态上增加分位数头，输出 `[q10, q50, q90]`：`q50` 是奖励评分，`q10` 是低回报风险，`q90-q10` 是结果不确定性。

训练使用 return-to-go 标签和 pinball loss。支持冻结 C-JEPA 只训练头，以及冻结视觉前端、微调 predictor/动作编码器/本体编码器与头两种策略。

## 1. 完整网络

```text
raw pixels 或 pixels_embed
        │
        ├─ raw pixels：encoder → initializer → slot_attention
        └─ pixels_embed：直接使用预抽取对象 slots
        │
对象 slots + proprio slot + action slot
        │
历史 embedding (B,H,S+2,D)
        │
按完整动作序列迭代 AP rollout
        │ 每轮 predictor 创建 P 帧 future query
        │ 写入下一段动作并滚动最后 H 帧 history
        │ 动作耗尽后再运行一次 predictor
        ▼
post_action_future (B,P,S+2,D)
        │
取 P×S 个未来对象 token → 单 query 注意力池化
        │
MLP + 单调分位数参数化
        ▼
quantiles (B,K)
```

PushT 当前默认：

```text
H=5
P=3
S=4
D=128
action_input_dim=frameskip×action_dim=3×2=6
K=3
```

未来动作长度记为 `L=x["action"].shape[1]`，不是固定超参数。要求 `L>=H`；模型消费完整序列，predictor 调用次数为 `ceil((L-H)/P)+1`。默认训练窗口 `L=H+P=8` 时调用两次，但推理可接收更长的 VLA 动作序列。

`action_dim` 和 `frameskip` 都是构造参数。单步机械臂动作本身为 6 维或 7 维时，应设置 `frameskip=1`，并分别设置 `action_dim=6` 或 `action_dim=7`；模型实例构造后输入宽度保持固定。

对应数据尺寸：

```text
每轮 history:          (B,5,6,128)
每轮 predictor grid:   (B,8,6,128)
post_action_future:     (B,3,6,128)
pool tokens:            (B,12,128)
quantiles:              (B,3)
```

## 2. Predictor 的真实输入

每一轮传给 predictor 的都是当前最后 H 帧 embedding `(B,H,S+2,D)`。`MaskedSlot_AP_Predictor.prepare_input()` 在内部为 P 帧创建：

```text
future query = mask_token + time_position + anchor_identity_query
```

历史和未来 query 组成 `(B,H+P,S+2,D)`，展平为 `(B,48,128)` 后一次送入 non-causal full-attention Transformer。未来 query 不是视觉 initializer 或 Slot Attention 生成的真实未来 slots。

评分 forward 在完整 rollout 期间临时把 `num_masked_slots` 设为 0，并始终使用可求导的普通 predictor forward。动作耗尽后额外一轮输出完整 P 帧，因此训练、推理都能把最终 `(B,P,S+2,D)` 交给同一个分位数头。anchor loss 仍保留配置的历史对象 mask，维持官方单轮预测训练目标。

## 3. Action 与 proprio

外部 `action` 只表示未来动作序列：

```text
action: (B,L_action,6)
L_action >= H
```

SPAVR 消费传入的全部 L 个动作块：

- 前 H 个由 `CausalWM_AP.encode(..., action_key="action")` 或等价的 action encoder 路径加入历史 embedding。
- 剩余动作每次最多取 P 个；predictor 生成 P 帧后，通过 `CausalWM_AP.replace_action_in_embedding()` 写入对应预测帧的 action 槽。
- 写入动作的预测帧追加到序列中，下一轮只取最后 H 帧作为 predictor history。
- 全部动作耗尽后再运行一次 predictor，返回完整 P 帧 post-action future 给分位数头。

这复用了 C-JEPA 官方 AP rollout 的动作传播顺序。某一轮新写入的动作不会反向改变该轮已经生成的对象 slots，而是在下一轮 predictor 中生效；最终评分使用动作耗尽后的额外预测，所以完整动作序列都会沿 rollout 路径影响分位数。调用次数严格为 `ceil((L-H)/P)+1`。

proprio 在原始像素路径通过 `CausalWM_AP.encode(..., proprio_key="proprio")` 加入；预抽取 slot 路径调用同一个 `proprio_encoder` 后拼成一个 token。

## 4. 双输入接口

原始像素：

```python
batch = {"pixels": pixels, "proprio": proprio, "action": future_actions}
z_future = backbone(batch)
```

数据流是 `pixels → CausalWM_AP.encode → encoder/initializer/slot_attention → 完整 embedding`。

预抽取 slots：

```python
batch = {
    "pixels_embed": object_slots,
    "proprio": proprio,
    "action": future_actions,
}
z_future = backbone(batch)
```

这条路径跳过视觉前端，适合直接复用 C-JEPA 的 slot pkl 和 Stable-Pretraining 数据流程。两条路径从 predictor 开始完全一致。

## 5. 分位数头

`QuantileHead` 只读取动作序列传播完成后的最终 P 帧对象 slots：

```text
z_future[:, :, :S, :]       (B,P,S,D)
flatten time/object          (B,P×S,D)
single-query attention       (B,D)
LayerNorm + MLP              (B,K)
```

输出采用基值加非负增量累加，所以始终满足 `q10 <= q50 <= q90`。训练损失为：

```text
ρτ(u) = max(τu,(τ-1)u)
u = return_to_go - qτ
```

## 6. Stable-Pretraining 集成

`RewardRiskModel` 是纯 `nn.Module`。两个训练策略分别由独立入口和独立 YAML 驱动：

```text
train/train_frozen.py             ↔ configs/frozen.yaml
train/train_finetune_predictor.py ↔ configs/finetune_predictor.yaml
```

Stable-Pretraining state 包含 `quantiles`、`loss_quantile`、可选的 `loss_anchor` 和总 `loss`。`spt.Module` 负责 backward、多优化器、梯度累积、裁剪、scheduler、DDP、验证和 checkpoint；`stage="predict"` 不要求标签。

### 策略 1：Frozen

```text
EvalOnly：整个 CJepaBackbone
训练：QuantileHead
loss：pinball
optimizer：head_opt
```

### 策略 2：Predictor

```text
EvalOnly：encoder / initializer / slot_attention
训练：predictor / action_encoder / proprio_encoder / QuantileHead
loss：pinball + λ×anchor
optimizer：head_opt + dynamics_opt
```

anchor loss 是 masked-history object MSE、future-object MSE 和 future-proprio MSE 之和。

## 7. 数据契约

Frozen 策略不计算 anchor 时只要求 H 帧状态：

```text
pixels:       (B,H,3,H_img,W_img) 或
pixels_embed: (B,H,S,D)
proprio:      (B,H,4)
action:       (B,L_action,6), L_action>=H
return_to_go: (B,)
```

Predictor 策略计算 anchor 时需要 H+P 帧真值：

```text
pixels:       (B,H+P,3,H_img,W_img) 或
pixels_embed: (B,H+P,S,D)
proprio:      (B,H+P,4)
action:       (B,L_action,6), L_action>=H+P
return_to_go: (B,)
```

return-to-go 定义为 `G(t)=Σ(k>=0) γ^k reward(t+k)`。样本必须覆盖相似状态/动作下的成功与失败结果，分位数宽度才有可学习的分布意义。

## 8. 训练与推理

Stable-Pretraining 训练：

```bash
python -m train.train_frozen \
  data.embedding_dir=/path/slots.pkl \
  data.action_dir=/path/actions.pkl \
  data.proprio_dir=/path/proprio.pkl \
  cjepa_model_object=/path/cjepa_object.ckpt

python -m train.train_finetune_predictor \
  data.embedding_dir=/path/slots.pkl \
  data.action_dir=/path/actions.pkl \
  data.proprio_dir=/path/proprio.pkl \
  cjepa_model_object=/path/cjepa_object.ckpt
```

普通推理不依赖 Stable-Pretraining：

```python
model.eval()
with torch.inference_mode():
    q10, q50, q90 = model(batch).unbind(dim=1)
```

`CJepaBackbone` 不读取外部 VideoSAUR YAML，也不提供 `visual_model` / `world_model` 注入参数。DINOv2 encoder、投影 MLP、RandomInit、SlotAttention、VideoSAUR 时序 processor 和 `CausalWM_AP` 都直接在代码中创建；`cjepa_model_object` 只用于从官方保存对象中加载参数。checkpoint 对应的 H、P、slot 配置、action 输入宽度和 proprio 维度必须与当前模型一致。

## 9. 当前代码状态

已实现：

- `spavr/backbone.py`：完整 C-JEPA、双输入、普通 forward、anchor loss。
- `spavr/heads.py`：未来对象注意力池化、单调分位数、pinball loss。
- `spavr/model.py`：纯 RewardRiskModel。
- `train/common.py`：两个入口共用的模型/数据构建、spt.Module、联合 loss、日志和 Manager。
- `train/train_frozen.py`：只运行 Frozen 策略。
- `train/train_finetune_predictor.py`：只运行 Predictor 策略。
- `configs/frozen.yaml`、`configs/finetune_predictor.yaml`。
- `tests/test_smoke.py`：两种输入、两种训练策略、冻结与梯度。

训练入口仍需提供真实 C-JEPA/VideoSAUR checkpoint、return-to-go 数据集、DataLoader、Lightning Trainer、日志器和保存目录。
