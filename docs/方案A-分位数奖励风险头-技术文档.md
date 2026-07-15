# 方案 A：分位数奖励 / 风险头 —— 技术文档

> 本文是方案 A 的落地级技术说明，覆盖模型架构、数据集、训练、损失函数、推理五个方面。总体设计与其它方案的对比见 `C-JEPA奖励风险模型-方案设计.md`。

## 0. 一句话定位

在 C-JEPA（AP 版）之上加一个**分位数头**：读对象隐状态 z，输出未来回报的一组分位数 `[q10, q50, q90]`。中位数当奖励，下尾当风险，分布宽度当不确定性。

**两种训练策略，各一个独立训练脚本，产出统一打包成一个 ckpt**（C-JEPA + 头端到端，不单独拆出头）：
- **策略 1**：冻结 C-JEPA 全部，只训输出头。
- **策略 2**：冻结 C-JEPA 的 encoder（图像 encoder / slot-attention / initializer），训练 predictor + 输出头。

---

## 1. 模型架构

### 1.1 Backbone（第三方库，只 import 不改源码）

- 复用 `third_party/cjepa` 的 `MaskedSlot_AP_Predictor`（`src/cjepa_predictor.py`）。
- 加载 C-JEPA 的 AP checkpoint。图像 encoder / slot-attention / initializer 始终冻结（图像 encoder 权重恒为官方 videosaur ckpt）；predictor 按训练策略决定冻结或训练。
- 输入每帧的槽结构（固定顺序）：

```
每帧 = [ obj_1, ..., obj_S , proprio , action ]     # 共 S+2 个槽
整段  (B, T, S+2, D)      PushT: S=4 → 6 槽，D=128
```

- transformer 输出同形状的隐表征 `z: (B, T, S+2, D)`。z 已融合动作 / 本体信息。

### 1.2 分位数头（我们新增，`spavr/heads.py`）

```
z (B, T, S+2, D)
  │  取对象槽 z[..., :S, :]                    # 只用 S 个对象槽当状态特征
  │  取最后历史帧 → (B, S, D)
  │  注意力池化：可学 query 对 S 个槽算权重 w，加权合并 → h (B, D)
  ▼
MLP: D → hidden → 3
  ▼
输出 (B, 3) = [q10, q50, q90]                   单调不减
```

- **注意力池化（默认且唯一的聚合方式）**：不做等权平均。设一个可学习 query `q`，对每个对象槽算相似度 `score_i = q·obj_i`，softmax 得权重 `w_i`，再 `h = Σ w_i · obj_i`。
  - **为什么**：风险常源于单个物体（要撞的那个）。等权平均会按 `1/S` 把它冲淡（S 越大越严重）；注意力池化把权重集中到关键物体，危险信号不被淹没。等权平均只是"权重恒为 1/S"的特例。
- **只 pool 对象槽**：头专注给"状态"打分。
- **单调性**：为避免分位数交叉（q10 > q90），头输出"基值 + 非负增量累加"：`q_k = base + Σ_{j≤k} softplus(δ_j)`。
- 头很轻：一个可学 query + 注意力池化 + 2 层 MLP，参数量远小于 backbone。

### 1.3 组装（`spavr/model.py`）

**当前实现**：`RewardRiskModel` 把 OC 编码阶段（视觉 encoder / initializer / slot-attention）也内置进来，`forward` 直接接受原始像素字典：

```
class RewardRiskModel(nn.Module):
    oc_encoder    : MapOverTime(FrameEncoder)          # videosaur 配置构建
    oc_initializer: RandomInit
    oc_processor  : ScanOverTime(LatentProcessor(SlotAttention))
    backbone      : CJepaBackbone（predictor + action/proprio encoders）
    head          : QuantileHead

    def forward(x: dict{pixels(B,T,3,H,W), action, proprio})
        slots  = oc_processor(oc_initializer(...), oc_encoder(pixels))
        z      = backbone.forward_repr(slots, action, proprio)  # (B, S+2, D)
        return head(z)                                           # (B, K) 分位数
```

OC 模型结构由 `third_party/cjepa/configs/config_train_causal_pusht_slot.yaml` 驱动构建（`OmegaConf.load` + `models.build`），权重注入留给 `train/factory.py`（待实现）。

整个 `RewardRiskModel`（OC encoder + backbone + head）作为一个模型统一保存 / 加载，产出一个打包 ckpt。

---

## 2. 所需数据集

### 2.1 底座（C-JEPA 现有，按 `episode_id` 对齐）

```
slots:   [T, S, D]      # 预抽取视觉对象槽（slot pkl）
action:  [T, act_dim]   # 原始动作（meta pkl）
proprio: [T, prop_dim]  # 本体感受（meta pkl）
```

### 2.2 方案 A 唯一额外要的：奖励

```
reward:  [T]            # 每步标量奖励
```

- **来源**：PushT 自带 reward（覆盖度 / 距离）；CLEVRER 从碰撞 / 因果标注派生 reward。
- **关键约束（决定分位数能否学出）**：需**相似 (状态, 动作) 对应有高有低的结局**的多样轨迹，分布的尾巴才学得出；若每个状态结局唯一，分位数会退化成一条线。所以数据要覆盖成功与失败两类轨迹。

### 2.3 训练目标：return-to-go

从 `reward[T]` 现场算未来累计回报（可带折扣 γ）：

```
G(t) = Σ_{k≥0} γ^k · reward[t+k]
```

每个时间步 t 产出一条训练样本 `(历史窗口结束于 t, G(t))`。一条轨迹 → 多条样本。

---

## 3. 训练

两种策略，各自一个独立训练脚本，产出都统一打包成一个 ckpt（C-JEPA + head 端到端）。

### 3.1 策略 1：冻结 C-JEPA 全部，只训输出头

脚本 `train/train_frozen.py`。

```
for batch in loader:                      # batch: 一组 (历史窗口, G)
    with torch.no_grad():
        z = backbone(history)             # 整个 C-JEPA 冻结
    q = head(z)
    loss = pinball_loss(q, G, taus)       # 见第 4 节
    loss.backward(); opt.step()           # 只更新 head
```

- 优化器只收 `head.parameters()`，head lr ≤ 5e-4。
- C-JEPA 全部 `requires_grad=False` + `eval()`。

### 3.2 策略 2：冻结 encoder，训练 predictor + 输出头

脚本 `train/train_finetune_predictor.py`。

```
for batch in loader:
    z = backbone(history)                 # encoder 冻结，predictor 可训
    q = head(z)
    loss = pinball_loss(q, G, taus) + λ · anchor_loss(backbone, history)
    loss.backward(); opt.step()           # 更新 predictor + head
```

- 冻结：图像 encoder / slot-attention / initializer（`requires_grad=False`）。
- 训练：predictor + head（可含 `action_encoder` / `proprio_encoder`）。
- 优化器用参数组：head lr ≤ 5e-4；predictor lr 5e-6 ~ 5e-5（远小于 head）。
- **锚 loss（必带）**：保留 C-JEPA 原自监督 loss（masked-history + future-prediction）一项，权重 λ，防 predictor 表征漂移：
  ```
  L = pinball(reward) + λ · (masked-history + future-prediction loss)
  ```

### 3.3 样本构造（两策略共用）

- 历史窗口长度 = `dinowm.history_size`（PushT cjepa 配置默认 **5**，backbone 默认值与之对齐），与 backbone 一致。
- 样本：窗口 = 历史帧，目标 = G(窗口末帧)。

### 3.4 权重产出（两策略共用）

- 保存整个 `RewardRiskModel`（C-JEPA + head）为**一个打包 ckpt**，不单独拆出 head。
- 图像 encoder 权重恒为官方 videosaur（冻结），随打包一起存。
- 策略 1 下 C-JEPA 部分与加载时一致；策略 2 下 predictor 部分为微调后权重。

---

## 4. 损失函数

### 4.1 Pinball loss（分位数回归）

对每个分位 τ，预测 `q_τ`、真值 `G`，误差 `u = G − q_τ`：

```
ρ_τ(u) = max(τ·u, (τ−1)·u)
       = u·τ         若 u ≥ 0   (低估，惩罚权重 τ)
         u·(τ−1)     若 u < 0   (高估，惩罚权重 1−τ)
```

总损失对所有分位求和、对 batch 求平均：

```
L = (1/B) Σ_b Σ_k ρ_{τk}( G_b − q_{τk,b} )
```

不对称惩罚是关键：对 τ=0.1，低估几乎不罚、高估重罚 → 输出被压到 10% 分位；τ=0.9 反之。这样一个头的不同输出各自对准不同分位。

### 4.2 分位数交叉的处理

- **首选**：架构上保证单调（第 1.2 节的"基值 + 非负增量累加"），从根上杜绝交叉。
- **备选**：输出后排序，或加一项 `Σ max(0, q_k − q_{k+1})` 惩罚。

### 4.3 锚 loss（策略 2）

- 策略 2 下锚 loss（保留 C-JEPA 原自监督 loss，权重 λ）为必带项，防表征漂移，详见 3.2。

---

## 5. 推理

实时监控当前局面：

```python
# x = {"pixels": (B,T,3,H,W), "action": (B,T,act_dim), "proprio": (B,T,prop_dim)}
q10, q50, q90 = model(x).unbind(dim=1)
# 等价展开：
#   slots = oc_processor(oc_initializer(B), oc_encoder(x["pixels"]))
#   z     = backbone.forward_repr(slots, x["action"], x["proprio"])  # (B, S+2, D)
#   q     = head(z)                                                   # (B, 3)
```

| 输出 | 读作 | 用途 |
|---|---|---|
| q50 | 期望奖励 | 局面好坏 |
| q10 | 风险下限（最坏情况） | q10 跌破阈值 → 报警"危险" |
| q90 − q10 | 不确定性 | 越宽越没把握，可触发保守策略 |

- **推理不需要任何标签**；标签只在训练当监督。

---

## 6. 关键超参 / 配置

| 项 | 参考值 | 说明 |
|---|---|---|
| 分位数集合 τ | [0.1, 0.5, 0.9]（可加 0.25/0.75） | 决定风险粒度 |
| history_size | **5**（PushT cjepa 配置默认值，backbone 对齐） | 历史窗口 |
| 折扣 γ | 0.99 或 1.0 | return-to-go 折扣 |
| head lr | ≤ 5e-4 | 两策略通用 |
| pool | 前 S 个对象槽，注意力池化 | 可学 query |
| predictor lr | 5e-6 ~ 5e-5 | 仅策略 2，远小于 head lr |
| anchor λ | 需调 | 仅策略 2，锚 loss 权重（必带） |

---

## 7. 代码落点（均在 `Code/SPAVR/`，不改 `third_party/cjepa`）

- ✅ `spavr/backbone.py`：`CJepaBackbone`，内部构建 `CausalWM_AP`（predictor + action/proprio encoders）；暴露 `build_embedding`、`forward_repr`、`anchor_loss`。**`set_trainable` 尚未实现（待补）。**
- ✅ `spavr/heads.py`：`QuantileHead`（注意力池化 + 单调 MLP）+ `pinball_loss`。
- ✅ `spavr/model.py`：`RewardRiskModel`，内置 OC encoder/initializer/processor + backbone + head；`forward(x)` 接收像素字典，返回分位数。打包保存 / 加载由 `train/factory.py` 负责（待实现）。
- 🔲 `train/train_frozen.py`：策略 1，冻结 C-JEPA 全部、只训头。（待实现）
- 🔲 `train/train_finetune_predictor.py`：策略 2，冻结 encoder、训 predictor + 头，带锚 loss。（待实现）
- 🔲 `configs/`：两策略各一份配置（分位数 τ、γ、head lr、predictor lr、anchor λ 等）。（待实现）

---

## 8. 文件树与分层

**分层目标**：`spavr/` 只定义网络结构（`nn.Module` + forward），**不碰权重文件、不碰数据、不碰训练**。凡是"加载 / 保存参数、读 ckpt、建 DataLoader、跑循环"这类带 I/O 或副作用的，全放 `train/`。

> **当前偏差**：`spavr/model.py` 的 `__init__` 中有 `OmegaConf.load(yaml)` + `models.build(cfg)`（见 §8.3 第 1 条待拍板）。

### 8.1 文件树（✅ 已实现 / 🔲 待实现）

```
Code/SPAVR/
├── third_party/cjepa/                # 第三方库，只 import，不改
├── docs/
├── spavr/                            # ✅ 已实现
│   ├── __init__.py
│   ├── backbone.py                   # CJepaBackbone（内部构建 CausalWM_AP）
│   ├── heads.py                      # QuantileHead（注意力池化 + 单调 MLP）+ pinball_loss
│   └── model.py                      # RewardRiskModel：OC encoder + backbone + head，forward(pixels_dict)
├── configs/                          # 🔲 待实现
│   ├── frozen.yaml                   # 策略 1 超参 + 路径
│   └── finetune_predictor.yaml       # 策略 2 超参 + 路径
├── train/                            # 🔲 待实现：建 cjepa、灌权重、数据、循环、保存
│   ├── factory.py                    # 灌 OC/predictor 权重 + 拼 RewardRiskModel；打包 save/load
│   ├── data.py                       # ReturnToGoDataset（真数据）/ FakeSlotDataset（冒烟）
│   ├── common.py                     # 训练循环 / 优化器参数组 / 日志
│   ├── train_frozen.py               # 策略 1
│   └── train_finetune_predictor.py   # 策略 2
├── scripts/train.sh                  # 🔲 待实现
└── tests/test_smoke.py               # 🔲 待实现
```

### 8.2 各文件职责

**`spavr/`（已实现）**

- `backbone.py` — `CJepaBackbone(nn.Module)`（✅ 已实现，⚠️ `set_trainable` 待补）：
  - `__init__(num_objects, slot_dim, history_size, num_preds, action_dim, proprio_dim, frameskip, ...)`：内部构建 `MaskedSlot_AP_Predictor` + `action_encoder` + `proprio_encoder`，再包进 `CausalWM_AP(encoder=None, slot_attention=None, initializer=None, ...)`；不做任何 `torch.load`。
  - `build_embedding(slots, action, proprio)` → `(B,T,S+2,D)`：拼接顺序 proprio 在前、action 在后。
  - `forward_repr(slots, action, proprio)` → z `(B,S+2,D)`：临时 `num_masked_slots=0` 跑 predictor，取最后历史帧的上下文化表征。
  - `anchor_loss(slots, action, proprio)`：masked-history MSE + future-obj MSE + future-proprio MSE，返回标量。
  - **⚠️ `set_trainable(mode)` 尚未实现**：训练脚本依赖此方法切换 `requires_grad`（`"frozen"` / `"predictor"`）。
- `heads.py` — ✅ `QuantileHead`（对象槽 → 可学 query 注意力池化 → LayerNorm+MLP → 基值+softplus 增量累加得单调 `[q10,q50,q90]`）+ `pinball_loss`（`torch.maximum(τ·u, (τ−1)·u).sum(1).mean()`）。
- `model.py` — ✅ `RewardRiskModel(nn.Module)`：
  - `__init__` 内置 OC encoder / initializer / processor（从 cjepa pusht yaml 用 `OmegaConf.load + models.build` 构建，`load_weights=None`）；OC 权重由 `train/factory.py` 注入（待实现）。
  - `forward(x: dict)` → `(B, K)`：`oc_encoder → oc_initializer → oc_processor → backbone.forward_repr → head`，全链路一次调用。
  - 无内置 save/load；打包 ckpt 的存取交给 `train/factory.py`。

**`train/`（全部待实现）**

- `factory.py` — 权重注入与打包：
  - `build_reward_model(ckpt_path, cfg)`：实例化 `RewardRiskModel`，加载 OC + predictor 权重，返回完整模型。
  - `save_packed / load_packed`：整包 ckpt 的存取。
- `data.py` — `ReturnToGoDataset`（接 slot/action/proprio/reward pkl，现场算 `G(t)`）+ `FakeSlotDataset`（假像素冒烟）。
- `common.py` — DataLoader、优化器参数组、训练 / 验证循环、日志、调用 `factory.save_packed`。
- `train_frozen.py` — 策略 1：`backbone.set_trainable("frozen")`，`z` 在 `no_grad` 下取，优化器只收 head，`loss=pinball`。
- `train_finetune_predictor.py` — 策略 2：`backbone.set_trainable("predictor")`，参数组（head 大 lr / predictor 小 lr），`loss=pinball + λ·anchor_loss`。
- `scripts/train.sh` — 命令行入口，选策略 + 传 config。
- `tests/test_smoke.py` — 用 `FakeSlotDataset` 跑 forward + pinball，断言 `q10≤q50≤q90` + 形状。

### 8.3 待拍板 / 待解决

1. **`spavr/model.py` 中的 I/O**：当前 `__init__` 含 `OmegaConf.load(yaml)`，违反"spavr 零 I/O"原则。两个选项：(a) 接受现状，model.py 允许读 yaml 做结构构建；(b) 把 `models.build` 移到 `train/factory.py`，`RewardRiskModel.__init__` 改为接收已构建的 `oc_encoder/oc_initializer/oc_processor`。
2. **`set_trainable` 缺失**：训练策略 1/2 的前置条件，需立即补到 `backbone.py`。
3. **OC 权重注入**：`model.py` 设 `load_weights=None`，OC 模型当前随机初始化，需 `train/factory.py` 实现后才能用真权重。
4. **外部依赖 `stable_worldmodel`**：`backbone.py` 的 `import stable_worldmodel as swm` 需确认安装；`Embedder` 类同样定义于 `cjepa/src/world_models/dinowm_causal_AP_node.py:475`，可改为直接 import 以消除额外依赖。
5. **`forward_repr` 取 z 不带随机 mask（`num_masked_slots=0`）**：已确认实现，见 `backbone.py:73-78`。✅

