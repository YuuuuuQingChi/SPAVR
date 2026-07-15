# 方案 B：世界模型不确定性即风险 —— 技术文档

> 本文是方案 B 的落地级技术说明，覆盖模型架构、数据集、训练、度量函数、推理五个方面。总体设计与其它方案的对比见 `C-JEPA奖励风险模型-方案设计.md`。

## 0. 一句话定位

不新学"什么危险"，而是**用 C-JEPA 自己预测得准不准来当风险**：预测不准 = 没见过 / 难以掌控 = 风险高。

**全程无标签**，直接白嫖 C-JEPA 的预测能力。两条风险信号：
- **预测误差**：rollout 出的未来槽 vs 真实后续观测，误差大 → 风险高。
- **模型分歧**：ensemble 或多次 masked-prediction 的方差，分歧大 → 认知不确定性高。

---

## 1. 模型架构

### 1.1 Backbone（第三方库，只 import 不改源码）

- 复用 `third_party/cjepa` 的 `MaskedSlot_AP_Predictor`（`src/cjepa_predictor.py`）。
- 加载 C-JEPA 的 AP checkpoint，**全程冻结**（`requires_grad=False` + `eval()`）——方案 B 不训练 backbone，只用它预测。
- 输入每帧的槽结构（固定顺序）：

```
每帧 = [ obj_1, ..., obj_S , proprio , action ]     # 共 S+2 个槽
整段  (B, T, S+2, D)      PushT: S=4 → 6 槽，D=128
```

### 1.2 两条风险信号（我们新增，`spavr/uncertainty.py`）

**信号① 预测误差**（无需额外模型）：

```
history (T 帧) ──backbone.rollout──► pred_slots (未来 H 帧预测)
真实后续观测 slots ──────────────────► true_slots (未来 H 帧真值)
risk_pred = 槽级距离( pred_slots , true_slots )     # MSE / 匈牙利匹配后 MSE
```

**信号② 模型分歧**（epistemic，两种取一或并用）：

```
方式 a（ensemble）：M 个 backbone 副本各自 rollout → 取预测两两方差
方式 b（MC-mask）：同一 history 做 N 次不同 masking 的 prediction → 取方差
risk_epi = var( {pred_1, ..., pred_M/N} )
```

- **槽级距离**：预测槽与真值槽不保证同序，用匈牙利匹配对齐后再算 MSE（复用 C-JEPA 的 `hungarian_cost` / `reorder_slots_to_match_AP`）。
- 两信号可各自归一化后加权合并成一个 risk 分，或分别输出（预测误差偏 aleatoric+OOD，分歧偏 epistemic）。

### 1.3 组装（`spavr/model.py`）

```
class UncertaintyRiskModel:
    backbone: 冻结的 C-JEPA（ensemble 时为 M 份）
    def risk(history, true_future=None):
        pred = backbone.rollout(history)
        r_epi  = disagreement(history)               # 信号②，随时可算
        r_pred = slot_dist(pred, true_future)         # 信号①，需真值才可算
        return combine(r_pred, r_epi)
```

整个模型作为一个单元统一保存 / 加载，产出一个打包 ckpt（ensemble 则打包 M 份 + 组合配置）。

---

## 2. 所需数据集

### 2.1 底座（C-JEPA 现有，按 `episode_id` 对齐）

```
slots:   [T, S, D]      # 预抽取视觉对象槽（slot pkl）
action:  [T, act_dim]   # 原始动作（meta pkl）
proprio: [T, prop_dim]  # 本体感受（meta pkl）
```

### 2.2 标签需求：无

- **不需要任何 reward / 风险标签。** 信号①的"真值"就是轨迹自身的后续观测槽（数据里现成），信号②不需真值。
- **唯一要求：覆盖面广。** 见过的场景即"正常"（预测准、分歧小），没见过的自然预测差、分歧大。数据越多样，"正常区"划得越准，OOD 判别越可靠。

### 2.3 可选：验证用风险标记

- 留几条带风险标记的 episode（碰撞 / 失败等），**仅用于验证**（确认预测误差 / 分歧确实与真实风险正相关），不参与训练。

---

## 3. 训练

方案 B 的 backbone 冻结、无监督信号，"训练"主要是**准备风险度量所需的组件**，按走哪条信号分两种。

### 3.1 路线 1：纯预测误差（零训练）

- backbone 直接用官方 C-JEPA ckpt，**不训练任何东西**。
- 只需在验证集上标定阈值（见 3.3），即可上线。

### 3.2 路线 2：ensemble 分歧（训练 M 份 predictor）

脚本 `train/train_ensemble.py`。

- 训 M 个 predictor 副本，制造多样性来源（任选）：不同随机种子 / 不同 masking 种子 / 数据 bootstrap。
- 每个副本仍是 C-JEPA 原自监督目标（masked-history + future-prediction），**无风险标签**。
- encoder / slot-attention / initializer 始终冻结，只训各自的 predictor（+ 两个小 encoder）。
- 方式 b（MC-mask）无需训练，用单个 backbone 多次不同 masking 前向即可，作为 ensemble 的轻量替代。

### 3.3 阈值标定（两路线共用）

- 在正常轨迹上跑出 risk 分布，取高分位（如 95%）作为"异常/危险"报警阈值。
- 若有 2.3 的验证标记，按"漏报 / 误报"曲线定阈值，而非拍脑袋。

### 3.4 权重产出

- 路线 1：无新增权重，直接用官方 C-JEPA ckpt + 阈值配置。
- 路线 2：打包 M 份 predictor（encoder 部分恒为官方 videosaur，冻结）+ 组合 / 阈值配置为一个 ckpt。

---

## 4. 度量函数

方案 B 无监督，"度量函数"是从预测算出 risk 分的公式，不是训练损失。

### 4.1 预测误差（信号①）

预测未来槽 `p`、真值槽 `g`，先匈牙利匹配对齐槽序，再算距离：

```
risk_pred = (1 / (H·S·D)) Σ_{h,i,d} ( p[h, π(i), d] − g[h, i, d] )²
```

- `π` 为匈牙利匹配得到的槽对应关系；H 预测步数，S 槽数，D 维度。
- 也可只取关键帧（如最后一帧）或对 H 步加权（越远权重越低）。

### 4.2 模型分歧（信号②）

M（或 N）个预测 `{p_m}`，逐槽逐维取方差再聚合：

```
risk_epi = mean_{i,d}  Var_m ( p_m[·, i, d] )
```

- ensemble：M 个副本的预测；MC-mask：同一 backbone N 次不同 masking 的预测。
- 方差大 = 各模型/各次预测互相不认同 = 认知不确定性高。

### 4.3 合并

```
risk = α · norm(risk_pred) + (1−α) · norm(risk_epi)
```

- 各信号先按正常轨迹的统计量归一化（z-score 或分位归一），再加权。
- 也可不合并、分别报：预测误差偏 OOD/aleatoric，分歧偏 epistemic。

---

## 5. 推理

实时监控当前局面：

```
risk_epi  = disagreement(当前历史)          # 随时可算，不需未来真值
risk_pred = slot_dist(rollout(历史), 真实后续)  # 需拿到后续观测才可算（延迟一步）
报警：risk 超过 3.3 标定的阈值
```

| 输出 | 读作 | 用途 |
|---|---|---|
| risk_epi | 认知不确定性 | 高 → 没见过 / 没把握，触发保守策略 |
| risk_pred | 预测失准度 | 高 → 世界超出模型掌控，OOD 报警 |

- **推理不需要任何标签。**
- risk_epi 可**先验**（动作执行前就能算，因为只看模型自己分不分歧）；risk_pred 需等到后续观测到达才能算（**事后核验**）。

---

## 6. 关键超参 / 配置

| 项 | 参考值 | 说明 |
|---|---|---|
| history_size | 3（PushT，与 backbone 对齐） | 历史窗口 |
| 预测步数 H | 1 ~ 数步 | rollout 多远算误差 |
| ensemble 数 M | 3 ~ 5 | 路线 2；越多越稳越贵 |
| MC-mask 次数 N | 5 ~ 10 | 方式 b；ensemble 的轻量替代 |
| 合并权重 α | 需调 | 预测误差 vs 分歧的配比 |
| 报警阈值 | 正常分布 95% 分位起 | 按漏报/误报曲线定 |

---

## 7. 待验证 / 风险点

1. **不确定性≠客观危险**：测的是"模型没把握"，未必等于真出事；需用 2.3 验证标记确认相关性。
2. **rollout 误差累积**：H 越大预测越不准，risk_pred 本底噪声上升；先用小 H。
3. **ensemble 成本**：M 份 backbone 训练 + 推理都翻 M 倍；成本敏感时优先 MC-mask。
4. **归一化基准漂移**：正常轨迹的统计量随环境变化会漂，阈值需定期重标。

---

## 8. 代码落点（均在 `Code/SPAVR/`，不改 `third_party/cjepa`）

- `spavr/backbone.py`：加载并冻结 C-JEPA，暴露 rollout、多次 masking 前向接口。
- `spavr/uncertainty.py`：预测误差（信号①）+ 分歧（信号②）+ 合并 + 阈值标定。
- `spavr/model.py`：组装 `UncertaintyRiskModel`（`risk`）+ 打包保存 / 加载。
- `train/train_ensemble.py`：路线 2，训 M 份 predictor（无标签，C-JEPA 自监督目标）。
- `configs/`：H、M、N、α、阈值等配置。
- `tests/test_smoke.py`：假数据跑通 rollout + 两信号计算。
