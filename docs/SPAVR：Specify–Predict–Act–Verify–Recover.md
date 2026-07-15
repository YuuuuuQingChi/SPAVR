# SPAVR：Specify–Predict–Act–Verify–Recover

ReAct 的核心是：

```text
Reason → Act → Observe
```

它适合软件环境，因为动作可以试错，观察能及时反馈。

具身 agent 更需要的是：

```text
Specify → Predict → Act → Verify → Recover
```

建议把它概括为：

**SPAVR：Specify–Predict–Act–Verify–Recover**

```text
Instruction
  → Specify
  → Predict
  → Act
  → Verify
  → Recover / Continue
```

也可以简称为：

```text
SPAVR Loop
```

读作 “saver”，含义上也对应 embodied safety。

------

## **1. Specify：把任务转成可执行、可校验、可约束的目标**

对应第一个问题：

**如何从具身任务指令到可校验的 goal，并对危险因素进行初步判断以调整计划？**

在 ReAct 中，goal 往往是隐含在 prompt 里的；但在具身任务中，goal 必须被显式化。

Specify 阶段要生成：

```text
目标状态
成功条件
失败条件
安全约束
不确定性条件
恢复策略
```

也就是把自然语言任务变成一个 **Goal Contract**。

例如：

```text
用户：把杯子放到桌子上

Specify:
- 目标物：杯子
- 目标状态：杯子位于桌面、直立、稳定、未被夹爪持有
- 成功条件：连续 N 帧检测到杯子稳定在桌面上
- 安全约束：不能碰撞、不能倾倒、不能放在桌边
- 不确定性：若杯子识别置信度低，需要重新观察
```

所以 Specify 不是普通 planning，而是：

把“我要做什么”转化为“什么状态才算做对，什么状态必须避免”。

------

## **2. Predict：在行动前和行动中预测风险与后果**

对应第二个问题：

**如何在 action 过程中保持高频短窗口危险预测以对危险行为进行避免与纠正？**

这是 SPAVR 和 ReAct 最大的区别。

ReAct 是：

```text
先行动，再观察结果
```

而具身 agent 必须是：

```text
行动前先预测，行动中持续预测
```

Predict 阶段关注：

```text
如果执行这个 action chunk，短时间后会发生什么？
是否可能碰撞？
是否可能滑落？
是否可能越界？
是否会破坏 goal contract？
是否会让任务进入不可恢复状态？
```

因此 Predict 不只是高层推理，而是一个短窗口安全预测器：

```text
current observation + robot state + action chunk + goal contract  
```

例如：

```text
准备抓杯子前：
- 预测夹爪路径是否会碰到旁边物体
- 预测抓取点是否可能导致杯子倾斜
- 预测夹爪力度是否过大
- 预测抬升后杯子是否可能滑落
```

这一阶段让具身 agent 从 **reactive agent** 变成 **predictive agent**。

------

## **3. Act：不是执行单个动作，而是执行受约束的 action chunk**

在具身任务中，Act 不应该是：

```text
执行一个动作，然后等观察
```

而应该是：

```text
执行一个受约束、可中断、可修正的 action chunk
```

例如：

```text
move_to_grasp_pose()
close_gripper()
lift_3cm_probe()
verify_grasp()
continue_lift()
```

Act 阶段应当满足三个条件：

```text
bounded：动作有边界
interruptible：动作可中断
correctable：动作可纠正
```

这非常重要，因为具身 action 不可完全回溯，所以不能让 policy 无条件输出长动作。

因此，SPAVR 中的 Act 不是“盲目执行”，而是：

在 Goal Contract 和 Risk Prediction 约束下执行短动作块。

------

## **4. Verify：验证任务意图、状态变化和下一步前提**

对应第三个问题：

**如何处理观测与行动之间的 gap，对任务意图以及状态进行有效校验？**

ReAct 中的 Observe 通常只是“拿到环境反馈”。
 但具身 agent 的 Observe 不够，因为观察本身可能不可靠。

所以 SPAVR 中不是 Observe，而是 Verify。

Verify 要检查三件事：

```text
Intent Alignment：当前状态是否仍然符合任务意图？
Transition Consistency：状态变化是否符合刚才 action 的预期？
Precondition Validity：下一步动作的前提是否仍然成立？
```

例如机器人抓杯子后：

```text
不是只问：杯子还在画面里吗？

而是问：
- 抓到的是不是目标杯子？
- 杯子是不是跟随夹爪移动？
- 杯子有没有倾斜或滑动？
- 当前状态是否支持下一步“放到桌面”？
- 桌面区域是否仍然可见、安全、可达？
```

因此 Verify 是对 observation 的主动解释，而不是被动接收。

------

## **5. Recover：失败不是终点，而是进入恢复或重规划**

ReAct 中，如果动作失败，通常继续 reasoning。
但具身 agent 的失败有不同等级：

```text
轻微偏差：修正动作
不确定：重新观察
局部失败：恢复
严重风险：停止
不可恢复：终止并报告
```

Recover 阶段要决定：

```text
continue
reobserve
correct
retry
rollback-to-safe-state
replan
abort
ask-for-help
```

例如：

```text
如果杯子轻微滑动：
→ 暂停，降低速度，重新夹紧

如果目标被遮挡：
→ 移动视角重新观察

如果人手进入工作区：
→ 停止并等待

如果物体掉落：
→ 中止当前任务，进入恢复流程
```

Recover 使 agent 不再是简单循环，而是有安全状态机。

------

## **6. SPAVR 和 ReAct 的关系**

可以这样对比：

| **范式**               | **核心循环**                               | **适用场景**                 | **关键假设**                                |
| ---------------------- | ------------------------------------------ | ---------------------------- | ------------------------------------------- |
| ReAct                  | Reason → Act → Observe                     | 软件任务、文本环境、工具调用 | action 可试错，反馈清晰                     |
| Goal-Action-Evaluation | Goal → Action → Evaluation                 | 一般 agent loop              | goal 可直接评估                             |
| SPAVR                  | Specify → Predict → Act → Verify → Recover | 具身任务、机器人、物理环境   | action 不可回溯，观察有 gap，风险需提前预测 |

ReAct 的强大在于它把 LLM agent 简化成：

```text
思考—行动—观察
```

SPAVR 的目标则是把 embodied agent 简化成：

```text
规定目标—预测后果—安全行动—验证状态—恢复闭环
```

------

## **7. 为什么这个范式是“正确”的**

一个具身 agent 范式要满足五个必要条件：

**第一，目标必须可验证**

否则 agent 无法知道任务是否完成。

所以需要 Specify。

**第二，动作必须先预测**

否则不可回溯 action 会导致危险。

所以需要 Predict。

**第三，执行必须可中断**

否则高层 agent 无法干预连续物理动作。

所以需要 Act as bounded action chunk。

**第四，观察必须被校验**

否则 perception error 会被当成真实状态。

所以需要 Verify。

**第五，失败必须可恢复**

否则长程任务会因为局部错误直接崩溃。

所以需要 Recover。

因此 SPAVR 不是任意拼接五个词，而是由 embodied agent 的基本约束推导出来的：

```text
physical irreversibility → Predict
partial observability → Verify
ambiguous instruction → Specify
continuous action → bounded Act
long-horizon uncertainty → Recover
```

------

## **8. 更简洁的表达方式**

如果希望像 ReAct 一样极简，可以写成一句话：

**SPAVR is an embodied agent paradigm that specifies verifiable goals, predicts short-horizon consequences, executes bounded actions, verifies grounded state transitions, and recovers from unsafe or inconsistent states.**

中文：

**SPAVR 是一种具身 agent 范式：将任务指定为可验证目标，预测短时物理后果，执行有边界动作，校验状态转移，并在不安全或不一致时恢复。**

更短：

```text
Specify the goal.
Predict the consequence.
Act with bounds.
Verify the state.
Recover safely.
```

------

## **9. 作为范式时的图示**

可以画成：

```text
        ┌────────────────────────────┐
        │        Instruction          │
        └──────────────┬─────────────┘
                       ↓
        ┌────────────────────────────┐
        │ Specify: Goal Contract      │
        └──────────────┬─────────────┘
                       ↓
        ┌────────────────────────────┐
        │ Predict: Risk & Outcome     │◄──────────────┐
        └──────────────┬─────────────┘               │
                       ↓                             │
        ┌────────────────────────────┐               │
        │ Act: Bounded Action Chunk   │               │
        └──────────────┬─────────────┘               │
                       ↓                             │
        ┌────────────────────────────┐               │
        │ Verify: State & Intent      │               │
        └──────────────┬─────────────┘               │
                       ↓                             │
        ┌────────────────────────────┐               │
        │ Recover / Continue / Replan │───────────────┘
        └────────────────────────────┘
```

------

## **10. 对应到具身 agent 的模块**

```text
SPAVR Loop
├── Specify
│   ├── instruction parser
│   ├── goal contract generator
│   ├── safety constraint extractor
│   └── success/failure predicate builder
│
├── Predict
│   ├── short-horizon world predictor
│   ├── action-chunk risk scorer
│   ├── collision / force / stability checker
│   └── uncertainty estimator
│
├── Act
│   ├── VLA policy
│   ├── skill runtime
│   ├── bounded action executor
│   └── interrupt controller
│
├── Verify
│   ├── perception grounding
│   ├── state predicate checker
│   ├── transition consistency verifier
│   └── intent alignment verifier
│
└── Recover
    ├── correction policy
    ├── reobserve policy
    ├── local retry policy
    ├── safe stop
    └── high-level replanner
```

这可以直接作为系统架构写进论文或 PhyAgentOS 设计文档。

------

## **11. 范式级别的核心主张**

可以把这个范式的主张写成三句话：

### **观点一**

ReAct treats action as reversible interaction; embodied agents must treat action as irreversible physical intervention.

### **观点二**

Therefore, embodied agent loops should not be post-hoc evaluative, but predictive and verifiable.

### **观点三**

SPAVR provides a general loop: specify verifiable goals, predict risks before acting, execute bounded actions, verify grounded state transitions, and recover safely.

中文：

ReAct 将 action 视为可试错的交互，而具身 agent 必须将 action 视为不可完全回溯的物理干预。因此，具身 agent 的循环不应是事后评价式，而应是预测式、可验证式和可恢复式。SPAVR 提供了这样一种通用循环：指定可验证目标、预测风险、执行有边界动作、校验状态转移，并安全恢复。

------

## **12. 和原始三个问题的对应关系**

三个问题可以非常自然地落到 SPAVR：

| **你的问题**                             | **SPAVR 中的位置** |
| ---------------------------------------- | ------------------ |
| 如何从任务指令到可校验 goal？            | Specify            |
| 如何初步判断危险因素并调整计划？         | Specify + Predict  |
| 如何在 action 过程中高频短窗口危险预测？ | Predict + Act      |
| 如何避免和纠正危险行为？                 | Predict + Recover  |
| 如何处理 observation-action gap？        | Verify             |
| 如何校验任务意图和状态？                 | Verify + Recover   |

所以 SPAVR 不是外加的框架，而是对问题的直接抽象。

------

## **13. 最后建议**

可以把你的整体研究方向收束为：

**SPAVR: A Predictive and Verifiable Agent Loop for Embodied Intelligence**

或者中文：

**SPAVR：面向具身智能的预测式可验证 Agent 循环范式**

它足够简洁，能像 ReAct 一样作为范式表达；同时又比 ReAct 更准确地覆盖具身系统的关键困难：

```text
任务歧义 → Specify
物理不可回溯 → Predict
连续动作 → Act with bounds
观测行动 gap → Verify
长程失败累积 → Recover
```

如果进一步做系统实现，最核心的落点应该是：

```text
Goal Contract + Action-Chunk Risk Prediction + Intent-State Verification
```

这三者是 SPAVR 能区别于普通 “LLM planner + robot policy + verifier” 的关键。