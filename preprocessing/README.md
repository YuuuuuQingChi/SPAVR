# SPAVR 数据预处理使用教程

这个脚本把原始机器人 episode 转换成 SPAVR 可以直接训练的数据。

处理后会得到：

- 统一帧率和尺寸的外部相机视频；
- 相机坐标系下的 `proprio` 和 `action`；
- 每一步的 reward 和 timestamp；
- train/val/test 划分；
- 仅由 train split 计算的标准化统计量。

## 1. 准备环境

```bash
conda activate spavr
cd /data7/yuqingchi/Code/SPAVR
python -m pip install -e .
```

## 2. 先做 dry-run

dry-run 只检查数据，不生成视频和轨迹文件：

```bash
spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /tmp/spavr_unused \
  --ambiguous-view-policy skip \
  --workers 4 \
  --dry-run
```

建议先确认命令正常结束，再开始完整处理。

## 3. 处理完整数据集

```bash
spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /data7/yuqingchi/Code/SPAVR/processed_spavr_test \
  --ambiguous-view-policy skip \
  --workers 4
```

第一次运行时，`--output-root` 必须不存在或为空。

如果运行中途报错或被中断，不需要删除已经生成的数据。修复问题后增加
`--resume-output`：

```bash
spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /data7/yuqingchi/Code/SPAVR/processed_spavr_test \
  --ambiguous-view-policy skip \
  --workers 4 \
  --resume-output
```

续跑时会先校验已有 episode，只处理缺失的文件，最后重新生成索引和标准化统计量。
续跑参数应与第一次运行保持一致。

## 4. 只处理部分数据

只处理 failure：

```bash
spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /data7/yuqingchi/Code/SPAVR/processed_spavr_failure \
  --outcomes failure \
  --ambiguous-view-policy skip \
  --workers 4
```

只处理指定任务：

```bash
spavr-preprocess \
  --input-root /data2/chenzexin/A_Projection/SPAVR_V1/dataset \
  --output-root /data7/yuqingchi/Code/SPAVR/processed_pickcube \
  --tasks PickCube-apple PickCube-lock \
  --ambiguous-view-policy skip \
  --workers 4
```

failure 数据还可以增加 `--balance-failure-phases`，裁短过长的失败前稳定片段：

```bash
--outcomes failure --balance-failure-phases
```

## 5. 相机坐标系

脚本优先读取 `annotation.json` 中的：

```text
data.videos.<view>
data.camera.<view>
```

例如选择 `primary` 视频，就会读取 `primary` 对应的 `T_cw`。处理后的末端状态和
动作前 6 维都会统一到这个 OpenCV 相机坐标系，第 7 维夹爪数据保持不变。

如果旧数据没有 `data.videos`，脚本才会根据视频文件名选择视角。

存在多个外部视角时，可以使用 `--ambiguous-view-policy skip` 跳过，也可以准备一个
JSON 文件明确指定视角：

```json
{
  "SpinStack-gen1/failure/SpinStack-gen1-000071": "front",
  "SpinStack-gen1/failure/SpinStack-gen1-000145": "side"
}
```

然后传入：

```bash
--view-map /path/to/view_map.json
```

## 6. 输出内容

```text
processed_spavr_test/
├── episodes/            # 每条 episode 的 proprio/action/reward/timestamp
├── videos/              # 处理后的外部相机视频
├── manifest.jsonl       # episode 索引和所选视角
├── metadata.json        # 处理参数及跳过记录
├── normalization.json   # train split 的 action/proprio mean 和 std
├── splits.json          # train/val/test episode 列表
└── task_vocab.json      # task 名称与编号
```

训练时 Dataset 会读取 `normalization.json`，对 action 和 proprio 执行：

```text
(x - mean) / std
```

## 7. 自动跳过的数据

以下数据会输出 warning 并跳过：

- 当前用户没有读取权限的 episode；
- 使用 `--ambiguous-view-policy skip` 时的多外部视角 episode；
- `failure_onset_step >= failure_end_step` 的无有效长度失败区间。

跳过列表会记录在 dry-run 输出和最终的 `metadata.json` 中。

缺少相机标定、外参不合法、轨迹长度严重不匹配等数据错误仍然会停止运行。

## 8. 常用参数

```text
--target-fps 10              输出帧率，默认 10 FPS
--image-size 224 224         输出视频尺寸
--workers 4                  并行 episode 数量
--outcomes success failure   选择成功/失败数据
--tasks TASK ...             只处理指定任务
--length-percentiles 1 99    按处理后长度过滤异常 episode
--ambiguous-view-policy skip 跳过无法自动选择视角的数据
--resume-output              从已有输出目录继续处理
--ffmpeg-bin auto            自动选择支持目标编码器的 FFmpeg
```

当前 conda 环境中的 FFmpeg 可能没有 `libx264`。默认 `auto` 会自动选择支持它的系统
FFmpeg；如有需要也可以显式指定：

```bash
--ffmpeg-bin /usr/bin/ffmpeg
```

查看全部参数：

```bash
spavr-preprocess --help
```

更详细的数据字段定义见
[SPAVR 数据集预处理与训练接口](../docs/SPAVR数据集预处理与训练接口.md)。
