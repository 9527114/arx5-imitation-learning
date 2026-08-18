# ARX5 ACT Cache 与训练启用方案

本文档记录当前 ARX5-ACT 训练链路的正确启动方式。当前版本已经把 ACT 的图像读取从“训练时随机 seek mp4”改成了“先构建 zarr 图像 cache，再训练时直接读 cache”，用途类似 Diffusion Policy 里的 dataset cache。

## 结论

当前方案可以用于训练。

已检查 `data_local/glue_motion`：

- 数据集：72 个 episode，13750 个 lowdim step。
- 视频覆盖：只有 2 个尾部 lowdim-camera pair 不覆盖。
- ACT cache 会自动裁剪：
  - episode 19: 181 -> 180
  - episode 26: 197 -> 196
- 训练/验证 split 已修正，不会再把 train 和 val 都错误地扩成全量 episode。
- cache 文件名已加入数据集签名：episode ends、视频文件大小、视频 mtime、相机配置、分辨率都会参与 hash。数据或视频变了会生成新的 cache，不会误用旧 cache。

仍需注意：

- 低维时间戳还有 WARN，说明部分 episode 内存在短暂停顿或不规则 dt。这不影响 cache 构建和训练启动，但后续追求更好部署效果时，可以再做严格清洗数据集。

## 当前实现

相关文件：

- `arx5_act/dataset.py`：ACT dataset 和图像 cache 逻辑。
- `arx5_act/build_cache.py`：单独构建 ACT cache。
- `arx5_act/train_act.py`：ACT 训练入口。
- `scripts/train_act.sh`：推荐使用的 ACT 训练脚本。

cache 生成位置：

```text
diffusion_policy-main/data_local/<dataset_name>/act_cache_<hash>.zarr
```

cache 内容：

```text
images[step, camera, height, width, channel]
episode_valid_ends
attrs:
  camera_names
  image_size
  safety_frames
  complete
```

训练时每个 sample 直接从 zarr 读取：

```text
image:  [camera, channel, height, width]
qpos:   [7]
action: [chunk_size, 7]
is_pad: [chunk_size]
```

## 1. 环境准备

每次新终端先执行：

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
source ./activate_arx5_env.sh
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
```

确认 GPU 可用：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

正常应看到 `True` 和 RTX 4070。

## 2. 检查数据对齐

训练前建议先检查一次：

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
source ./activate_arx5_env.sh
cd diffusion_policy-main

python -m arx5_collector.scripts.check_dataset_alignment data_local/glue_motion
```

当前 `glue_motion` 的结果是：

```text
Video timestamp coverage:
  FAIL
  uncovered lowdim-camera pairs: 2
  episode 19: miss=1
  episode 26: miss=1

Lowdim timestamp regularity:
  WARN
  dt warnings: 61
```

这里的 video FAIL 不是致命问题，因为 ACT cache 会自动按三路相机可用帧裁掉尾部不可覆盖 step。真正需要警惕的是大量 episode 出现很长 `dt_max`，这说明采集时可能有停顿；如果部署效果抖动或不稳定，再考虑做更严格的数据清洗。

## 3. 构建 ACT cache

首次训练前先单独构建 cache：

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp
source ./activate_arx5_env.sh
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp

python -m arx5_act.build_cache \
  --dataset-path diffusion_policy-main/data_local/glue_motion \
  --image-width 320 \
  --image-height 240 \
  --chunk-size 50
```

开始时可能显示：

```text
Building ACT image cache: 0%| | 0/72 [00:00<?, ?it/s]
```

这是正常的。`?it/s` 只是 tqdm 还没估计出速度。第一个 episode 完成后会开始显示速度。

另开终端可查看 cache 是否在增长：

```bash
du -sh /media/star/Elyos_PSSD/ARX5/CY_arx5_dp/diffusion_policy-main/data_local/glue_motion/act_cache_*.zarr
```

构建完成后应看到：

```text
Saved ACT image cache: ...
ACT cache ready: ...
Samples: ...
```

## 4. 启动 ACT 训练

测试版 2 epoch：

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp

setsid bash -lc 'cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp; env \
  RUN_NAME=glue_motion_act_cache_b16_test \
  DATASET_PATH=data_local/glue_motion \
  ACT_EPOCHS=2 \
  ACT_BATCH_SIZE=16 \
  ACT_NUM_WORKERS=0 \
  ACT_CHUNK_SIZE=50 \
  ACT_CHECKPOINT_EVERY=1 \
  ACT_DEVICE=cuda \
  ACT_USE_CACHE=1 \
  ./scripts/train_act.sh' \
  > logs/glue_motion_act_cache_b16_test_nohup.log 2>&1 < /dev/null &
```

正式版 50 epoch：

```bash
cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp

setsid bash -lc 'cd /media/star/Elyos_PSSD/ARX5/CY_arx5_dp; env \
  RUN_NAME=glue_motion_act_cache_b16 \
  DATASET_PATH=data_local/glue_motion \
  ACT_EPOCHS=50 \
  ACT_BATCH_SIZE=16 \
  ACT_NUM_WORKERS=0 \
  ACT_CHUNK_SIZE=50 \
  ACT_CHECKPOINT_EVERY=10 \
  ACT_DEVICE=cuda \
  ACT_USE_CACHE=1 \
  ./scripts/train_act.sh' \
  > logs/glue_motion_act_cache_b16_nohup.log 2>&1 < /dev/null &
```

说明：

- `ACT_BATCH_SIZE=16`：4070 当前更合适，step 数比 batch 4 少约 4 倍。
- `ACT_NUM_WORKERS=0`：避免多 worker 读视频或 zarr 时触发内存/worker killed 问题。cache 稳定后可以再尝试 2。
- `ACT_USE_CACHE=1`：默认开启 cache。
- `ACT_DEVICE=cuda`：强制使用 GPU。

## 5. 监控训练

看日志：

```bash
tail -f /media/star/Elyos_PSSD/ARX5/CY_arx5_dp/logs/glue_motion_act_cache_b16_test_nohup.log
```

看 GPU：

```bash
nvidia-smi
```

看 ckpt：

```bash
find /media/star/Elyos_PSSD/ARX5/CY_arx5_dp/act_outputs/glue_motion_act_cache_b16_test \
  -type f \( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" \)
```

训练输出目录：

```text
act_outputs/<RUN_NAME>/
  config.json
  dataset_stats.pkl
  policy_latest.ckpt
  policy_best.ckpt
  policy_epoch_*.ckpt
```

## 6. 停止训练

查 ACT 进程：

```bash
ps -ef | grep arx5_act.train_act | grep -v grep
```

停止指定 PID：

```bash
kill <PID>
```

如果进程不退出，再用：

```bash
kill -9 <PID>
```

## 7. 常见问题

### 看到 `?it/s`

正常。tqdm 尚未估算速度，不是错误。

### 第一次 cache 构建仍然慢

正常。第一次必须把三路 mp4 解码成 zarr。后续训练会复用 cache，不再随机 seek mp4。

### 训练仍然慢

优先检查：

```bash
nvidia-smi
```

如果显存占用存在但 GPU util 不高，可能是数据读取或 CPU resize 瓶颈。当前 cache 已经去掉 mp4 seek，后续可继续优化：

- 尝试 `ACT_NUM_WORKERS=2`
- 尝试更大的 `ACT_BATCH_SIZE=24` 或 `32`
- 改 cache chunk，让 batch 读取更连续

### 数据集更新后是否需要手动删除旧 cache

一般不需要。cache 文件名包含视频大小、mtime、episode ends、分辨率和相机配置。数据变了会自动生成新 cache。

旧 cache 可以留着，确认不用后再清理。

### 想临时回到旧版直读 mp4

训练时设置：

```bash
ACT_USE_CACHE=0
```

不推荐长期使用，效率低。

## 8. 当前建议流程

1. 先跑 `check_dataset_alignment`。
2. 跑 `python -m arx5_act.build_cache ...`。
3. 跑 2 epoch 测试版，确认 `policy_latest.ckpt` 能保存。
4. 跑 50 epoch 正式版。
5. 用 ACT loader 做离线 dry-run，再接入在线部署。
