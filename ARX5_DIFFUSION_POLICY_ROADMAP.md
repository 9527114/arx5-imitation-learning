# ARX5 扩散策略迁移路线图

本工作空间包含两个上游风格的代码树：

- `arx5-sdk-main`：ARX5 机器人SDK，包含Python示例、SpaceMouse遥操作、控制器绑定
- `diffusion_policy-main`：原始扩散策略实现，包含真实的UR5数据采集功能

迁移目标是保持扩散策略训练和数据集接口基本不变，仅替换真实硬件层和采集脚本。

## 目标系统

- 机器人：通过 `arx5_interface.Arx5CartesianController` 控制的ARX5机械臂
- 遥操作输入：SpaceMouse，基于 `arx5-sdk-main/python/examples/spacemouse_teleop.py`
- 摄像头：三个RGB视频流：
  - 两个Intel RealSense摄像头
  - 一个USB摄像头
- 数据集格式：DP真实世界格式：
  - `replay_buffer.zarr`
  - `videos/<episode_id>/<camera_idx>.mp4`
  - 低维数据键，如 `robot_eef_pose`、`robot_joint`、`action`、`timestamp`、`stage`
  - 图像键，如 `camera_0`、`camera_1`、`camera_2`

## 当前DP真实世界入口点

- 采集脚本：`diffusion_policy-main/demo_real_robot.py`
- 评估脚本：`diffusion_policy-main/eval_real_robot.py`
- 环境封装器：`diffusion_policy-main/diffusion_policy/real_world/real_env.py`
- 待替换的UR5控制器：`diffusion_policy-main/diffusion_policy/real_world/rtde_interpolation_controller.py`
- RealSense摄像头栈：`diffusion_policy-main/diffusion_policy/real_world/multi_realsense.py`
- 数据集转换：`diffusion_policy-main/diffusion_policy/real_world/real_data_conversion.py`
- 数据集加载器：`diffusion_policy-main/diffusion_policy/dataset/real_pusht_image_dataset.py`
- 训练任务配置：`diffusion_policy-main/diffusion_policy/config/task/real_pusht_image.yaml`

## 阶段一：环境搭建

使用DP真实环境作为基础，因为训练和真实数据工具链都固定在Python 3.9、PyTorch 1.12、zarr、pyrealsense2、av、spnav和ur-rtde-era依赖上。

任务：

1. 从 `diffusion_policy-main/conda_environment_real.yaml` 创建DP真实conda环境
2. 以可编辑模式安装 `diffusion_policy-main`
3. 在同一环境中安装ARX5 Python接口，优先使用 `pip install arx5-interface`
4. 仅在pip安装不足时添加本地SDK路径：
   - `PYTHONPATH=/media/star/Elyos_PSSD/ARX5/CY_arx5_dp/arx5-sdk-main/python`
   - `LD_LIBRARY_PATH=/media/star/Elyos_PSSD/ARX5/CY_arx5_dp/arx5-sdk-main/lib/x86_64`
5. 验证导入：
   - `import diffusion_policy`
   - `import arx5_interface`
   - `import pyrealsense2`
   - `import spnav`
   - `import cv2`
6. 验证硬件服务和权限：
   - SpaceMouse：`spacenavd` 服务运行中
   - RealSense：两个序列号均可见
   - USB摄像头：通过OpenCV可见
   - ARX5 CAN/EtherCAT接口可访问

退出标准：

- 单个conda环境可以导入DP、ARX5 SDK、RealSense、OpenCV和SpaceMouse依赖
- ARX5 SDK示例可以读取机器人状态（不执行危险运动）
- RealSense测试可以列出并流式传输两个RealSense摄像头
- OpenCV可以流式传输USB摄像头

## 阶段二：ARX5控制器适配器

创建与DP兼容的ARX5控制器类，提供与 `RealEnv` 使用的外部API相同的接口：

- `start(wait=True)`
- `stop(wait=True)`
- `start_wait()`
- `stop_wait()`
- `is_ready`
- `get_state(k=None, out=None)`
- `get_all_state()`
- `schedule_waypoint(pose, target_time)`

推荐的新文件：

- `diffusion_policy-main/diffusion_policy/real_world/arx5_interpolation_controller.py`

该类应封装 `Arx5CartesianController`，将ARX5的 `EEFState` 和 `JointState` 转换为DP风格的状态字典，并写入 `SharedMemoryRingBuffer`。

状态键映射应尽可能保留现有的DP命名：

- `ActualTCPPose`：当前ARX5 EEF位姿，形状 `(6,)`
- `ActualTCPSpeed`：估计的EEF速度，形状 `(6,)`
- `ActualQ`：当前关节位置，形状 `(joint_dof,)`
- `ActualQd`：当前关节速度，形状 `(joint_dof,)`
- `TargetTCPPose`：目标ARX5 EEF指令，形状 `(6,)`
- `TargetTCPSpeed`：估计的目标EEF速度，形状 `(6,)`
- `TargetQ`：目标关节位置（如有），否则为当前/指令关节
- `TargetQd`：目标关节速度（如有）
- `robot_receive_timestamp`：壁钟时间戳

退出标准：

- 一个最小脚本可以实例化适配器，读取 `get_state()`，并调度一个安全的小目标点
- `RealEnv.get_robot_state()` 可以不关心机器人是UR5还是ARX5而正常工作

## 阶段三：混合三摄像头栈

原始的 `MultiRealsense` 仅处理RealSense设备。对于两个RealSense加一个USB摄像头，需要添加一个DP兼容的混合摄像头封装器。

推荐的新文件：

- `diffusion_policy-main/diffusion_policy/real_world/single_usb_camera.py`
- `diffusion_policy-main/diffusion_policy/real_world/multi_camera.py`

混合封装器应匹配 `RealEnv` 使用的 `MultiRealsense` 子集接口：

- `n_cameras`
- `is_ready`
- `start/stop/start_wait/stop_wait`
- `get(k=None, out=None)` 返回 `{camera_idx: {'color': T,H,W,C, 'timestamp': T}}`
- `get_vis(out=None)`
- `start_recording(video_path, start_time)`
- `stop_recording()`
- `restart_put(start_time)`
- 曝光和白平衡的no-op或逐摄像头实现（如适用）

退出标准：

- `get_obs()` 返回 `camera_0`、`camera_1`、`camera_2`
- 剧集视频文件夹包含 `0.mp4`、`1.mp4`、`2.mp4`
- 时间戳是单调的，并且在所选采集频率下能够较好地对齐

## 阶段四：ARX5真实环境

创建ARX5特定的环境封装器，而不是破坏性地重写 `real_env.py`。

推荐的新文件：

- `diffusion_policy-main/diffusion_policy/real_world/arx5_real_env.py`

可以先复制 `RealEnv` 并替换：

- 将UR5的 `RTDEInterpolationController` 替换为ARX5控制器适配器
- 将 `MultiRealsense` 替换为混合摄像头封装器
- 将CLI参数从 `robot_ip` 替换为ARX5特定的 `model` 和 `interface`

保持DP输出结构不变。

退出标准：

- `Arx5RealEnv.get_obs()` 返回DP兼容的观测数据
- `start_episode/end_episode` 写入replay buffer和视频
- `real_data_to_replay_buffer()` 可以转换数据集

## 阶段五：ARX5演示采集脚本

基于 `demo_real_robot.py` 创建采集脚本。

推荐的新文件：

- `diffusion_policy-main/demo_arx5_robot.py`

改动：

- 将 `RealEnv` 导入替换为 `Arx5RealEnv`
- 将 `--robot_ip` 替换为：
  - `--model`，例如 `X5` 或 `L5`
  - `--interface`，例如 `can0` 或以太网接口
  - 摄像头设备配置
- 重用SpaceMouse命令生成
- 保留录制控制：
  - `c`：开始剧集
  - `s`：停止剧集
  - `q`：退出
  - 退格键：删除最新剧集

退出标准：

- 可以采集一个短剧集
- 剧集包含同步的机器人状态、动作、阶段、时间戳和三个摄像头视频

## 阶段六：三摄像头的训练配置

创建ARX5任务配置，而不是直接编辑 `real_pusht_image.yaml`。

推荐的新文件：

- `diffusion_policy-main/diffusion_policy/config/task/arx5_real_image.yaml`

预期的形状元数据：

- `camera_0`、`camera_1`、`camera_2`：RGB，`[3, H, W]`
- `robot_eef_pose`：完全笛卡尔控制为 `[6]`，或有意模仿原始PushT仅xy行为时为 `[2]`
- `action`：完全位姿动作为 `[6]`，或仅xy时为 `[2]`

退出标准：

- 数据集缓存成功构建
- 一条训练命令可以启动并加载ARX5数据集，无键值/形状错误

## 阶段七：在ARX5上评估策略

基于 `eval_real_robot.py` 创建ARX5评估脚本。

推荐的新文件：

- `diffusion_policy-main/eval_arx5_robot.py`

应使用相同的 `Arx5RealEnv`、相同的观测预处理，以及训练产生的动作形状。

退出标准：

- 人工交接模式正常工作
- 策略推理产生预期形状的动作
- 机器人在监督下可以执行低速有界指令

## 安全注意事项

- 在任何运动前确认ARX5型号（`X5` 或 `L5`）
- 从低 `max_pos_speed`、低 `max_rot_speed` 和较小的工作空间边界开始
- 在启用长时间遥操作或策略部署前添加软件工作空间限制
- 所有机器人运动期间保持物理急停按钮可触及
- 在数据采集循环和replay buffer离线验证完成之前，不要启用策略控制

## 下一步立即行动

完成阶段一：构建并验证一个可以同时导入DP和ARX5 SDK依赖的Python环境。