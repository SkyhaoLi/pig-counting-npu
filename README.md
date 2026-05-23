# 猪只智能计数系统

基于 YOLOv8 + BYTETracker + 三线中位数投票的猪只自动计数系统，部署在华为 Atlas 200I DK A2 (Ascend 310B4) NPU 开发板上，支持 RTSP 摄像头实时计数和网页监控。

## 系统架构

```
RTSP摄像头/视频上传 → 抓帧 → 跳帧优化 → YOLOv8n NPU推理(~33ms) → 蓝色物体过滤 → BYTETracker多目标跟踪 → 双向穿线计数 → Web实时展示
```

### 核心算法：双向穿线计数 (bidir)

在画面中设置 3 条竖直计数线（25% / 35% / 45% 位置），对每个跟踪 ID：

- **右→左穿线**：计数 +1（同一 ID 每条线只计一次）
- **左→右穿线**：计数 -1（仅当该 ID 之前已 +1，防止折返重复计数）
- **最终计数** = 三条线计数的**中位数**（消除单线噪声）

### 网页监控功能

- 浏览器上传视频文件进行离线推理
- RTSP 摄像头实时监控（MJPEG 流推送）
- 三栏布局：视频流 + 实时统计 + 推理历史
- 推理历史持久化，支持诊断报告生成与下载
- 单条记录删除
- 统一 PigCountingAgent 提供运行状态监控与异常检测

### 跳帧优化

- 每 2 帧执行 1 次检测+追踪（`skip_interval=2`）
- 跳过帧仍更新画面显示，不影响视频流畅度
- ByteTrack 卡尔曼滤波在跳帧间隙自动预测位置，追踪精度几乎无损


## 项目结构

```
├── Jin的U盘资料/YOLO_MindSpore/
│   ├── track_and_count.py          # PC端计数主脚本 (PyTorch YOLO + ByteTrack)
│   ├── pig_counting_agent.py       # 统一Agent (监控/诊断/人工复核)
│   ├── npu_detector.py             # NPU推理封装类 (ACL接口)
│   ├── deploy_to_atlas.py          # SSH/SFTP自动部署到Atlas板子
│   ├── diagnose_existing_outputs.py # 为既有输出补生成诊断TXT报告
│   ├── batch_rerun_group*.py       # PC端批量处理脚本
│   ├── review_agent.py             # 人工复核Agent入口
│   ├── diagnosis_agent.py          # 诊断Agent入口
│   ├── human_review.py             # 人工复核工具函数
│   ├── review_registry.json        # 人工复核修正记录
│   ├── 项目说明.txt                 # 详细文件说明
│   │
│   ├── deploy_atlas/               # Atlas板子部署包
│   │   ├── web_monitor.py          # 实时网页监控系统 (含跳帧优化)
│   │   ├── track_and_count_npu.py  # NPU版计数主脚本
│   │   ├── npu_detector.py         # NPU检测器 (ACL + OM模型)
│   │   ├── batch_run_npu.py        # NPU端批量处理
│   │   ├── autonomous_agent.py     # 自主运维Agent
│   │   ├── bootstrap_board.sh      # 板端环境初始化脚本
│   │   └── trackers/               # ByteTrack追踪器副本
│   │
│   └── trackers/                   # ByteTrack追踪器
│       └── byte_tracker/
│           ├── byte_tracker.py     # BYTETracker主体
│           ├── basetrack.py        # 轨迹基类 & 状态机
│           ├── kalman_filter.py    # 卡尔曼滤波
│           └── matching.py         # IoU匹配 & 匈牙利算法
│
├── paper_assets/                   # 论文素材生成脚本
├── 同步教程acl.docx                 # ACL环境配置教程
└── README.md
```

## 运行环境

### PC 端

- Python 3.10+
- 依赖：`ultralytics`, `opencv-python`, `numpy`, `tqdm`
- 模型：YOLOv8n 训练权重 (`best.pt`)

### Atlas 板端

- 硬件：Atlas 200I DK A2 (Ascend 310B4 NPU)
- 系统：CANN 7.0.RC1, Python 3.10.6
- 模型：`yolov8n_pig_fp16.om`（FP16 量化）
- 依赖：`opencv-python`, `numpy`
- 初始化脚本：`deploy_atlas/bootstrap_board.sh`

### 摄像头

- 大华 IP 摄像头，RTSP 协议接入

## 快速开始

### PC 端推理

```bash
python track_and_count.py \
  --video_path 视频路径.mp4 \
  --model_path best.pt \
  --output_dir ./output
```

### 部署到 Atlas 板子

```bash
# 1. PC 端自动上传部署包、OM 模型，并执行板端自检
python deploy_to_atlas.py

# 2. SSH 到板子；如果是全新板子，先手动再跑一遍初始化脚本确认环境
ssh HwHiAiUser@192.168.137.100
cd ~/pig_counting
./bootstrap_board.sh

# 3. 启动网页监控
python3 web_monitor.py \
  --video datasets/group4/1-12头.mp4 \
  --om models/yolov8n_pig_fp16.om

# 4. 浏览器打开
# http://192.168.137.100:8080
```

如果只想上传代码和模型，不上传任何本地视频：

```bash
python deploy_to_atlas.py --skip-datasets
```

如果只想部署，不立刻执行板端初始化脚本：

```bash
python deploy_to_atlas.py --skip-bootstrap
```

### NPU 批量测试

```bash
# 在板子上
python3 batch_run_npu.py \
  --om models/yolov8n_pig_fp16.om \
  --video_dir videos/
```

## 技术细节

### 检测

- YOLOv8n 单类别（猪）检测，640x640 输入
- FP16 量化后通过 ATC 转为华为 .om 格式
- NPU 推理 ~33ms/帧

### 过滤

- HSV 色彩空间蓝色物体过滤（排除蓝色桶/管道等非猪目标）
- 蓝色像素占比 > 30% 的检测框被丢弃

### 跟踪

- BYTETracker：卡尔曼滤波预测 + 两阶段 IoU 关联
- 高置信度检测优先匹配，低置信度检测二次关联
- `track_buffer=90`（丢失后保留约 6 秒）

### 计数

- 3 条计数线取中位数，抑制单线误触发
- 双向计数自动抵消折返猪只
- Ghost ID 过滤：存活帧数 < 5 的短命轨迹不参与计数

### 跳帧策略

- `skip_interval=2`：每 2 帧做 1 次 NPU 推理，有效帧率翻倍
- 跳过帧仅跳过检测和追踪更新，画面仍正常推送
- ByteTrack 卡尔曼滤波在间隙帧自动预测轨迹位置
- 猪只移动速度慢，跳 1 帧对追踪精度几乎无影响

### 输出文件

每次推理完成后生成：

| 文件 | 内容 |
|------|------|
| `ByteTrack_id_events.csv` | 每个 ID 的出现/区域变化事件日志 |
| `ByteTrack_state_changes.txt` | 各 ID 轨迹详情与有效性判定 |
| `ByteTrack_trajectory_report.csv` | 轨迹分析汇总 |
| `ByteTrack_summary.csv` | 三线计数与最终结果 |
| `ByteTrack_diagnosis.txt` | 诊断报告（异常检测、建议） |
