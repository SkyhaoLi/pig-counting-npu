# 18.mp4 诊断报告

- 主因：反向进入目标或非统计方向目标混入
- 次要原因：入口区停留或折返导致计数波动
- 诊断置信度：0.91

## 证据

- Ghost(Started in OUT) 轨迹 6 个，前 3 秒在 OUT 区新生 ID 0 个。
- Retried 轨迹 3 个，Stuck in Wait 轨迹 0 个。

## 可疑时间窗口

- 反向进入或方向冲突：8.20s - 21.10s，轨迹 [27, 40, 69, 75, 118, 119]
- 停留/折返风险：0.00s - 21.90s，轨迹 [1, 6, 74]

## 汇总

- total_line: 106
- valid_traj: 106
- total_ids: 116
- raw_error: None
- paper_error: None
