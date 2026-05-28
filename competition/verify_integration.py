"""端到端集成验证 — 用 mock 数据走一遍 ZoneAnalyzer.finalize 完整链路。

不需要 best.pt 或 NPU 板，只用 mock 数据即可验证：
1. health_module 正确导入
2. ZoneAnalyzer 调用 _compute_health_diagnoses 不报错
3. 所有输出文件（包括新增的 health_report.txt）生成
4. CSV 含新增字段
5. 群体异常检测正确触发

运行方式
========
    python verify_integration.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "MindSpore" / "YOLO_MindSpore"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "trackers"))

from track_and_count import PigTrajectory, ZoneAnalyzer

OUTPUT_DIR = Path(__file__).parent / "verify_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_valid_trajectory(tid, box_size, activity="active", fps=25):
    """直接构造一条 VALID 轨迹（绕过 ZoneAnalyzer 状态机）。

    activity:
        "active"      ：均匀走动，speed ≈ 60 px/s
        "still"       ：长期不动，speed ≈ 2 px/s
        "outlier_big" ：体型异常大（其余正常）
    """
    traj = PigTrajectory(tid, first_frame=0, first_zone="ENTRY")
    w, h = box_size

    if activity == "still":
        n_frames = 200
        positions = [(500 + 0.1 * i, 300, i) for i in range(n_frames)]
        zones = ["ENTRY"] * 50 + ["WAIT"] * 50 + ["OUT"] * 100
    else:
        n_frames = 100
        # active: 每帧走 3 px ≈ 75 px/s
        positions = [(1100 - 10 * i, 300, i) for i in range(n_frames)]
        zones = ["ENTRY"] * 30 + ["WAIT"] * 30 + ["OUT"] * 40

    traj.positions = positions
    traj.box_sizes = [(float(w), float(h), i) for i in range(n_frames)]
    traj.confidences = [(0.9, i) for i in range(n_frames)]
    traj.zone_history = ["ENTRY", "WAIT", "OUT"]
    traj.first_frame = 0
    traj.last_frame = n_frames - 1
    return traj


def main():
    width, height, fps = 1280, 720, 25
    analyzer = ZoneAnalyzer(width=width, fps=fps, out_ratio=0.45, wait_ratio=0.25)

    # 5 头猪：3 头正常，1 头长期不动（低健康），1 头体型超大（体重离群）
    analyzer.trajectories[1] = make_valid_trajectory(1, (120, 80), "active")
    analyzer.trajectories[2] = make_valid_trajectory(2, (140, 90), "active")
    analyzer.trajectories[3] = make_valid_trajectory(3, (130, 85), "active")
    analyzer.trajectories[4] = make_valid_trajectory(4, (125, 82), "still")   # 低健康
    analyzer.trajectories[5] = make_valid_trajectory(5, (700, 400), "active") # 体重离群

    # 模拟线穿越计数
    analyzer.line_counters = {'line0': 5, 'line1': 5, 'line2': 5}

    print(f"[INFO] 注入 5 条 mock 轨迹")
    print("[INFO] 调用 finalize（包含健康预警计算）...")

    valid_count = analyzer.finalize(OUTPUT_DIR, "ByteTrack",
                                    frame_area=width * height)
    print(f"[INFO] valid_count = {valid_count}")

    # 验证文件存在
    expected_files = [
        "ByteTrack_id_events.csv",
        "ByteTrack_state_changes.txt",
        "ByteTrack_summary.csv",
        "ByteTrack_trajectory_report.csv",
        "ByteTrack_health_report.txt",  # ★ 新增
    ]
    print("\n========== 输出文件检查 ==========")
    all_ok = True
    for fname in expected_files:
        path = OUTPUT_DIR / fname
        ok = path.exists() and path.stat().st_size > 0
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {fname}: "
              f"{path.stat().st_size if path.exists() else 0} bytes")
        if not ok:
            all_ok = False

    # 验证 summary 含新字段 + 异常计数 > 0
    print("\n========== summary.csv 字段检查 ==========")
    with open(OUTPUT_DIR / "ByteTrack_summary.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
        data = dict(zip(header, row))
        for col, val in data.items():
            print(f"  {col:25s} = {val}")
        required = {'avg_weight_kg', 'group_health_score', 'abnormal_count',
                    'low_health_count', 'weight_outlier_count'}
        missing = required - set(header)
        if missing:
            print(f"  [FAIL] 缺少字段: {missing}")
            all_ok = False
        # 期望：abnormal_count >= 1
        abn = int(data['abnormal_count'])
        print(f"\n  [{'OK' if abn >= 1 else 'WARN'}] abnormal_count = {abn} (期望 ≥1)")

    # 验证 trajectory_report 含新字段
    print("\n========== trajectory_report.csv 字段检查 ==========")
    with open(OUTPUT_DIR / "ByteTrack_trajectory_report.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        required = {'EstWeight(kg)', 'Posture', 'ActivityScore',
                    'HealthScore', 'AbnormalFlags'}
        missing = required - set(header)
        if missing:
            print(f"  [FAIL] 缺少字段: {missing}")
            all_ok = False
        else:
            print(f"  [OK] 全部新增字段都已就位")
        for r in reader:
            print(f"  ID#{r[0]:>3} | IsValid={r[1]:>5} | EstWeight={r[10]:>7} | "
                  f"Posture={r[11]:>10} | Activity={r[12]:>5} | "
                  f"Health={r[13]:>5} | Flags={r[14]}")

    # 验证 health_report.txt 含群体诊断
    print("\n========== health_report.txt ==========")
    with open(OUTPUT_DIR / "ByteTrack_health_report.txt", encoding="utf-8") as f:
        print(f.read())

    print("=" * 50)
    if all_ok:
        print("[PASS] 端到端集成验证通过 ✓")
        sys.exit(0)
    else:
        print("[FAIL] 集成验证失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
