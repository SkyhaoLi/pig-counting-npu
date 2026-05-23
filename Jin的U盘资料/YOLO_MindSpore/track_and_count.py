#!/usr/bin/env python3
"""
猪只追踪与计数系统 - MindSpore + ByteTrack
使用 PyTorch YOLO 检测 + ByteTrack 追踪

输出:
1. 带标注的结果视频（包含3条分隔线、线穿越次数、TOTAL计数）
2. ID事件日志 CSV
3. 状态变化记录 TXT
4. 轨迹分析报告 CSV

计数逻辑:
- 使用3条分隔线统计穿越次数
- TOTAL = round((Line0 + Line1 + Line2) / 3)  # 四舍五入
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
from collections import defaultdict

# 导入ByteTrack追踪器（使用原项目完整实现）
from trackers.byte_tracker.byte_tracker import BYTETracker
from types import SimpleNamespace

# 导入YOLO
from ultralytics import YOLO


class PigTrajectory:
    """跟踪单只猪的轨迹和状态变化"""

    def __init__(self, track_id, first_frame, first_zone):
        self.track_id = track_id
        self.zone_history = [first_zone]
        self.state_changes = []
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.status = "ACTIVE"

        # 扩展调试信息
        self.positions = []  # [(cx, cy, frame)]
        self.box_sizes = []  # [(w, h, frame)]
        self.confidences = []  # [(score, frame)]
        self.lost_frames = []  # 丢失帧
        self.recovered_count = 0
        self.last_cx = None

    def add_point(self, zone, frame_idx, fps, cx=None, cy=None, w=None, h=None, conf=None):
        # 检测丢失帧
        if frame_idx - self.last_frame > 1:
            for lost_f in range(self.last_frame + 1, frame_idx):
                self.lost_frames.append(lost_f)
            self.recovered_count += 1

        self.last_frame = frame_idx

        # 记录位置
        if cx is not None:
            self.positions.append((cx, cy, frame_idx))
            self.last_cx = cx
        if w is not None:
            self.box_sizes.append((w, h, frame_idx))
        if conf is not None:
            self.confidences.append((conf, frame_idx))

        # 区域变化
        if self.zone_history[-1] != zone:
            timestamp = frame_idx / fps
            self.state_changes.append({
                'frame': frame_idx,
                'from_zone': self.zone_history[-1],
                'to_zone': zone,
                'timestamp': f"{timestamp:.2f}s"
            })
            self.zone_history.append(zone)

    def get_travel_distance(self):
        if len(self.positions) < 2:
            return 0
        total = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            total += (dx ** 2 + dy ** 2) ** 0.5
        return total

    def get_avg_speed(self, fps):
        dist = self.get_travel_distance()
        duration = (self.last_frame - self.first_frame) / fps
        return dist / duration if duration > 0 else 0

    def get_avg_confidence(self):
        if not self.confidences:
            return 0
        return sum(c[0] for c in self.confidences) / len(self.confidences)

    def analyze(self):
        """分析轨迹是否有效"""
        history = self.zone_history

        if not history:
            return False, "Empty"
        if history[0] != 'ENTRY':
            return False, f"Ghost (Started in {history[0]})"
        if 'OUT' not in history:
            last_zone = history[-1]
            if last_zone == 'ENTRY':
                return False, "Turned Back"
            elif last_zone == 'WAIT':
                return False, "Stuck in Wait"
            return False, "Incomplete"

        # Last attempt analysis
        try:
            last_entry_idx = len(history) - 1 - history[::-1].index('ENTRY')
            final_attempt = history[last_entry_idx:]

            if 'WAIT' in final_attempt:
                first_wait = final_attempt.index('WAIT')
                if 'OUT' in final_attempt[first_wait:]:
                    if last_entry_idx > 0:
                        return True, "Valid (Retried)"
                    first_out = final_attempt.index('OUT', first_wait)
                    if 'WAIT' in final_attempt[first_out:]:
                        return True, "Valid (Hesitated)"
                    return True, "Valid Sequence"
            return False, "Jumped Over"
        except ValueError:
            return False, "Logic Error"


class ZoneAnalyzer:
    """区域分析器"""

    def __init__(self, width, fps, out_ratio=0.45, wait_ratio=0.25):
        self.width = width
        self.fps = fps
        self.out_ratio = out_ratio
        self.wait_ratio = wait_ratio
        self.split_0 = width * (out_ratio / 2)  # OUT1 | OUT2 (OUT区中间)
        self.split_1 = width * out_ratio  # OUT2 | WAIT
        self.split_2 = width * (out_ratio + wait_ratio)  # WAIT | ENTRY

        self.trajectories = {}
        self.valid_count = 0
        self.id_events = []  # 所有ID事件日志

        # 线穿越计数器（3条线）
        self.line_counters = {
            'line0': 0,  # OUT区内部 (split_0)
            'line1': 0,  # OUT-WAIT边界 (split_1)
            'line2': 0   # WAIT-ENTRY边界 (split_2)
        }
        self.prev_positions = {}  # 记录每个ID的上一帧x坐标

    def get_zone(self, cx):
        if cx < self.split_1:
            return 'OUT'
        elif cx < self.split_2:
            return 'WAIT'
        else:
            return 'ENTRY'

    def update(self, track_id, cx, frame_idx, cy=None, w=None, h=None, conf=None):
        zone = self.get_zone(cx)

        # 检测线穿越（方向：右到左为正方向+1，左到右为反方向-1）
        if track_id in self.prev_positions:
            prev_cx = self.prev_positions[track_id]

            # 检测穿越Line0 (split_0: OUT区内部)
            if prev_cx > self.split_0 >= cx:  # 右到左穿越（正方向）
                self.line_counters['line0'] += 1
            elif prev_cx < self.split_0 <= cx:  # 左到右穿越（反方向）
                self.line_counters['line0'] -= 1

            # 检测穿越Line1 (split_1: OUT-WAIT边界)
            if prev_cx > self.split_1 >= cx:  # 右到左穿越（正方向）
                self.line_counters['line1'] += 1
            elif prev_cx < self.split_1 <= cx:  # 左到右穿越（反方向）
                self.line_counters['line1'] -= 1

            # 检测穿越Line2 (split_2: WAIT-ENTRY边界)
            if prev_cx > self.split_2 >= cx:  # 右到左穿越（正方向）
                self.line_counters['line2'] += 1
            elif prev_cx < self.split_2 <= cx:  # 左到右穿越（反方向）
                self.line_counters['line2'] -= 1

        # 更新位置记录
        self.prev_positions[track_id] = cx

        if track_id not in self.trajectories:
            self.trajectories[track_id] = PigTrajectory(track_id, frame_idx, zone)
            self.id_events.append({
                'frame': frame_idx,
                'timestamp': f"{frame_idx / self.fps:.2f}s",
                'event': 'NEW_ID',
                'track_id': track_id,
                'zone': zone,
                'details': f"ID {track_id} appeared in {zone} (conf={conf:.2f})" if conf else f"ID {track_id} appeared in {zone}"
            })
        else:
            old_zone = self.trajectories[track_id].zone_history[-1]
            self.trajectories[track_id].add_point(zone, frame_idx, self.fps, cx, cy, w, h, conf)

            if old_zone != zone:
                self.id_events.append({
                    'frame': frame_idx,
                    'timestamp': f"{frame_idx / self.fps:.2f}s",
                    'event': 'ZONE_CHANGE',
                    'track_id': track_id,
                    'zone': zone,
                    'details': f"ID {track_id}: {old_zone} -> {zone}"
                })

    def get_current_valid_count(self):
        """计算当前有效计数"""
        count = 0
        for tid, traj in self.trajectories.items():
            is_valid, _ = traj.analyze()
            if is_valid:
                count += 1
        return count

    def finalize(self, output_dir, tracker_name):
        """保存详细报告 - 精确到秒"""
        # 1. ID事件日志 (CSV)
        events_path = output_dir / f"{tracker_name}_id_events.csv"
        with open(events_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'event', 'track_id', 'zone', 'details'])
            writer.writeheader()
            writer.writerows(self.id_events)

        # 2. 详细状态变化日志 (TXT) - 精确到秒
        txt_path = output_dir / f"{tracker_name}_state_changes.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"Tracker: {tracker_name}\n")
            f.write(f"{'=' * 70}\n\n")

            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                status = "✅ VALID" if is_valid else "❌ INVALID"

                f.write(f"ID {tid}: {status} ({reason})\n")
                f.write(f"  首次出现: Frame {traj.first_frame} ({traj.first_frame / self.fps:.2f}s)\n")
                f.write(f"  最后出现: Frame {traj.last_frame} ({traj.last_frame / self.fps:.2f}s)\n")
                f.write(f"  持续时间: {(traj.last_frame - traj.first_frame) / self.fps:.2f}s\n")
                f.write(f"  区域路径: {' -> '.join(traj.zone_history)}\n")

                if traj.state_changes:
                    f.write(f"  状态变化:\n")
                    for change in traj.state_changes:
                        f.write(
                            f"    [{change['timestamp']}] Frame {change['frame']}: {change['from_zone']} -> {change['to_zone']}\n")

                # 新增调试信息
                f.write(f"  移动距离: {traj.get_travel_distance():.1f} px\n")
                f.write(f"  平均速度: {traj.get_avg_speed(self.fps):.1f} px/s\n")
                f.write(f"  平均置信度: {traj.get_avg_confidence():.2f}\n")
                f.write(f"  丢失帧数: {len(traj.lost_frames)}\n")
                f.write(f"  恢复次数: {traj.recovered_count}\n")
                f.write("\n")

                if is_valid:
                    self.valid_count += 1

            f.write(f"{'=' * 70}\n")
            f.write(f"TOTAL VALID: {self.valid_count}\n")
            f.write(f"TOTAL IDs: {len(self.trajectories)}\n")
            f.write(f"{'=' * 70}\n")

        # 保存线穿越法汇总
        line0 = self.line_counters['line0']
        line1 = self.line_counters['line1']
        line2 = self.line_counters['line2']
        total_line = round((line0 + line1 + line2) / 3.0)
        summary_path = output_dir / f"{tracker_name}_summary.csv"
        with open(summary_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['line0', 'line1', 'line2', 'total_line', 'valid_traj', 'total_ids'])
            writer.writerow([line0, line1, line2, total_line, self.valid_count, len(self.trajectories)])

        # 3. 轨迹分析报告 (CSV)
        report_path = output_dir / f"{tracker_name}_trajectory_report.csv"
        with open(report_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TrackID', 'IsValid', 'Reason', 'FirstFrame', 'FirstTime(s)', 'LastFrame', 'LastTime(s)',
                             'Duration(s)', 'ZoneHistory', 'StateChanges'])

            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                duration = (traj.last_frame - traj.first_frame) / self.fps
                zone_str = "->".join(traj.zone_history)
                changes_str = "; ".join(
                    [f"{c['timestamp']}: {c['from_zone']}->{c['to_zone']}" for c in traj.state_changes])

                writer.writerow([
                    tid, is_valid, reason,
                    traj.first_frame, f"{traj.first_frame / self.fps:.2f}",
                    traj.last_frame, f"{traj.last_frame / self.fps:.2f}",
                    f"{duration:.2f}", zone_str, changes_str
                ])

        print(f"  {tracker_name}: Events CSV    -> {events_path}")
        print(f"  {tracker_name}: State TXT     -> {txt_path}")
        print(f"  {tracker_name}: Trajectory CSV -> {report_path}")
        return self.valid_count


def is_blue_object(frame, bbox, blue_threshold=1.3):
    """
    检测物体是否为蓝色
    Args:
        frame: 原始图像帧
        bbox: 检测框 [x1, y1, x2, y2, score]
        blue_threshold: 蓝色判定阈值，蓝色通道要大于红色和绿色的平均值的倍数
    Returns:
        True if 物体主要为蓝色
    """
    x1, y1, x2, y2 = map(int, bbox[:4])

    # 确保边界在图像范围内
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    # 提取检测框区域
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # 计算BGR通道的平均值
    b_mean = np.mean(roi[:, :, 0])  # 蓝色通道
    g_mean = np.mean(roi[:, :, 1])  # 绿色通道
    r_mean = np.mean(roi[:, :, 2])  # 红色通道

    # 判断是否为蓝色：蓝色通道明显大于红色和绿色
    rg_mean = (r_mean + g_mean) / 2.0

    if b_mean > rg_mean * blue_threshold and b_mean > 80:  # 蓝色要足够明显且有一定亮度
        return True

    return False


def run_bytetrack(model, video_path, output_dir, fps, width, height, limit=0, out_ratio=0.45,
                  wait_ratio=0.25, conf_thres=0.5, track_thresh=0.5):
    """运行 ByteTrack 跟踪器"""
    print(f"\n{'=' * 50}")
    print(f"Running: ByteTrack")
    print(f"{'=' * 50}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit > 0:
        total_frames = min(total_frames, limit)

    output_video = output_dir / "ByteTrack_result.mp4"
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)

    # Initialize ByteTrack (使用原项目完整实现)
    byte_args = SimpleNamespace(
        track_thresh=track_thresh,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False
    )
    tracker = BYTETracker(byte_args, frame_rate=int(fps))

    pbar = tqdm(total=total_frames, desc="ByteTrack")
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect with YOLO
        results = model(frame, conf=conf_thres, verbose=False)
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                score = float(boxes.conf[i].cpu().numpy())

                # 原项目格式: [x1, y1, x2, y2, score] (不包含cls)
                bbox = [x1, y1, x2, y2, score]

                # 过滤蓝色物体
                if is_blue_object(frame, bbox):
                    continue  # 跳过蓝色物体

                detections.append(bbox)

        # 2. Track with ByteTrack
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracks = tracker.update(dets, (height, width), (height, width))
        active_tracks = [(int(track.track_id), track.tlbr) for track in tracks if track.is_activated]

        # 3. Analyze
        for tid, bbox in active_tracks:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            analyzer.update(tid, cx, frame_idx, cy, w, h)

        # 4. Draw
        annotated = frame.copy()

        # Draw zones (3 lines)
        cv2.line(annotated, (int(analyzer.split_0), 0), (int(analyzer.split_0), height), (255, 128, 0), 2)  # Line 0
        cv2.line(annotated, (int(analyzer.split_1), 0), (int(analyzer.split_1), height), (0, 255, 255), 2)  # Line 1
        cv2.line(annotated, (int(analyzer.split_2), 0), (int(analyzer.split_2), height), (0, 255, 255), 2)  # Line 2

        # Draw total count (average of 3 lines, round)
        avg_count = (analyzer.line_counters['line0'] + analyzer.line_counters['line1'] +
                     analyzer.line_counters['line2']) / 3.0
        total_count = round(avg_count)  # Round (四舍五入)
        cv2.rectangle(annotated, (width - 200, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"TOTAL: {total_count}", (width - 190, 40), 0, 1.0, (0, 255, 0), 2)

        # Draw tracker name
        cv2.putText(annotated, "ByteTrack", (10, height - 20), 0, 0.6, (255, 255, 255), 2)

        # Draw line counters in bottom-left corner
        line_y_start = height - 120
        cv2.rectangle(annotated, (5, line_y_start - 5), (220, height - 40), (0, 0, 0), -1)
        cv2.putText(annotated, "Line Crossings:", (10, line_y_start + 15), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Line 0: {analyzer.line_counters['line0']}",
                    (10, line_y_start + 35), 0, 0.5, (255, 128, 0), 1)
        cv2.putText(annotated, f"Line 1: {analyzer.line_counters['line1']}",
                    (10, line_y_start + 55), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Line 2: {analyzer.line_counters['line2']}",
                    (10, line_y_start + 75), 0, 0.5, (0, 255, 255), 1)

        # Draw tracks
        for tid, bbox in active_tracks:
            x1, y1, x2, y2 = map(int, bbox)
            np.random.seed(tid)
            color = tuple(map(int, np.random.randint(50, 255, 3)))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 5), 0, 0.4, color, 1)

        out.write(annotated)
        pbar.update(1)
        frame_idx += 1

    cap.release()
    out.release()
    pbar.close()

    # Finalize
    valid_count = analyzer.finalize(output_dir, "ByteTrack")
    print(f"  Video: {output_video}")
    print(f"  Valid Count: {valid_count}")

    return valid_count, len(analyzer.trajectories)


def main():
    parser = argparse.ArgumentParser(description='猪只追踪与计数系统 - ByteTrack')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model_path', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=0, help='限制处理帧数（0=全部）')
    parser.add_argument('--out_ratio', type=float, default=0.45, help='第一区域占比（用于Line 0和1的位置）')
    parser.add_argument('--wait_ratio', type=float, default=0.25, help='第二区域占比（用于Line 2的位置）')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='检测置信度阈值')
    parser.add_argument('--track_thresh', type=float, default=0.5, help='追踪置信度阈值')
    parser.add_argument('--no_timestamp', action='store_true', help='不使用时间戳子目录')

    args = parser.parse_args()

    # Create output directory
    if args.no_timestamp:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_ratio = 1.0 - args.out_ratio - args.wait_ratio
    print(f"Output Directory: {output_dir}")
    print(f"Line Config: Line0={args.out_ratio/2*100:.0f}% | Line1={args.out_ratio*100:.0f}% | Line2={(args.out_ratio+args.wait_ratio)*100:.0f}%")

    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO(args.model_path)

    # Get video info
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Run ByteTrack
    try:
        valid, total_ids = run_bytetrack(
            model, args.video_path, output_dir, fps, width, height,
            args.limit, args.out_ratio, args.wait_ratio, args.conf_thres, args.track_thresh
        )
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Valid Count: {valid}")
        print(f"Total IDs: {total_ids}")
        print(f"Output Directory: {output_dir}")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
