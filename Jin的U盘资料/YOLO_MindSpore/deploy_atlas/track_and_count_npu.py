#!/usr/bin/env python3
"""
猪只追踪与计数系统 - NPU (Atlas 200I DK A2) + ByteTrack
使用 ACL NPU 检测 + ByteTrack 追踪

输出:
1. 带标注的结果视频（包含3条分隔线、线穿越次数、TOTAL计数）
2. ID事件日志 CSV
3. 状态变化记录 TXT
4. 轨迹分析报告 CSV

计数逻辑:
- 使用3条分隔线统计穿越次数
- TOTAL = round((Line0 + Line1 + Line2) / 3)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
from collections import defaultdict

from trackers.byte_tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from npu_detector import NPUDetector


class PigTrajectory:
    """跟踪单只猪的轨迹和状态变化"""

    def __init__(self, track_id, first_frame, first_zone):
        self.track_id = track_id
        self.zone_history = [first_zone]
        self.state_changes = []
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.status = "ACTIVE"
        self.positions = []
        self.box_sizes = []
        self.confidences = []
        self.lost_frames = []
        self.recovered_count = 0
        self.last_cx = None

    def add_point(self, zone, frame_idx, fps, cx=None, cy=None, w=None, h=None, conf=None):
        if frame_idx - self.last_frame > 1:
            for lost_f in range(self.last_frame + 1, frame_idx):
                self.lost_frames.append(lost_f)
            self.recovered_count += 1
        self.last_frame = frame_idx
        if cx is not None:
            self.positions.append((cx, cy, frame_idx))
            self.last_cx = cx
        if w is not None:
            self.box_sizes.append((w, h, frame_idx))
        if conf is not None:
            self.confidences.append((conf, frame_idx))
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
        self.split_0 = width * (out_ratio / 2)
        self.split_1 = width * out_ratio
        self.split_2 = width * (out_ratio + wait_ratio)
        self.trajectories = {}
        self.valid_count = 0
        self.id_events = []
        self.line_counters = {'line0': 0, 'line1': 0, 'line2': 0}
        self.prev_positions = {}

    def get_zone(self, cx):
        if cx < self.split_1:
            return 'OUT'
        elif cx < self.split_2:
            return 'WAIT'
        else:
            return 'ENTRY'

    def update(self, track_id, cx, frame_idx, cy=None, w=None, h=None, conf=None):
        zone = self.get_zone(cx)
        if track_id in self.prev_positions:
            prev_cx = self.prev_positions[track_id]
            if prev_cx > self.split_0 >= cx:
                self.line_counters['line0'] += 1
            elif prev_cx < self.split_0 <= cx:
                self.line_counters['line0'] -= 1
            if prev_cx > self.split_1 >= cx:
                self.line_counters['line1'] += 1
            elif prev_cx < self.split_1 <= cx:
                self.line_counters['line1'] -= 1
            if prev_cx > self.split_2 >= cx:
                self.line_counters['line2'] += 1
            elif prev_cx < self.split_2 <= cx:
                self.line_counters['line2'] -= 1
        self.prev_positions[track_id] = cx

        if track_id not in self.trajectories:
            self.trajectories[track_id] = PigTrajectory(track_id, frame_idx, zone)
            self.id_events.append({
                'frame': frame_idx, 'timestamp': f"{frame_idx / self.fps:.2f}s",
                'event': 'NEW_ID', 'track_id': track_id, 'zone': zone,
                'details': f"ID {track_id} appeared in {zone} (conf={conf:.2f})" if conf else f"ID {track_id} appeared in {zone}"
            })
        else:
            old_zone = self.trajectories[track_id].zone_history[-1]
            self.trajectories[track_id].add_point(zone, frame_idx, self.fps, cx, cy, w, h, conf)
            if old_zone != zone:
                self.id_events.append({
                    'frame': frame_idx, 'timestamp': f"{frame_idx / self.fps:.2f}s",
                    'event': 'ZONE_CHANGE', 'track_id': track_id, 'zone': zone,
                    'details': f"ID {track_id}: {old_zone} -> {zone}"
                })

    def finalize(self, output_dir, tracker_name):
        events_path = output_dir / f"{tracker_name}_id_events.csv"
        with open(events_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'event', 'track_id', 'zone', 'details'])
            writer.writeheader()
            writer.writerows(self.id_events)

        txt_path = output_dir / f"{tracker_name}_state_changes.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"Tracker: {tracker_name}\n")
            f.write(f"{'=' * 70}\n\n")
            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                status = "VALID" if is_valid else "INVALID"
                f.write(f"ID {tid}: {status} ({reason})\n")
                f.write(f"  First: Frame {traj.first_frame} ({traj.first_frame / self.fps:.2f}s)\n")
                f.write(f"  Last: Frame {traj.last_frame} ({traj.last_frame / self.fps:.2f}s)\n")
                f.write(f"  Duration: {(traj.last_frame - traj.first_frame) / self.fps:.2f}s\n")
                f.write(f"  Zones: {' -> '.join(traj.zone_history)}\n")
                if traj.state_changes:
                    f.write(f"  Changes:\n")
                    for change in traj.state_changes:
                        f.write(f"    [{change['timestamp']}] {change['from_zone']} -> {change['to_zone']}\n")
                f.write(f"  Distance: {traj.get_travel_distance():.1f} px\n")
                f.write(f"  Avg Speed: {traj.get_avg_speed(self.fps):.1f} px/s\n")
                f.write(f"  Avg Conf: {traj.get_avg_confidence():.2f}\n")
                f.write(f"  Lost Frames: {len(traj.lost_frames)}\n")
                f.write(f"  Recoveries: {traj.recovered_count}\n\n")
                if is_valid:
                    self.valid_count += 1
            f.write(f"{'=' * 70}\n")
            f.write(f"TOTAL VALID: {self.valid_count}\n")
            f.write(f"TOTAL IDs: {len(self.trajectories)}\n")
            f.write(f"{'=' * 70}\n")

        line0 = self.line_counters['line0']
        line1 = self.line_counters['line1']
        line2 = self.line_counters['line2']
        total_line = round((line0 + line1 + line2) / 3.0)
        summary_path = output_dir / f"{tracker_name}_summary.csv"
        with open(summary_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['line0', 'line1', 'line2', 'total_line', 'valid_traj', 'total_ids'])
            writer.writerow([line0, line1, line2, total_line, self.valid_count, len(self.trajectories)])

        report_path = output_dir / f"{tracker_name}_trajectory_report.csv"
        with open(report_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TrackID', 'IsValid', 'Reason', 'FirstFrame', 'FirstTime(s)',
                             'LastFrame', 'LastTime(s)', 'Duration(s)', 'ZoneHistory', 'StateChanges'])
            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                duration = (traj.last_frame - traj.first_frame) / self.fps
                zone_str = "->".join(traj.zone_history)
                changes_str = "; ".join(
                    [f"{c['timestamp']}: {c['from_zone']}->{c['to_zone']}" for c in traj.state_changes])
                writer.writerow([tid, is_valid, reason, traj.first_frame,
                                 f"{traj.first_frame / self.fps:.2f}", traj.last_frame,
                                 f"{traj.last_frame / self.fps:.2f}", f"{duration:.2f}",
                                 zone_str, changes_str])

        print(f"  {tracker_name}: Events CSV    -> {events_path}")
        print(f"  {tracker_name}: State TXT     -> {txt_path}")
        print(f"  {tracker_name}: Trajectory CSV -> {report_path}")
        return self.valid_count


def is_blue_object(frame, bbox, blue_threshold=1.3):
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
    y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])
    rg_mean = (r_mean + g_mean) / 2.0
    return b_mean > rg_mean * blue_threshold and b_mean > 80


def run_bytetrack(detector, video_path, output_dir, fps, width, height, limit=0,
                  out_ratio=0.45, wait_ratio=0.25, track_thresh=0.5):
    print(f"\n{'=' * 50}")
    print(f"Running: ByteTrack (NPU)")
    print(f"{'=' * 50}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit > 0:
        total_frames = min(total_frames, limit)

    output_video = output_dir / "ByteTrack_result.mp4"
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)

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

        # NPU detection: returns [x1, y1, x2, y2, conf, cls]
        raw_dets = detector.detect(frame)

        # Filter blue objects and convert to [x1, y1, x2, y2, score]
        detections = []
        for det in raw_dets:
            bbox = det[:5]  # x1, y1, x2, y2, conf
            if is_blue_object(frame, bbox):
                continue
            detections.append(bbox[:5])

        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracks = tracker.update(dets, (height, width), (height, width))
        active_tracks = [(int(track.track_id), track.tlbr) for track in tracks if track.is_activated]

        for tid, bbox in active_tracks:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            analyzer.update(tid, cx, frame_idx, cy, w, h)

        # Draw
        annotated = frame.copy()
        cv2.line(annotated, (int(analyzer.split_0), 0), (int(analyzer.split_0), height), (255, 128, 0), 2)
        cv2.line(annotated, (int(analyzer.split_1), 0), (int(analyzer.split_1), height), (0, 255, 255), 2)
        cv2.line(annotated, (int(analyzer.split_2), 0), (int(analyzer.split_2), height), (0, 255, 255), 2)

        avg_count = (analyzer.line_counters['line0'] + analyzer.line_counters['line1'] +
                     analyzer.line_counters['line2']) / 3.0
        total_count = round(avg_count)
        cv2.rectangle(annotated, (width - 200, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"TOTAL: {total_count}", (width - 190, 40), 0, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, "ByteTrack NPU", (10, height - 20), 0, 0.6, (255, 255, 255), 2)

        line_y_start = height - 120
        cv2.rectangle(annotated, (5, line_y_start - 5), (220, height - 40), (0, 0, 0), -1)
        cv2.putText(annotated, "Line Crossings:", (10, line_y_start + 15), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Line 0: {analyzer.line_counters['line0']}",
                    (10, line_y_start + 35), 0, 0.5, (255, 128, 0), 1)
        cv2.putText(annotated, f"Line 1: {analyzer.line_counters['line1']}",
                    (10, line_y_start + 55), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Line 2: {analyzer.line_counters['line2']}",
                    (10, line_y_start + 75), 0, 0.5, (0, 255, 255), 1)

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

    valid_count = analyzer.finalize(output_dir, "ByteTrack")
    print(f"  Video: {output_video}")
    print(f"  Valid Count: {valid_count}")
    return valid_count, len(analyzer.trajectories)


def main():
    parser = argparse.ArgumentParser(description='NPU Pig Tracking & Counting')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--om', type=str, default='models/yolov8n_pig_fp16.om', help='OM model path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--limit', type=int, default=0, help='Limit frames (0=all)')
    parser.add_argument('--out_ratio', type=float, default=0.45)
    parser.add_argument('--wait_ratio', type=float, default=0.25)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--no_timestamp', action='store_true')
    args = parser.parse_args()

    if args.no_timestamp:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output Directory: {output_dir}")
    print(f"Line Config: Line0={args.out_ratio/2*100:.0f}% | Line1={args.out_ratio*100:.0f}% | Line2={(args.out_ratio+args.wait_ratio)*100:.0f}%")

    print("\nLoading NPU model...")
    detector = NPUDetector(args.om, conf_thres=args.conf_thres)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    try:
        valid, total_ids = run_bytetrack(
            detector, args.video, output_dir, fps, width, height,
            args.limit, args.out_ratio, args.wait_ratio, args.track_thresh
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
