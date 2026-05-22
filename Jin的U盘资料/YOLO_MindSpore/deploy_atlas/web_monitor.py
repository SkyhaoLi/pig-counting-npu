#!/usr/bin/env python3
"""
Pig Counting Web Monitor - Atlas 200I DK A2
Real-time RTSP/video stream + NPU inference + ByteTrack + Web UI + Agent

Usage:
    python3 web_monitor.py --port 8080 --om models/yolov8n_pig_fp16.om
    python3 web_monitor.py --video videos/test.mp4 --om models/yolov8n_pig_fp16.om
    python3 web_monitor.py --rtsp "rtsp://admin:admin123@192.168.1.108:554/..."
"""

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from npu_detector import NPUDetector
from trackers.byte_tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from track_and_count_npu import ZoneAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pig_counting_agent import PigCountingAgent

# ── Globals ──────────────────────────────────────────────────
app_state = {
    'running': False,
    'mode': 'idle',
    'frame_jpeg': None,
    'fps_inference': 0,
    'fps_source': 0,
    'frame_count': 0,
    'line_counters': {'line0': 0, 'line1': 0, 'line2': 0},
    'total_count': 0,
    'valid_traj': 0,
    'total_ids': 0,
    'start_time': None,
    'source': '',
    'resolution': '',
    'history': [],
    'agent_status': 'BOOT',
    'health_score': 100.0,
    'anomaly_count': 0,
    'recovery_count': 0,
    'latest_event': '',
    'agent_events': [],
    'manual_reviews': [],
    'result_root': '',
    'current_result_dir': '',
    'completed_result_dir': '',
    'lock': threading.Lock(),
    'reset_flag': False,
    'stop_flag': False,
    'agent': None,
    'diagnosis': None,
    'inference_history': [],
}

stream_ctl = {
    'cap': None, 'source': '', 'reconnect_flag': False,
    'reconnect_result': None, 'lock': threading.Lock(),
}

review_store = {"path": None}
server_config = {"om_path": "", "output_dir": "", "conf_thres": 0.5,
                 "track_thresh": 0.5, "out_ratio": 0.45, "wait_ratio": 0.25}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024


def append_manual_review(subject, decision, note=""):
    entry = {"ts": datetime.now().isoformat(timespec="seconds"),
             "subject": subject, "decision": decision, "note": note}
    path = review_store.get("path")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with app_state['lock']:
        reviews = app_state["manual_reviews"]
        reviews.insert(0, entry)
        app_state["manual_reviews"] = reviews[:12]
    return entry


def _history_path():
    return Path(server_config['output_dir']) / 'inference_history.json'


def load_inference_history():
    p = _history_path()
    if p.exists():
        try:
            with open(p, encoding='utf-8') as f:
                app_state['inference_history'] = json.load(f)
        except Exception:
            app_state['inference_history'] = []


def save_inference_history():
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(app_state['inference_history'], f, ensure_ascii=False, indent=2)


def append_inference_record(source, total_count, valid_traj, total_ids, result_dir, duration_s):
    record = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'ts': datetime.now().isoformat(timespec='seconds'),
        'source': Path(source).name if source else 'unknown',
        'total_count': total_count,
        'valid_traj': valid_traj,
        'total_ids': total_ids,
        'result_dir': str(result_dir),
        'duration_s': round(duration_s, 1),
        'diagnosis': None,
    }
    with app_state['lock']:
        app_state['inference_history'].insert(0, record)
    save_inference_history()
    return record


# ── Helpers ──────────────────────────────────────────────────
RESULT_FILES = {
    'summary.csv': 'ByteTrack_summary.csv',
    'id_events.csv': 'ByteTrack_id_events.csv',
    'trajectory.csv': 'ByteTrack_trajectory_report.csv',
    'state_changes.txt': 'ByteTrack_state_changes.txt',
}


def count_valid_trajectories(analyzer):
    valid = 0
    for traj in analyzer.trajectories.values():
        is_valid, _ = traj.analyze()
        if is_valid:
            valid += 1
    return valid


def get_total_count(analyzer):
    line0 = analyzer.line_counters['line0']
    line1 = analyzer.line_counters['line1']
    line2 = analyzer.line_counters['line2']
    return round((line0 + line1 + line2) / 3.0)


def export_result_files(analyzer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.valid_count = 0
    analyzer.finalize(output_dir, "ByteTrack")


def snapshot_completed_results(current_dir, completed_dir):
    completed_dir.mkdir(parents=True, exist_ok=True)
    for filename in RESULT_FILES.values():
        src = current_dir / filename
        if src.exists():
            shutil.copy2(src, completed_dir / filename)


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
    b = np.mean(roi[:, :, 0])
    g = np.mean(roi[:, :, 1])
    r = np.mean(roi[:, :, 2])
    rg = (r + g) / 2.0
    return b > rg * blue_threshold and b > 80


# ── Frame grabber ────────────────────────────────────────────
latest_frame = {'frame': None, 'lock': threading.Lock()}
MAX_RECONNECT_ATTEMPTS = 5


def grabber_loop():
    while app_state.get('running', True):
        with stream_ctl['lock']:
            if stream_ctl['reconnect_flag']:
                old_cap = stream_ctl['cap']
                if old_cap:
                    old_cap.release()
                source = stream_ctl['source']
                success = False
                for attempt in range(MAX_RECONNECT_ATTEMPTS):
                    new_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if new_cap.isOpened():
                        stream_ctl['cap'] = new_cap
                        success = True
                        break
                    new_cap.release()
                    time.sleep(1.0 * (attempt + 1))
                stream_ctl['reconnect_flag'] = False
                stream_ctl['reconnect_result'] = success
                if not success:
                    stream_ctl['cap'] = None
                continue
        cap = stream_ctl.get('cap')
        if cap is None:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with latest_frame['lock']:
            latest_frame['frame'] = frame


# ── Inference loop ───────────────────────────────────────────
_detector_instance = None
_detector_lock = threading.Lock()


def get_detector(om_path, conf_thres):
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = NPUDetector(om_path, conf_thres=conf_thres)
        return _detector_instance


def inference_loop(source, om_path, conf_thres, track_thresh, out_ratio, wait_ratio, output_root):
    try:
        _inference_loop_inner(source, om_path, conf_thres, track_thresh, out_ratio, wait_ratio, output_root)
    except Exception as e:
        print(f"[ERROR] inference_loop crashed: {e}")
        import traceback
        traceback.print_exc()
        with app_state['lock']:
            app_state['running'] = False
            app_state['mode'] = 'idle'


def _inference_loop_inner(source, om_path, conf_thres, track_thresh, out_ratio, wait_ratio, output_root):
    detector = get_detector(om_path, conf_thres)
    agent = PigCountingAgent(log_dir=output_root)
    app_state['agent'] = agent

    if source.startswith('rtsp://') or source.startswith('http://'):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_stream = True
    else:
        cap = cv2.VideoCapture(source)
        is_stream = False

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        with app_state['lock']:
            app_state['mode'] = 'idle'
            app_state['running'] = False
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    with app_state['lock']:
        app_state['fps_source'] = fps
        app_state['resolution'] = f"{width}x{height}"
        app_state['source'] = source
        app_state['start_time'] = time.time()
        app_state['running'] = True
        app_state['mode'] = 'running'
        app_state['stop_flag'] = False

    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)
    output_root = Path(output_root)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_result_dir = output_root / "runs" / run_id
    completed_result_dir = output_root / "latest_completed"
    current_result_dir.mkdir(parents=True, exist_ok=True)
    completed_result_dir.mkdir(parents=True, exist_ok=True)

    byte_args = SimpleNamespace(track_thresh=track_thresh, track_buffer=30, match_thresh=0.8, mot20=False)
    tracker = BYTETracker(byte_args, frame_rate=max(1, int(fps)))

    if is_stream:
        with stream_ctl['lock']:
            stream_ctl['cap'] = cap
            stream_ctl['source'] = source
            stream_ctl['reconnect_result'] = None
        agent.note_stream_started(source)
        gt = threading.Thread(target=grabber_loop, daemon=True)
        gt.start()

    frame_idx = 0
    t_last = time.time()
    fps_counter = 0
    last_export = 0.0
    failure_streak = 0
    last_good_frame_time = time.time()

    with app_state['lock']:
        app_state['result_root'] = str(output_root)
        app_state['current_result_dir'] = str(current_result_dir)
        app_state['completed_result_dir'] = str(completed_result_dir)

    while app_state['running']:
        if app_state['stop_flag']:
            break

        if app_state['reset_flag']:
            export_result_files(analyzer, current_result_dir)
            snapshot_completed_results(current_result_dir, completed_result_dir)
            analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)
            tracker = BYTETracker(byte_args, frame_rate=max(1, int(fps)))
            frame_idx = 0
            with app_state['lock']:
                app_state['reset_flag'] = False
                app_state['history'] = []
                app_state['start_time'] = time.time()
            print("[RESET] Counters reset")

        if is_stream:
            actions = agent.consume_actions()
            if actions.get("reconnect"):
                with stream_ctl['lock']:
                    stream_ctl['reconnect_flag'] = True
                    stream_ctl['reconnect_result'] = None
                for _ in range(80):
                    time.sleep(0.1)
                    with stream_ctl['lock']:
                        result = stream_ctl.get('reconnect_result')
                    if result is not None:
                        break
                success = result if result is not None else False
                agent.note_reconnect_result(success)
                if success:
                    failure_streak = 0
                    last_good_frame_time = time.time()
                continue

            with latest_frame['lock']:
                frame = latest_frame['frame']
            if frame is None:
                failure_streak += 1
                wait_seconds = time.time() - last_good_frame_time
                agent.note_waiting_for_frame(wait_seconds, failure_streak)
                time.sleep(0.01)
                continue
            with latest_frame['lock']:
                latest_frame['frame'] = None
            failure_streak = 0
            last_good_frame_time = time.time()
        else:
            ret, frame = cap.read()
            if not ret:
                export_result_files(analyzer, current_result_dir)
                snapshot_completed_results(current_result_dir, completed_result_dir)
                break

        raw_dets = detector.detect(frame)
        detections = [det[:5] for det in raw_dets if not is_blue_object(frame, det[:5])]
        dets = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(dets, (height, width), (height, width))
        active_tracks = [(int(t.track_id), t.tlbr) for t in tracks if t.is_activated]

        for tid, bbox in active_tracks:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            analyzer.update(tid, cx, frame_idx, cy=cy, w=bw, h=bh, conf=None)

        annotated = frame.copy()
        cv2.line(annotated, (int(analyzer.split_0), 0), (int(analyzer.split_0), height), (255, 128, 0), 2)
        cv2.line(annotated, (int(analyzer.split_1), 0), (int(analyzer.split_1), height), (0, 255, 255), 2)
        cv2.line(annotated, (int(analyzer.split_2), 0), (int(analyzer.split_2), height), (0, 255, 255), 2)
        total = get_total_count(analyzer)
        cv2.rectangle(annotated, (width - 200, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"TOTAL: {total}", (width - 190, 40), 0, 1.0, (0, 255, 0), 2)
        for tid, bbox in active_tracks:
            x1, y1, x2, y2 = map(int, bbox)
            np.random.seed(tid)
            color = tuple(map(int, np.random.randint(50, 255, 3)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{tid}", (x1, y1 - 5), 0, 0.4, color, 1)

        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])

        fps_counter += 1
        if time.time() - t_last >= 1.0:
            infer_fps = fps_counter / (time.time() - t_last)
            t_last = time.time()
            fps_counter = 0
        else:
            infer_fps = app_state['fps_inference']

        valid_traj = count_valid_trajectories(analyzer)
        total_ids = len(analyzer.trajectories)
        agent.note_frame(frame_idx, infer_fps, total, valid_traj, total_ids)
        snapshot = agent.snapshot()

        do_export = False
        with app_state['lock']:
            app_state['frame_jpeg'] = jpeg.tobytes()
            app_state['fps_inference'] = round(infer_fps, 1)
            app_state['frame_count'] = frame_idx
            app_state['line_counters'] = dict(analyzer.line_counters)
            app_state['total_count'] = total
            app_state['valid_traj'] = valid_traj
            app_state['total_ids'] = total_ids
            app_state['agent_status'] = snapshot['status']
            app_state['health_score'] = snapshot['health_score']
            app_state['anomaly_count'] = snapshot['anomaly_count']
            app_state['recovery_count'] = snapshot['recovery_count']
            app_state['latest_event'] = snapshot['latest_event']
            app_state['agent_events'] = snapshot['events']
            elapsed = time.time() - app_state['start_time']
            if len(app_state['history']) == 0 or elapsed - app_state['history'][-1]['time'] >= 5:
                app_state['history'].append({'time': round(elapsed, 1), 'total': total})
            if elapsed - last_export >= 5:
                do_export = True
                last_export = elapsed

        if do_export:
            export_result_files(analyzer, current_result_dir)
        frame_idx += 1

    if not is_stream:
        export_result_files(analyzer, current_result_dir)
        snapshot_completed_results(current_result_dir, completed_result_dir)
    cap.release()
    duration = time.time() - app_state['start_time'] if app_state['start_time'] else 0
    with app_state['lock']:
        final_total = app_state['total_count']
        final_valid = app_state['valid_traj']
        final_ids = app_state['total_ids']
        app_state['running'] = False
        app_state['mode'] = 'finished'
    append_inference_record(source, final_total, final_valid, final_ids, str(current_result_dir), duration)


def start_inference(source):
    cfg = server_config
    with app_state['lock']:
        if app_state['mode'] == 'running':
            return False
        app_state['history'] = []
        app_state['diagnosis'] = None
        app_state['frame_jpeg'] = None
    t = threading.Thread(target=inference_loop, args=(
        source, cfg['om_path'], cfg['conf_thres'], cfg['track_thresh'],
        cfg['out_ratio'], cfg['wait_ratio'], cfg['output_dir']), daemon=True)
    t.start()
    return True


def stop_inference():
    with app_state['lock']:
        app_state['stop_flag'] = True
        app_state['running'] = False


def run_diagnosis():
    agent = app_state.get('agent')
    if not agent:
        return None
    with app_state['lock']:
        completed_dir = app_state.get('completed_result_dir')
        source = app_state.get('source', '')
    if not completed_dir:
        return None
    completed_dir = Path(completed_dir)
    video_name = Path(source).name if source else "unknown"
    try:
        diagnosis = agent.analyze(completed_dir, video_name)
        agent.write_reports(completed_dir, diagnosis)
        with app_state['lock']:
            app_state['diagnosis'] = diagnosis
            if app_state['inference_history']:
                app_state['inference_history'][0]['diagnosis'] = diagnosis
        save_inference_history()
        return diagnosis
    except Exception as e:
        return {"error": str(e)}


def run_diagnosis_for_run(run_id):
    with app_state['lock']:
        records = app_state['inference_history']
    record = next((r for r in records if r['id'] == run_id), None)
    if not record:
        return None
    result_dir = Path(record['result_dir'])
    if not result_dir.exists():
        return {"error": "结果目录不存在"}
    agent = app_state.get('agent')
    if not agent:
        agent = PigCountingAgent(log_dir=server_config['output_dir'])
    video_name = record.get('source', 'unknown')
    try:
        diagnosis = agent.analyze(result_dir, video_name)
        agent.write_reports(result_dir, diagnosis)
        with app_state['lock']:
            for r in app_state['inference_history']:
                if r['id'] == run_id:
                    r['diagnosis'] = diagnosis
                    break
        save_inference_history()
        return diagnosis
    except Exception as e:
        return {"error": str(e)}


# ── HTML Page (based on original working frontend) ───────────
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse
import mimetypes

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pig Counter - Live Monitor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; overflow: hidden; }
.header { background: #1e293b; padding: 8px 20px; display: flex;
           align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; }
.header h1 { font-size: 1.1rem; color: #38bdf8; }
.header .status { font-size: 0.8rem; }
.status .live { color: #22c55e; font-weight: bold; }
.status .idle-tag { color: #94a3b8; }
.status .done-tag { color: #fb923c; font-weight: bold; }
.container { display: grid; grid-template-columns: 1fr 240px 300px; gap: 10px;
             padding: 10px; height: calc(100vh - 44px); }
.video-panel { position: relative; background: #000; border-radius: 8px; overflow: hidden; min-height: 0; }
.video-panel img { width: 100%; height: 100%; object-fit: contain; }
.video-panel .placeholder { display: flex; align-items: center; justify-content: center;
                            height: 100%; color: #475569; font-size: 1.1rem; }
.mid-panel { display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
.right-panel { display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
.card { background: #1e293b; border-radius: 8px; padding: 12px; border: 1px solid #334155; }
.card h3 { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
            letter-spacing: 0.05em; margin-bottom: 8px; }
.big-number { font-size: 2.4rem; font-weight: 700; color: #22c55e; text-align: center; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.stat { text-align: center; padding: 6px; background: #0f172a; border-radius: 6px; }
.stat .label { font-size: 0.7rem; color: #64748b; }
.stat .value { font-size: 0.95rem; font-weight: 600; margin-top: 2px; }
.line-stats { display: flex; flex-direction: column; gap: 4px; }
.line-row { display: flex; justify-content: space-between; align-items: center;
            padding: 4px 8px; background: #0f172a; border-radius: 4px; }
.line-row .name { font-size: 0.8rem; }
.line-row .count { font-size: 0.9rem; font-weight: 600; }
.line0 .name { color: #fb923c; }
.line1 .name, .line2 .name { color: #22d3ee; }
.btn { display: inline-block; padding: 6px 12px; border-radius: 6px; border: none;
       cursor: pointer; font-size: 0.78rem; font-weight: 500; text-decoration: none; color: white; }
.btn-primary { background: #3b82f6; }
.btn-primary:hover { background: #2563eb; }
.btn-danger { background: #ef4444; }
.btn-success { background: #22c55e; }
.actions { display: flex; gap: 6px; flex-wrap: wrap; }
canvas { width: 100%; height: 80px; }
input[type=text] { width: 100%; padding: 7px; border-radius: 6px; border: 1px solid #334155;
                   background: #0f172a; color: #e2e8f0; margin-bottom: 8px; font-size: 0.82rem; }
.upload-zone { border: 2px dashed #334155; border-radius: 8px; padding: 16px;
               text-align: center; margin-bottom: 8px; cursor: pointer; font-size: 0.82rem; }
.upload-zone:hover { border-color: #38bdf8; }
.file-list { list-style: none; max-height: 100px; overflow-y: auto; margin-bottom: 8px; }
.file-list li { padding: 5px 8px; background: #0f172a; margin-bottom: 3px; border-radius: 4px;
                cursor: pointer; border: 1px solid transparent; font-size: 0.82rem; }
.file-list li.selected { border-color: #38bdf8; background: #1e3a5f; }
.file-list li:hover { background: #1e293b; }
.progress-bar { height: 3px; background: #334155; border-radius: 2px; margin: 6px 0; display: none; }
.progress-bar .fill { height: 100%; background: #38bdf8; border-radius: 2px; transition: width 0.3s; }
.hist-item { padding: 8px 10px; background: #0f172a; border-radius: 6px; margin-bottom: 5px;
             border: 1px solid #334155; }
.hist-item .hist-header { display: flex; justify-content: space-between; align-items: center; }
.hist-item .hist-name { font-size: 0.82rem; font-weight: 500; }
.hist-item .hist-time { font-size: 0.7rem; color: #64748b; }
.hist-item .hist-stats { font-size: 0.75rem; color: #94a3b8; margin-top: 3px; }
.hist-item .hist-actions { margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }
.btn-sm { padding: 3px 8px; font-size: 0.7rem; border-radius: 4px; }
.tabs { display: flex; gap: 2px; margin-bottom: 8px; }
.tab-btn { padding: 6px 12px; background: #1e293b; border: 1px solid #334155; color: #94a3b8;
           cursor: pointer; border-radius: 6px 6px 0 0; font-size: 0.8rem; }
.tab-btn.active { background: #334155; color: #38bdf8; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }
@media (max-width: 1100px) { .container { grid-template-columns: 1fr 300px; }
  .mid-panel { display: none; } }
</style>
</head>
<body>
<div class="header">
    <h1>Pig Counter - NPU Live Monitor</h1>
    <div class="status">
        <span id="status-dot" class="idle-tag">IDLE</span>
        <span id="info-src"></span>
    </div>
</div>
<div class="container">
    <!-- Left: Video -->
    <div class="video-panel" id="videoPanel">
        <img id="stream" src="" alt="Video Stream" style="display:none">
        <div class="placeholder" id="videoPlaceholder">等待推理开始...</div>
    </div>
    <!-- Middle: Stats -->
    <div class="mid-panel">
        <div class="card">
            <h3>Total Count</h3>
            <div class="big-number" id="total">0</div>
        </div>
        <div class="card">
            <h3>Line Crossings</h3>
            <div class="line-stats">
                <div class="line-row line0"><span class="name">Line 0</span><span class="count" id="line0">0</span></div>
                <div class="line-row line1"><span class="name">Line 1</span><span class="count" id="line1">0</span></div>
                <div class="line-row line2"><span class="name">Line 2</span><span class="count" id="line2">0</span></div>
            </div>
        </div>
        <div class="card">
            <h3>Statistics</h3>
            <div class="stat-grid">
                <div class="stat"><div class="label">FPS</div><div class="value" id="fps">0</div></div>
                <div class="stat"><div class="label">IDs</div><div class="value" id="ids">0</div></div>
                <div class="stat"><div class="label">Valid</div><div class="value" id="valid">0</div></div>
                <div class="stat"><div class="label">Frames</div><div class="value" id="frames">0</div></div>
            </div>
        </div>
        <div class="card">
            <h3>Agent</h3>
            <div class="stat-grid">
                <div class="stat"><div class="label">Status</div><div class="value" id="agent-status">BOOT</div></div>
                <div class="stat"><div class="label">Health</div><div class="value" id="agent-health">100</div></div>
            </div>
        </div>
        <div class="card">
            <h3>Trend</h3>
            <canvas id="chart"></canvas>
        </div>
        <div class="card">
            <div class="actions">
                <button class="btn btn-danger" onclick="stopInference()">停止</button>
                <button class="btn btn-danger" onclick="resetCount()">重置</button>
                <a class="btn btn-primary" href="/download/csv" download>CSV</a>
            </div>
        </div>
    </div>
    <!-- Right: Controls + History -->
    <div class="right-panel">
        <div class="card">
            <h3>数据源</h3>
            <div class="tabs">
                <button class="tab-btn" onclick="switchTab('camera')">摄像头</button>
                <button class="tab-btn active" onclick="switchTab('upload')">上传视频</button>
            </div>
            <div class="tab-pane" id="tabCamera">
                <input type="text" id="rtspUrl" placeholder="rtsp://...">
                <button class="btn btn-primary" onclick="startCamera()">开始监控</button>
            </div>
            <div class="tab-pane active" id="tabUpload">
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <p>点击上传视频</p>
                    <p style="color:#64748b;font-size:0.72rem;margin-top:4px">mp4/avi/mkv, max 500MB</p>
                </div>
                <input type="file" id="fileInput" accept=".mp4,.avi,.mkv,.mov" style="display:none" onchange="uploadFile(this)">
                <div class="progress-bar" id="uploadProgress"><div class="fill" id="uploadFill"></div></div>
                <ul class="file-list" id="fileList"></ul>
                <button class="btn btn-primary" id="btnStartVideo" disabled onclick="startVideo()">开始推理</button>
            </div>
        </div>
        <div class="card">
            <h3>历史推理记录</h3>
            <div id="historyList" style="max-height:calc(100vh - 380px);overflow-y:auto">
                <p style="color:#64748b;font-size:0.82rem">暂无历史记录</p>
            </div>
        </div>
    </div>
</div>
<script>
let mode='idle', selectedFile=null;
function switchTab(t){
  document.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active',i===(t==='camera'?0:1)));
  document.getElementById('tabCamera').classList.toggle('active',t==='camera');
  document.getElementById('tabUpload').classList.toggle('active',t==='upload');
}
function startCamera(){
  const url=document.getElementById('rtspUrl').value.trim();
  if(!url){alert('请输入RTSP地址');return;}
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:url})})
  .then(r=>r.json()).then(d=>{if(!d.ok)alert(d.error||'启动失败');});
}
function startVideo(){
  if(!selectedFile){alert('请先选择视频');return;}
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:'upload:'+selectedFile})})
  .then(r=>r.json()).then(d=>{if(!d.ok)alert(d.error||'启动失败');});
}
function stopInference(){ fetch('/api/stop',{method:'POST'}); }
function resetCount(){ fetch('/api/reset',{method:'POST'}); }
function uploadFile(input){
  const file=input.files[0]; if(!file)return;
  const fd=new FormData(); fd.append('file',file);
  const bar=document.getElementById('uploadProgress'),fill=document.getElementById('uploadFill');
  bar.style.display='block'; fill.style.width='0%';
  const xhr=new XMLHttpRequest();
  xhr.upload.onprogress=e=>{if(e.lengthComputable)fill.style.width=(e.loaded/e.total*100)+'%';};
  xhr.onload=()=>{bar.style.display='none';if(xhr.status===200)loadUploads();else alert('上传失败');};
  xhr.onerror=()=>{bar.style.display='none';alert('上传出错');};
  xhr.open('POST','/api/upload'); xhr.send(fd);
}
function loadUploads(){
  fetch('/api/uploads').then(r=>r.json()).then(files=>{
    const ul=document.getElementById('fileList'); ul.innerHTML='';
    files.forEach(f=>{
      const li=document.createElement('li'); li.textContent=f;
      li.onclick=()=>{selectedFile=f;document.querySelectorAll('.file-list li').forEach(x=>x.classList.remove('selected'));li.classList.add('selected');document.getElementById('btnStartVideo').disabled=false;};
      ul.appendChild(li);
    });
  });
}
function deleteRun(runId){
  if(!confirm('确定删除此记录？'))return;
  fetch('/api/history/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:runId})})
  .then(r=>r.json()).then(d=>{if(d.ok)loadHistory();else alert(d.error||'删除失败');});
}
function runDiagnosis(runId){
  const body = runId ? JSON.stringify({run_id:runId}) : JSON.stringify({});
  fetch('/api/diagnose',{method:'POST',headers:{'Content-Type':'application/json'},body:body})
  .then(r=>r.json()).then(d=>{
    if(d.error){alert(d.error);return;}
    loadHistory();
  });
}
function loadHistory(){
  fetch('/api/history').then(r=>r.json()).then(records=>{
    const el=document.getElementById('historyList');
    if(!records||!records.length){el.innerHTML='<p style="color:#64748b;font-size:0.82rem">暂无历史记录</p>';return;}
    el.innerHTML=records.map(r=>{
      let acts='<a class="btn btn-primary btn-sm" href="/download/'+r.id+'/summary.csv" download>汇总</a>';
      acts+='<a class="btn btn-primary btn-sm" href="/download/'+r.id+'/trajectory.csv" download>轨迹</a>';
      if(r.diagnosis){
        acts+='<a class="btn btn-primary btn-sm" href="/download/'+r.id+'/diagnosis.json" download>诊断</a>';
      } else {
        acts+='<button class="btn btn-success btn-sm" onclick="runDiagnosis(&quot;'+r.id+'&quot;)">诊断</button>';
      }
      acts+='<button class="btn btn-danger btn-sm" onclick="deleteRun(&quot;'+r.id+'&quot;)">删除</button>';
      const dur=r.duration_s>=60?Math.round(r.duration_s/60)+'min':Math.round(r.duration_s)+'s';
      return '<div class="hist-item"><div class="hist-header"><span class="hist-name">'+r.source+'</span><span class="hist-time">'+dur+'</span></div>'
        +'<div class="hist-stats">计数:'+r.total_count+' 有效:'+r.valid_traj+' ID:'+r.total_ids+'</div>'
        +'<div class="hist-actions">'+acts+'</div></div>';
    }).join('');
  });
}
function update(){
  fetch('/api/stats').then(r=>r.json()).then(d=>{
    const dot=document.getElementById('status-dot');
    const img=document.getElementById('stream');
    const ph=document.getElementById('videoPlaceholder');
    if(d.mode!==mode){
      mode=d.mode;
      if(d.mode==='running'){
        dot.className='live'; dot.textContent='LIVE';
        img.src='/stream?t='+Date.now(); img.style.display=''; ph.style.display='none';
      } else if(d.mode==='finished'){
        dot.className='done-tag'; dot.textContent='FINISHED';
        img.style.display='none'; ph.style.display='flex'; ph.textContent='推理完成';
        loadHistory();
      } else {
        dot.className='idle-tag'; dot.textContent='IDLE';
        img.style.display='none'; ph.style.display='flex'; ph.textContent='等待推理开始...';
      }
    }
    if(d.mode==='running'){
      document.getElementById('total').textContent=d.total_count;
      document.getElementById('line0').textContent=d.line0;
      document.getElementById('line1').textContent=d.line1;
      document.getElementById('line2').textContent=d.line2;
      document.getElementById('fps').textContent=d.fps_inference;
      document.getElementById('ids').textContent=d.total_ids;
      document.getElementById('valid').textContent=d.valid_traj;
      document.getElementById('frames').textContent=d.frame_count;
      document.getElementById('agent-status').textContent=d.agent_status;
      document.getElementById('agent-health').textContent=d.health_score;
      document.getElementById('info-src').textContent=d.resolution+' | '+d.fps_source+'fps';
      drawChart(d.history);
    }
  }).catch(()=>{});
}
function drawChart(history){
  const c=document.getElementById('chart'); if(!c)return;
  const ctx=c.getContext('2d');
  c.width=c.offsetWidth*2; c.height=c.offsetHeight*2; ctx.scale(2,2);
  const W=c.offsetWidth, H=c.offsetHeight; ctx.clearRect(0,0,W,H);
  if(!history||history.length<2)return;
  const maxT=Math.max(...history.map(h=>h.total),1);
  ctx.strokeStyle='#22c55e'; ctx.lineWidth=1.5; ctx.beginPath();
  history.forEach((h,i)=>{const x=(i/(history.length-1))*W;const y=H-(h.total/maxT)*(H-10)-5;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
}
setInterval(update,800);
document.addEventListener('DOMContentLoaded',()=>{loadUploads();loadHistory();update();});
</script>
</body>
</html>'''


# ── HTTP Handler ─────────────────────────────────────────────
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/' or path == '/index.html':
            body = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                idle_count = 0
                while True:
                    with app_state['lock']:
                        jpeg = app_state.get('frame_jpeg')
                        running = app_state.get('running', False)
                    if not running:
                        break
                    if jpeg:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpeg)).encode() + b'\r\n\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        idle_count = 0
                    else:
                        idle_count += 1
                        if idle_count > 50:
                            break
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        elif path == '/api/stats':
            with app_state['lock']:
                data = {
                    'mode': app_state['mode'],
                    'total_count': app_state['total_count'],
                    'line0': app_state['line_counters'].get('line0', 0),
                    'line1': app_state['line_counters'].get('line1', 0),
                    'line2': app_state['line_counters'].get('line2', 0),
                    'fps_inference': app_state['fps_inference'],
                    'fps_source': app_state['fps_source'],
                    'frame_count': app_state['frame_count'],
                    'valid_traj': app_state['valid_traj'],
                    'total_ids': app_state['total_ids'],
                    'resolution': app_state['resolution'],
                    'history': app_state['history'][-60:],
                    'agent_status': app_state.get('agent_status', 'BOOT'),
                    'health_score': app_state.get('health_score', 100),
                    'anomaly_count': app_state.get('anomaly_count', 0),
                    'recovery_count': app_state.get('recovery_count', 0),
                    'agent_events': app_state.get('agent_events', []),
                }
            self._json(data)

        elif path == '/api/uploads':
            upload_dir = Path(server_config['output_dir']) / 'uploads'
            files = []
            if upload_dir.exists():
                files = sorted([f.name for f in upload_dir.iterdir() if f.is_file()])
            self._json(files)

        elif path == '/api/history':
            with app_state['lock']:
                hist = list(app_state['inference_history'])
            self._json(hist)

        elif path == '/api/diagnosis':
            with app_state['lock']:
                diag = app_state.get('diagnosis')
            self._json(diag if diag else {'error': '暂无诊断报告'}, 200 if diag else 404)

        elif path.startswith('/download/'):
            parts = path[len('/download/'):].split('/')
            if len(parts) == 2:
                run_id, key = parts
                fpath = self._resolve_run_download(run_id, key)
            else:
                key = parts[0]
                fpath = self._resolve_download(key)
            if fpath and fpath.exists():
                self.send_response(200)
                ct = mimetypes.guess_type(str(fpath))[0] or 'application/octet-stream'
                self.send_header('Content-Type', ct)
                self.send_header('Content-Disposition', f'attachment; filename="{fpath.name}"')
                data = fpath.read_bytes()
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def _resolve_download(self, key):
        filename = RESULT_FILES.get(key)
        if key == 'csv':
            filename = RESULT_FILES['summary.csv']
        elif key == 'diagnosis.json':
            d = app_state.get('completed_result_dir')
            if d:
                p = Path(d) / "ByteTrack_diagnosis.json"
                return p if p.exists() else None
            return None
        elif key == 'diagnosis.md':
            d = app_state.get('completed_result_dir')
            if d:
                p = Path(d) / "ByteTrack_diagnosis.md"
                return p if p.exists() else None
            return None
        if not filename:
            return None
        for base_key in ('completed_result_dir', 'current_result_dir'):
            d = app_state.get(base_key)
            if d:
                candidate = Path(d) / filename
                if candidate.exists():
                    return candidate
        return None

    def _resolve_run_download(self, run_id, key):
        with app_state['lock']:
            records = app_state['inference_history']
        record = next((r for r in records if r['id'] == run_id), None)
        if not record:
            return None
        result_dir = Path(record['result_dir'])
        if key == 'summary.csv':
            filename = RESULT_FILES['summary.csv']
        elif key == 'trajectory.csv':
            filename = RESULT_FILES['trajectory.csv']
        elif key == 'diagnosis.json':
            return result_dir / "ByteTrack_diagnosis.json"
        elif key == 'diagnosis.md':
            return result_dir / "ByteTrack_diagnosis.md"
        else:
            filename = RESULT_FILES.get(key)
        if not filename:
            return None
        return result_dir / filename

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/api/start':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            source = body.get('source', '')
            if source.startswith('upload:'):
                filename = source[7:]
                upload_dir = Path(server_config['output_dir']) / 'uploads'
                filepath = upload_dir / filename
                if not filepath.exists():
                    self._json({'ok': False, 'error': '文件不存在'}, 400)
                    return
                source = str(filepath)
            if not source:
                self._json({'ok': False, 'error': '未指定数据源'}, 400)
                return
            ok = start_inference(source)
            self._json({'ok': True} if ok else {'ok': False, 'error': '已有推理在运行'}, 200 if ok else 409)

        elif path == '/api/stop':
            stop_inference()
            self._json({'ok': True})

        elif path == '/api/reset':
            with app_state['lock']:
                app_state['reset_flag'] = True
            self._json({'ok': True})

        elif path == '/api/upload':
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self._json({'error': '需要 multipart/form-data'}, 400)
                return
            length = int(self.headers.get('Content-Length', 0))
            if length > MAX_UPLOAD_SIZE:
                self._json({'error': '文件过大'}, 413)
                return
            boundary = content_type.split('boundary=')[1].encode()
            raw = self.rfile.read(length)
            filename, filedata = self._parse_multipart(raw, boundary)
            if not filename:
                self._json({'error': '未找到文件'}, 400)
                return
            upload_dir = Path(server_config['output_dir']) / 'uploads'
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / Path(filename).name
            dest.write_bytes(filedata)
            self._json({'ok': True, 'filename': dest.name})

        elif path == '/api/diagnose':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            run_id = body.get('run_id')
            if run_id:
                result = run_diagnosis_for_run(run_id)
            else:
                result = run_diagnosis()
            if result:
                self._json(result)
            else:
                self._json({'error': '无法生成诊断报告，请先完成一次推理'}, 400)

        elif path == '/api/history/delete':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            run_id = body.get('run_id', '')
            if not run_id:
                self._json({'ok': False, 'error': '缺少run_id'}, 400)
                return
            with app_state['lock']:
                before = len(app_state['inference_history'])
                app_state['inference_history'] = [r for r in app_state['inference_history'] if r['id'] != run_id]
                removed = len(app_state['inference_history']) < before
            if removed:
                save_inference_history()
                self._json({'ok': True})
            else:
                self._json({'ok': False, 'error': '记录不存在'}, 404)

        elif path == '/api/manual_review':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            subject = body.get('subject', '').strip()
            decision = body.get('decision', '').strip()
            note = body.get('note', '').strip()
            if not subject or not decision:
                self._json({'ok': False, 'error': 'subject and decision required'}, 400)
                return
            entry = append_manual_review(subject, decision, note)
            self._json({'ok': True, 'entry': entry})
        else:
            self.send_response(404)
            self.end_headers()

    def _parse_multipart(self, raw, boundary):
        parts = raw.split(b'--' + boundary)
        for part in parts:
            if b'filename="' not in part:
                continue
            header_end = part.find(b'\r\n\r\n')
            if header_end < 0:
                continue
            header = part[:header_end].decode('utf-8', errors='replace')
            fn_start = header.find('filename="') + 10
            fn_end = header.find('"', fn_start)
            filename = header[fn_start:fn_end]
            data = part[header_end + 4:]
            if data.endswith(b'\r\n'):
                data = data[:-2]
            return filename, data
        return None, None


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pig Counting Web Monitor")
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--om', type=str, default='models/yolov8n_pig_fp16.om')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--rtsp', type=str, default='')
    parser.add_argument('--output', type=str, default='output_web')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--track-thresh', type=float, default=0.5)
    parser.add_argument('--out-ratio', type=float, default=0.45)
    parser.add_argument('--wait-ratio', type=float, default=0.25)
    args = parser.parse_args()

    server_config['om_path'] = args.om
    server_config['output_dir'] = args.output
    server_config['conf_thres'] = args.conf
    server_config['track_thresh'] = args.track_thresh
    server_config['out_ratio'] = args.out_ratio
    server_config['wait_ratio'] = args.wait_ratio

    Path(args.output).mkdir(parents=True, exist_ok=True)
    review_store["path"] = str(Path(args.output) / "manual_reviews.jsonl")
    load_inference_history()

    source = args.rtsp or args.video
    if source:
        start_inference(source)

    server = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    print(f"[WEB] http://0.0.0.0:{args.port}")
    print(f"[MODE] {'Auto-started: ' + source if source else 'Idle — use web UI to start'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_inference()
        server.shutdown()
        print("\n[EXIT] Server stopped")


if __name__ == '__main__':
    main()
