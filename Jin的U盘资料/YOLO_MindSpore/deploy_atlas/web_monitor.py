#!/usr/bin/env python3
"""
Pig Counting Web Monitor - Atlas 200I DK A2
Real-time RTSP/video stream + NPU inference + ByteTrack + Web UI + Agent

Usage:
    python3 web_monitor.py --port 8080 --om models/yolov8n_pig_fp16.om
    python3 web_monitor.py --video datasets/si/11.mp4 --om models/yolov8n_pig_fp16.om
    python3 web_monitor.py --rtsp "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
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
    'mode': 'idle',  # idle / running / finished
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
    'result_root': '',
    'current_result_dir': '',
    'completed_result_dir': '',
    'lock': threading.Lock(),
    'reset_flag': False,
    'stop_flag': False,
    'agent': None,
    'manual_reviews': [],
    'diagnosis': None,
}
# ── Stream control ──
stream_ctl = {
    'cap': None, 'source': '', 'reconnect_flag': False,
    'reconnect_result': None, 'lock': threading.Lock(),
}

# ── Manual review store ──
review_store = {"path": None}

# ── Config (set in main) ──
server_config = {"om_path": "", "output_dir": "", "conf_thres": 0.5,
                 "track_thresh": 0.5, "out_ratio": 0.45, "wait_ratio": 0.25}

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


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
        reviews.append(entry)
        if len(reviews) > 50:
            app_state["manual_reviews"] = reviews[-50:]
    return entry


# ── Zone helpers ─────────────────────────────────────────────
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


def resolve_download_path(request_path):
    key = request_path.rsplit('/', 1)[-1]
    filename = RESULT_FILES.get(key)
    if request_path == '/download/csv':
        filename = RESULT_FILES['summary.csv']
    if key == 'diagnosis.json':
        with app_state['lock']:
            d = app_state.get('completed_result_dir')
        if d:
            p = Path(d) / "ByteTrack_diagnosis.json"
            return p if p.exists() else None
        return None
    if key == 'diagnosis.md':
        with app_state['lock']:
            d = app_state.get('completed_result_dir')
        if d:
            p = Path(d) / "ByteTrack_diagnosis.md"
            return p if p.exists() else None
        return None
    if not filename:
        return None
    with app_state['lock']:
        completed_dir = Path(app_state['completed_result_dir']) if app_state['completed_result_dir'] else None
        current_dir = Path(app_state['current_result_dir']) if app_state['current_result_dir'] else None
    for base in (completed_dir, current_dir):
        if base:
            candidate = base / filename
            if candidate.exists():
                return candidate
    return None


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


# ── Frame grabber thread ─────────────────────────────────────
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


# ── Inference thread ─────────────────────────────────────────
def inference_loop(source, om_path, conf_thres, track_thresh, out_ratio, wait_ratio, output_root):
    detector = NPUDetector(om_path, conf_thres=conf_thres)
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
    current_result_dir = output_root / "current"
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
                # Video finished — don't loop, go to finished state
                export_result_files(analyzer, current_result_dir)
                snapshot_completed_results(current_result_dir, completed_result_dir)
                break

        # Detect
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

        # Draw
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

        do_export = False
        with app_state['lock']:
            app_state['frame_jpeg'] = jpeg.tobytes()
            app_state['fps_inference'] = round(infer_fps, 1)
            app_state['frame_count'] = frame_idx
            app_state['line_counters'] = dict(analyzer.line_counters)
            app_state['total_count'] = total
            app_state['valid_traj'] = count_valid_trajectories(analyzer)
            app_state['total_ids'] = len(analyzer.trajectories)
            elapsed = time.time() - app_state['start_time']
            if len(app_state['history']) == 0 or elapsed - app_state['history'][-1]['time'] >= 5:
                app_state['history'].append({'time': round(elapsed, 1), 'total': total})
            if elapsed - last_export >= 5:
                do_export = True
                last_export = elapsed

        agent.note_frame(frame_idx, infer_fps, total, app_state['valid_traj'], app_state['total_ids'])
        if do_export:
            export_result_files(analyzer, current_result_dir)
        frame_idx += 1

    # Cleanup
    if not is_stream:
        export_result_files(analyzer, current_result_dir)
        snapshot_completed_results(current_result_dir, completed_result_dir)
    cap.release()
    with app_state['lock']:
        app_state['running'] = False
        app_state['mode'] = 'finished'


def start_inference(source):
    """Start inference in a background thread. Called from HTTP handler."""
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
    """Run diagnosis on the latest completed results."""
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
        return diagnosis
    except Exception as e:
        return {"error": str(e)}


# ── HTML Page ───────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>猪只计数监控系统</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a2e;color:#eee;min-height:100vh}
.header{background:#16213e;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #0f3460}
.header h1{font-size:1.3rem;color:#4fc3f7}
.header .status-badge{padding:4px 12px;border-radius:12px;font-size:0.8rem;font-weight:600}
.badge-idle{background:#555;color:#ccc}
.badge-running{background:#2e7d32;color:#a5d6a7}
.badge-finished{background:#e65100;color:#ffcc80}
.container{display:flex;height:calc(100vh - 52px)}
.main-area{flex:1;padding:16px;overflow-y:auto}
.sidebar{width:320px;background:#16213e;padding:16px;border-left:1px solid #0f3460;overflow-y:auto;display:none}
/* ─ Idle Panel ─ */
.idle-panel{max-width:700px;margin:40px auto}
.tabs{display:flex;gap:4px;margin-bottom:16px}
.tab-btn{padding:10px 20px;background:#0f3460;border:none;color:#aaa;cursor:pointer;border-radius:6px 6px 0 0;font-size:0.9rem}
.tab-btn.active{background:#1a4080;color:#4fc3f7}
.tab-content{background:#16213e;border-radius:0 8px 8px 8px;padding:24px}
.tab-pane{display:none}
.tab-pane.active{display:block}
input[type=text],input[type=file]{width:100%;padding:10px;border-radius:6px;border:1px solid #333;background:#1a1a2e;color:#eee;margin-bottom:12px}
.btn{padding:10px 20px;border:none;border-radius:6px;cursor:pointer;font-weight:600;font-size:0.9rem}
.btn-primary{background:#1976d2;color:#fff}
.btn-primary:hover{background:#1565c0}
.btn-danger{background:#c62828;color:#fff}
.btn-danger:hover{background:#b71c1c}
.btn-success{background:#2e7d32;color:#fff}
.btn-success:hover{background:#1b5e20}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.upload-zone{border:2px dashed #333;border-radius:8px;padding:40px;text-align:center;margin-bottom:16px;cursor:pointer}
.upload-zone:hover{border-color:#4fc3f7}
.file-list{list-style:none;max-height:200px;overflow-y:auto}
.file-list li{padding:8px 12px;background:#1a1a2e;margin-bottom:4px;border-radius:4px;display:flex;justify-content:space-between;align-items:center;cursor:pointer}
.file-list li.selected{border:1px solid #4fc3f7}
.file-list li:hover{background:#222}
/* ─ Running Panel ─ */
.running-panel{display:none}
.video-box{background:#000;border-radius:8px;overflow:hidden;position:relative}
.video-box img{width:100%;display:block}
.stats-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px}
.stat-card{background:#16213e;border-radius:8px;padding:12px;text-align:center}
.stat-card .val{font-size:1.6rem;font-weight:700;color:#4fc3f7}
.stat-card .lbl{font-size:0.75rem;color:#888;margin-top:4px}
/* ─ Finished Panel ─ */
.finished-panel{display:none;max-width:800px;margin:20px auto}
.result-summary{background:#16213e;border-radius:8px;padding:20px;margin-bottom:16px}
.diagnosis-box{background:#16213e;border-radius:8px;padding:20px;margin-top:16px}
.diagnosis-box h3{color:#4fc3f7;margin-bottom:12px}
.dl-links a{display:inline-block;margin-right:12px;color:#4fc3f7;text-decoration:none;padding:6px 12px;border:1px solid #4fc3f7;border-radius:4px;margin-top:8px}
.dl-links a:hover{background:#4fc3f7;color:#000}
.chart-box{height:180px;background:#111;border-radius:8px;margin-top:12px;position:relative}
canvas{width:100%!important;height:100%!important}
.progress-bar{height:4px;background:#333;border-radius:2px;margin-top:8px}
.progress-bar .fill{height:100%;background:#4fc3f7;border-radius:2px;transition:width 0.3s}
</style>
</head>
<body>
<div class="header">
  <h1>&#x1f437; 猪只计数监控系统</h1>
  <span id="badge" class="status-badge badge-idle">空闲</span>
</div>
<div class="container">
<div class="main-area">
<!-- IDLE -->
<div class="idle-panel" id="panelIdle">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('camera')">摄像头</button>
    <button class="tab-btn" onclick="switchTab('upload')">上传视频</button>
  </div>
  <div class="tab-content">
    <div class="tab-pane active" id="tabCamera">
      <p style="margin-bottom:12px;color:#aaa">输入 RTSP 地址开始实时监控</p>
      <input type="text" id="rtspUrl" placeholder="rtsp://admin:admin123@192.168.1.108:554/...">
      <button class="btn btn-primary" onclick="startCamera()">开始监控</button>
    </div>
    <div class="tab-pane" id="tabUpload">
      <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <p>点击或拖拽视频文件到此处上传</p>
        <p style="color:#666;font-size:0.8rem;margin-top:8px">支持 mp4/avi/mkv，最大 500MB</p>
      </div>
      <input type="file" id="fileInput" accept=".mp4,.avi,.mkv,.mov" style="display:none" onchange="uploadFile(this)">
      <div class="progress-bar" id="uploadProgress" style="display:none"><div class="fill" id="uploadFill"></div></div>
      <ul class="file-list" id="fileList"></ul>
      <button class="btn btn-primary" id="btnStartVideo" disabled onclick="startVideo()">开始推理</button>
    </div>
  </div>
</div>
<!-- RUNNING -->
<div class="running-panel" id="panelRunning">
  <div class="video-box"><img id="streamImg" src="/stream_frame"></div>
  <div class="stats-grid">
    <div class="stat-card"><div class="val" id="sTotal">0</div><div class="lbl">总计数</div></div>
    <div class="stat-card"><div class="val" id="sFps">0</div><div class="lbl">推理 FPS</div></div>
    <div class="stat-card"><div class="val" id="sValid">0</div><div class="lbl">有效轨迹</div></div>
    <div class="stat-card"><div class="val" id="sIds">0</div><div class="lbl">总 ID 数</div></div>
  </div>
  <div style="margin-top:12px;text-align:right">
    <button class="btn btn-danger" onclick="stopInference()">停止</button>
  </div>
  <div class="chart-box"><canvas id="chartCanvas"></canvas></div>
</div>
<!-- FINISHED -->
<div class="finished-panel" id="panelFinished">
  <div class="result-summary">
    <h3 style="color:#4fc3f7;margin-bottom:12px">推理完成</h3>
    <p>总计数: <strong id="fTotal">0</strong> | 有效轨迹: <strong id="fValid">0</strong> | 总 ID: <strong id="fIds">0</strong></p>
    <div class="dl-links">
      <a href="/download/summary.csv">汇总 CSV</a>
      <a href="/download/id_events.csv">ID 事件</a>
      <a href="/download/trajectory.csv">轨迹报告</a>
    </div>
  </div>
  <div style="margin-bottom:12px">
    <label style="color:#aaa;margin-right:8px">诊断来源:</label>
    <select id="diagSource" style="padding:6px;border-radius:4px;background:#1a1a2e;color:#eee;border:1px solid #333">
      <option value="current">当前监控结果</option>
    </select>
    <button class="btn btn-success" onclick="runDiagnosis()">生成诊断报告</button>
  </div>
  <div class="diagnosis-box" id="diagBox" style="display:none">
    <h3>诊断报告</h3>
    <div id="diagContent"></div>
    <div class="dl-links" style="margin-top:12px">
      <a href="/download/diagnosis.json">下载 JSON</a>
      <a href="/download/diagnosis.md">下载 Markdown</a>
    </div>
  </div>
  <div style="margin-top:16px">
    <button class="btn btn-primary" onclick="backToIdle()">返回首页</button>
  </div>
</div>
</div>
<div class="sidebar" id="sidebar">
  <h3 style="color:#4fc3f7;margin-bottom:12px">线计数</h3>
  <div id="lineCounters"></div>
  <h3 style="color:#4fc3f7;margin:16px 0 8px">Agent 状态</h3>
  <div id="agentStatus" style="font-size:0.85rem;color:#aaa"></div>
</div>
</div>
<script>
let mode='idle', selectedFile=null, chart=null, historyData=[];
function switchTab(t){
  document.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active',i===(t==='camera'?0:1)));
  document.getElementById('tabCamera').classList.toggle('active',t==='camera');
  document.getElementById('tabUpload').classList.toggle('active',t==='upload');
}
function showPanel(m){
  mode=m;
  document.getElementById('panelIdle').style.display=m==='idle'?'block':'none';
  document.getElementById('panelRunning').style.display=m==='running'?'block':'none';
  document.getElementById('panelFinished').style.display=m==='finished'?'block':'none';
  document.getElementById('sidebar').style.display=m==='running'?'block':'none';
  const b=document.getElementById('badge');
  b.className='status-badge '+(m==='idle'?'badge-idle':m==='running'?'badge-running':'badge-finished');
  b.textContent=m==='idle'?'空闲':m==='running'?'运行中':'已完成';
}
function startCamera(){
  const url=document.getElementById('rtspUrl').value.trim();
  if(!url){alert('请输入RTSP地址');return;}
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:url})})
  .then(r=>r.json()).then(d=>{if(d.ok)showPanel('running');else alert(d.error||'启动失败');});
}
function startVideo(){
  if(!selectedFile){alert('请先选择视频');return;}
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:'upload:'+selectedFile})})
  .then(r=>r.json()).then(d=>{if(d.ok)showPanel('running');else alert(d.error||'启动失败');});
}
function stopInference(){
  fetch('/api/stop',{method:'POST'}).then(()=>{});
}
function uploadFile(input){
  const file=input.files[0];if(!file)return;
  const fd=new FormData();fd.append('file',file);
  const bar=document.getElementById('uploadProgress'),fill=document.getElementById('uploadFill');
  bar.style.display='block';fill.style.width='0%';
  const xhr=new XMLHttpRequest();
  xhr.upload.onprogress=e=>{if(e.lengthComputable)fill.style.width=(e.loaded/e.total*100)+'%';};
  xhr.onload=()=>{bar.style.display='none';if(xhr.status===200)loadUploads();else alert('上传失败');};
  xhr.onerror=()=>{bar.style.display='none';alert('上传出错');};
  xhr.open('POST','/api/upload');xhr.send(fd);
}
function loadUploads(){
  fetch('/api/uploads').then(r=>r.json()).then(files=>{
    const ul=document.getElementById('fileList');ul.innerHTML='';
    const sel=document.getElementById('diagSource');
    while(sel.options.length>1)sel.remove(1);
    files.forEach(f=>{
      const li=document.createElement('li');li.textContent=f;
      li.onclick=()=>{selectedFile=f;document.querySelectorAll('.file-list li').forEach(x=>x.classList.remove('selected'));li.classList.add('selected');document.getElementById('btnStartVideo').disabled=false;};
      ul.appendChild(li);
      const opt=document.createElement('option');opt.value='upload:'+f;opt.textContent='上传: '+f;sel.appendChild(opt);
    });
  });
}
function runDiagnosis(){
  const src=document.getElementById('diagSource').value;
  fetch('/api/diagnose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})})
  .then(r=>r.json()).then(d=>{
    if(d.error){alert(d.error);return;}
    const box=document.getElementById('diagBox');box.style.display='block';
    let html='<table style="width:100%;border-collapse:collapse">';
    html+='<tr><td style="padding:6px;color:#aaa">主要原因</td><td style="padding:6px">'+esc(d.primary_cause||'-')+'</td></tr>';
    html+='<tr><td style="padding:6px;color:#aaa">置信度</td><td style="padding:6px">'+(d.confidence||'-')+'</td></tr>';
    html+='<tr><td style="padding:6px;color:#aaa">次要原因</td><td style="padding:6px">'+(d.secondary_causes||[]).join(', ')+'</td></tr>';
    html+='<tr><td style="padding:6px;color:#aaa">可疑窗口</td><td style="padding:6px">'+(d.suspect_windows||[]).length+'个</td></tr>';
    if(d.recommendation)html+='<tr><td style="padding:6px;color:#aaa">建议</td><td style="padding:6px">'+esc(d.recommendation)+'</td></tr>';
    html+='</table>';
    document.getElementById('diagContent').innerHTML=html;
  });
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function backToIdle(){showPanel('idle');historyData=[];}
function update(){
  fetch('/api/stats').then(r=>r.json()).then(d=>{
    if(d.mode==='running'&&mode!=='running')showPanel('running');
    else if(d.mode==='finished'&&mode!=='finished'){
      showPanel('finished');
      document.getElementById('fTotal').textContent=d.total_count;
      document.getElementById('fValid').textContent=d.valid_traj;
      document.getElementById('fIds').textContent=d.total_ids;
    }
    if(d.mode==='running'){
      document.getElementById('sTotal').textContent=d.total_count;
      document.getElementById('sFps').textContent=d.fps_inference;
      document.getElementById('sValid').textContent=d.valid_traj;
      document.getElementById('sIds').textContent=d.total_ids;
      document.getElementById('lineCounters').innerHTML=Object.entries(d.line_counters||{}).map(([k,v])=>'<div style="margin:4px 0">'+k+': <strong>'+v+'</strong></div>').join('');
      document.getElementById('agentStatus').textContent='帧: '+d.frame_count+' | 源FPS: '+d.fps_source;
      if(d.history&&d.history.length)historyData=d.history;
      drawChart();
    }
  }).catch(()=>{});
}
function drawChart(){
  const canvas=document.getElementById('chartCanvas');if(!canvas)return;
  const ctx=canvas.getContext('2d'),W=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
  canvas.width=W;canvas.height=H;ctx.clearRect(0,0,W,H);
  if(historyData.length<2)return;
  const maxT=historyData[historyData.length-1].time,maxV=Math.max(...historyData.map(p=>p.total),1);
  ctx.strokeStyle='#4fc3f7';ctx.lineWidth=2;ctx.beginPath();
  historyData.forEach((p,i)=>{const x=p.time/maxT*(W-20)+10,y=H-10-p.total/maxV*(H-20);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
}
let refreshImg;
function refreshStream(){
  const img=document.getElementById('streamImg');
  if(mode==='running')img.src='/stream_frame?t='+Date.now();
}
setInterval(update,1500);
setInterval(refreshStream,200);
document.addEventListener('DOMContentLoaded',()=>{loadUploads();update();});
const dz=document.getElementById('dropZone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.style.borderColor='#4fc3f7';});
dz.addEventListener('dragleave',()=>{dz.style.borderColor='#333';});
dz.addEventListener('drop',e=>{e.preventDefault();dz.style.borderColor='#333';const f=e.dataTransfer.files[0];if(f){document.getElementById('fileInput').files=e.dataTransfer.files;uploadFile(document.getElementById('fileInput'));}});
</script>
</body>
</html>"""
# ── END HTML (JS follows) ──
# ── HTTP Handler ─────────────────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse
import mimetypes


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._html_response(HTML_PAGE)

        elif path == '/api/stats':
            with app_state['lock']:
                data = {
                    'mode': app_state['mode'],
                    'fps_inference': app_state['fps_inference'],
                    'fps_source': app_state['fps_source'],
                    'frame_count': app_state['frame_count'],
                    'line_counters': app_state['line_counters'],
                    'total_count': app_state['total_count'],
                    'valid_traj': app_state['valid_traj'],
                    'total_ids': app_state['total_ids'],
                    'source': app_state['source'],
                    'resolution': app_state['resolution'],
                    'history': app_state['history'][-200:],
                }
            self._json_response(data)

        elif path == '/stream_frame':
            with app_state['lock']:
                jpeg = app_state.get('frame_jpeg')
            if jpeg:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self.send_response(204)
                self.end_headers()

        elif path == '/api/uploads':
            upload_dir = Path(server_config['output_dir']) / 'uploads'
            files = []
            if upload_dir.exists():
                files = sorted([f.name for f in upload_dir.iterdir() if f.is_file()])
            self._json_response(files)

        elif path == '/api/diagnosis':
            with app_state['lock']:
                diag = app_state.get('diagnosis')
            if diag:
                self._json_response(diag)
            else:
                self._json_response({'error': '暂无诊断报告'}, 404)

        elif path.startswith('/download/'):
            fpath = resolve_download_path(path)
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
                    self._json_response({'ok': False, 'error': '文件不存在'}, 400)
                    return
                source = str(filepath)
            if not source:
                self._json_response({'ok': False, 'error': '未指定数据源'}, 400)
                return
            ok = start_inference(source)
            if ok:
                self._json_response({'ok': True})
            else:
                self._json_response({'ok': False, 'error': '已有推理在运行'}, 409)

        elif path == '/api/stop':
            stop_inference()
            self._json_response({'ok': True})

        elif path == '/api/upload':
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self._json_response({'error': '需要 multipart/form-data'}, 400)
                return
            length = int(self.headers.get('Content-Length', 0))
            if length > MAX_UPLOAD_SIZE:
                self._json_response({'error': '文件过大'}, 413)
                return
            boundary = content_type.split('boundary=')[1].encode()
            raw = self.rfile.read(length)
            filename, filedata = self._parse_multipart(raw, boundary)
            if not filename:
                self._json_response({'error': '未找到文件'}, 400)
                return
            upload_dir = Path(server_config['output_dir']) / 'uploads'
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / Path(filename).name
            dest.write_bytes(filedata)
            self._json_response({'ok': True, 'filename': dest.name})

        elif path == '/api/diagnose':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            result = run_diagnosis()
            if result:
                self._json_response(result)
            else:
                self._json_response({'error': '无法生成诊断报告，请先完成一次推理'}, 400)
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

    source = args.rtsp or args.video
    if source:
        start_inference(source)

    server = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    print(f"[WEB] http://0.0.0.0:{args.port}")
    print(f"[MODE] {'Auto-started: ' + source if source else 'Idle — waiting for frontend control'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_inference()
        server.shutdown()
        print("\n[EXIT] Server stopped")


if __name__ == '__main__':
    main()
