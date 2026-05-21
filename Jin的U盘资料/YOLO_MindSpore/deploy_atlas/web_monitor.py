#!/usr/bin/env python3
"""
Pig Counting Web Monitor - Atlas 200I DK A2
Real-time RTSP/video stream + NPU inference + ByteTrack + Web UI

Usage:
    python3 web_monitor.py --rtsp "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    python3 web_monitor.py --video datasets/si/11.mp4
    python3 web_monitor.py --rtsp "rtsp://..." --port 8080
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

# ── Globals ──────────────────────────────────────────────────
app_state = {
    'running': False,
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
    'history': [],       # [{time, total}]
    'result_root': '',
    'current_result_dir': '',
    'completed_result_dir': '',
    'lock': threading.Lock(),
    'reset_flag': False,
}


# ── Zone Analyzer (simplified for real-time) ─────────────────
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


# ── Frame grabber thread (drains RTSP buffer for low latency) ─
latest_frame = {'frame': None, 'lock': threading.Lock()}

def grabber_loop(cap):
    """Continuously grab frames so the buffer never stalls."""
    while app_state.get('running', True):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with latest_frame['lock']:
            latest_frame['frame'] = frame


# ── Inference thread ─────────────────────────────────────────
def inference_loop(source, om_path, conf_thres, track_thresh, out_ratio, wait_ratio, output_root):
    global app_state

    detector = NPUDetector(om_path, conf_thres=conf_thres)

    if source.startswith('rtsp://') or source.startswith('http://'):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_stream = True
    else:
        cap = cv2.VideoCapture(source)
        is_stream = False

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
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

    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)
    output_root = Path(output_root)
    current_result_dir = output_root / "current"
    completed_result_dir = output_root / "latest_completed"
    current_result_dir.mkdir(parents=True, exist_ok=True)
    completed_result_dir.mkdir(parents=True, exist_ok=True)

    byte_args = SimpleNamespace(track_thresh=track_thresh, track_buffer=30, match_thresh=0.8, mot20=False)
    tracker = BYTETracker(byte_args, frame_rate=max(1, int(fps)))

    # Start grabber thread for streams to drain buffer
    if is_stream:
        gt = threading.Thread(target=grabber_loop, args=(cap,), daemon=True)
        gt.start()

    frame_idx = 0
    t_last = time.time()
    fps_counter = 0
    last_export = 0.0

    with app_state['lock']:
        app_state['result_root'] = str(output_root)
        app_state['current_result_dir'] = str(current_result_dir)
        app_state['completed_result_dir'] = str(completed_result_dir)

    while app_state['running']:
        # Check reset
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
            # Always take the latest frame (skip stale ones)
            with latest_frame['lock']:
                frame = latest_frame['frame']
            if frame is None:
                time.sleep(0.01)
                continue
            # Clear so we don't reprocess same frame
            with latest_frame['lock']:
                latest_frame['frame'] = None
        else:
            ret, frame = cap.read()
            if not ret:
                export_result_files(analyzer, current_result_dir)
                snapshot_completed_results(current_result_dir, completed_result_dir)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)
                tracker = BYTETracker(byte_args, frame_rate=max(1, int(fps)))
                continue

        # Detect
        raw_dets = detector.detect(frame)
        detections = []
        for det in raw_dets:
            if not is_blue_object(frame, det[:5]):
                detections.append(det[:5])

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

        # Encode JPEG
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])

        # FPS calc
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
            # History point every 5 seconds
            elapsed = time.time() - app_state['start_time']
            if len(app_state['history']) == 0 or elapsed - app_state['history'][-1]['time'] >= 5:
                app_state['history'].append({'time': round(elapsed, 1), 'total': total})
            if elapsed - last_export >= 5:
                do_export = True
                last_export = elapsed

        if do_export:
            export_result_files(analyzer, current_result_dir)

        frame_idx += 1

    export_result_files(analyzer, current_result_dir)
    cap.release()


# ── HTTP server (no Flask dependency) ────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pig Counter - Live Monitor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; }
.header { background: #1e293b; padding: 12px 24px; display: flex;
           align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; }
.header h1 { font-size: 1.2rem; color: #38bdf8; }
.header .status { font-size: 0.85rem; }
.status .live { color: #22c55e; font-weight: bold; }
.container { display: grid; grid-template-columns: 1fr 340px; gap: 16px;
             padding: 16px; height: calc(100vh - 56px); }
.video-panel { position: relative; background: #000; border-radius: 8px; overflow: hidden; }
.video-panel img { width: 100%; height: 100%; object-fit: contain; }
.side-panel { display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
.card { background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }
.card h3 { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;
            letter-spacing: 0.05em; margin-bottom: 12px; }
.big-number { font-size: 3rem; font-weight: 700; color: #22c55e; text-align: center; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat { text-align: center; padding: 8px; background: #0f172a; border-radius: 6px; }
.stat .label { font-size: 0.75rem; color: #64748b; }
.stat .value { font-size: 1.1rem; font-weight: 600; margin-top: 2px; }
.line-stats { display: flex; flex-direction: column; gap: 6px; }
.line-row { display: flex; justify-content: space-between; align-items: center;
            padding: 6px 10px; background: #0f172a; border-radius: 4px; }
.line-row .name { font-size: 0.85rem; }
.line-row .count { font-size: 1rem; font-weight: 600; }
.line0 .name { color: #fb923c; }
.line1 .name, .line2 .name { color: #22d3ee; }
.btn { display: inline-block; padding: 8px 16px; border-radius: 6px; border: none;
       cursor: pointer; font-size: 0.85rem; font-weight: 500; text-decoration: none; }
.btn-primary { background: #3b82f6; color: white; }
.btn-primary:hover { background: #2563eb; }
.btn-danger { background: #ef4444; color: white; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
canvas { width: 100%; height: 120px; }
@media (max-width: 900px) {
    .container { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="header">
    <h1>Pig Counter - NPU Live Monitor</h1>
    <div class="status"><span class="live" id="status-dot">LIVE</span>
        <span id="info-src"></span></div>
</div>
<div class="container">
    <div class="video-panel">
        <img id="stream" src="/stream" alt="Video Stream">
    </div>
    <div class="side-panel">
        <div class="card">
            <h3>Total Count</h3>
            <div class="big-number" id="total">0</div>
        </div>
        <div class="card">
            <h3>Line Crossings</h3>
            <div class="line-stats">
                <div class="line-row line0">
                    <span class="name">Line 0 (OUT mid)</span>
                    <span class="count" id="line0">0</span>
                </div>
                <div class="line-row line1">
                    <span class="name">Line 1 (OUT|WAIT)</span>
                    <span class="count" id="line1">0</span>
                </div>
                <div class="line-row line2">
                    <span class="name">Line 2 (WAIT|ENTRY)</span>
                    <span class="count" id="line2">0</span>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Statistics</h3>
            <div class="stat-grid">
                <div class="stat">
                    <div class="label">Inference FPS</div>
                    <div class="value" id="fps">0</div>
                </div>
                <div class="stat">
                    <div class="label">Active IDs</div>
                    <div class="value" id="ids">0</div>
                </div>
                <div class="stat">
                    <div class="label">Valid Traj</div>
                    <div class="value" id="valid">0</div>
                </div>
                <div class="stat">
                    <div class="label">Frames</div>
                    <div class="value" id="frames">0</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Trend</h3>
            <canvas id="chart"></canvas>
        </div>
        <div class="card">
            <h3>Actions</h3>
            <div class="actions">
                <a class="btn btn-primary" href="/download/summary.csv" download>Summary CSV</a>
                <a class="btn btn-primary" href="/download/id_events.csv" download>Events CSV</a>
                <a class="btn btn-primary" href="/download/trajectory.csv" download>Trajectory CSV</a>
                <a class="btn btn-primary" href="/download/state_changes.txt" download>State TXT</a>
                <button class="btn btn-danger" onclick="resetCount()">Reset</button>
            </div>
        </div>
    </div>
</div>
<script>
function update() {
    fetch('/api/stats').then(r => r.json()).then(d => {
        document.getElementById('total').textContent = d.total_count;
        document.getElementById('line0').textContent = d.line0;
        document.getElementById('line1').textContent = d.line1;
        document.getElementById('line2').textContent = d.line2;
        document.getElementById('fps').textContent = d.fps_inference;
        document.getElementById('ids').textContent = d.total_ids;
        document.getElementById('valid').textContent = d.valid_traj;
        document.getElementById('frames').textContent = d.frame_count;
        document.getElementById('info-src').textContent =
            d.resolution + ' | ' + d.fps_source + ' fps';
        drawChart(d.history);
    }).catch(() => {});
}
function drawChart(history) {
    const c = document.getElementById('chart');
    const ctx = c.getContext('2d');
    c.width = c.offsetWidth * 2; c.height = c.offsetHeight * 2;
    ctx.scale(2, 2);
    const W = c.offsetWidth, H = c.offsetHeight;
    ctx.clearRect(0, 0, W, H);
    if (!history || history.length < 2) return;
    const maxT = Math.max(...history.map(h => h.total), 1);
    ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    history.forEach((h, i) => {
        const x = (i / (history.length - 1)) * W;
        const y = H - (h.total / maxT) * (H - 10) - 5;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
}
function resetCount() {
    fetch('/api/reset', {method:'POST'}).then(r => r.json()).then(d => {
        if (d.ok) {
            document.getElementById('total').textContent = '0';
            document.getElementById('line0').textContent = '0';
            document.getElementById('line1').textContent = '0';
            document.getElementById('line2').textContent = '0';
            document.getElementById('frames').textContent = '0';
            document.getElementById('ids').textContent = '0';
            document.getElementById('valid').textContent = '0';
            var c = document.getElementById('chart');
            c.getContext('2d').clearRect(0,0,c.width,c.height);
        }
    });
}
setInterval(update, 500);
update();
</script>
</body>
</html>'''


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while app_state['running']:
                    with app_state['lock']:
                        jpeg = app_state.get('frame_jpeg')
                    if jpeg:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    time.sleep(0.08)  # ~12 fps max for web
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        elif self.path == '/api/stats':
            with app_state['lock']:
                data = {
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
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        elif self.path.startswith('/download/'):
            file_path = resolve_download_path(self.path)
            if not file_path:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            content_type = 'text/plain'
            if file_path.suffix == '.csv':
                content_type = 'text/csv'
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Disposition', f'attachment; filename={file_path.name}')
            self.end_headers()
            self.wfile.write(file_path.read_bytes())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/reset':
            with app_state['lock']:
                app_state['reset_flag'] = True
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description='Pig Counter Web Monitor')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL (full)')
    parser.add_argument('--camera_ip', type=str, help='Camera IP (builds RTSP URL)')
    parser.add_argument('--camera_user', type=str, default='admin')
    parser.add_argument('--camera_pass', type=str, default='admin123')
    parser.add_argument('--video', type=str, help='Local video file path')
    parser.add_argument('--om', type=str, default='models/yolov8n_pig_fp16.om')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--out_ratio', type=float, default=0.45)
    parser.add_argument('--wait_ratio', type=float, default=0.25)
    parser.add_argument('--output_dir', type=str, default='output/web_monitor')
    args = parser.parse_args()

    if args.camera_ip:
        # Don't URL-encode: OpenCV RTSP splits on the LAST @ in the authority
        # Use subtype=1 (sub stream) for lower resolution, better for web streaming
        source = f"rtsp://{args.camera_user}:{args.camera_pass}@{args.camera_ip}:554/cam/realmonitor?channel=1&subtype=1"
    else:
        source = args.rtsp or args.video
    if not source:
        print("Error: specify --camera_ip, --rtsp, or --video")
        sys.exit(1)

    print(f"Source: {source}")
    print(f"Model: {args.om}")
    print(f"Web UI: http://0.0.0.0:{args.port}")

    # Start inference in background thread
    t = threading.Thread(target=inference_loop, args=(
        source, args.om, args.conf_thres, args.track_thresh,
        args.out_ratio, args.wait_ratio, args.output_dir), daemon=True)
    t.start()

    # Wait for first frame
    print("Waiting for first frame...")
    for _ in range(100):
        if app_state['frame_jpeg'] is not None:
            break
        time.sleep(0.1)

    # Start HTTP server (threaded so /stream doesn't block other requests)
    server = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    print(f"\n>>> Open http://192.168.137.100:{args.port} in your browser <<<\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app_state['running'] = False
        server.shutdown()
        print("\nShutdown.")


if __name__ == '__main__':
    main()
