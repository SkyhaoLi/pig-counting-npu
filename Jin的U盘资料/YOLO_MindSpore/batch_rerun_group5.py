#!/usr/bin/env python3
import re, csv, subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRACK_SCRIPT = SCRIPT_DIR / 'track_and_count.py'
MODEL_PATH = Path('E:/Jin的U盘资料/YOLO/runs/detect/train/weights/best.pt')
VIDEO_DIR = SCRIPT_DIR / '数据集/五'
OUTPUT_BASE = Path('C:/Users/Skyha/Desktop/pig_couter/output/batch_rerun_group5')
PYTHON = r'C:\Users\Skyha\.conda\envs\ai-study\python.exe'

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

videos = sorted(VIDEO_DIR.glob('*.mp4'), key=lambda p: p.stem)
results = []

for video in videos:
    out_dir = OUTPUT_BASE / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n处理: {video.name}')
    cmd = [
        PYTHON, str(TRACK_SCRIPT),
        '--video_path', str(video),
        '--model_path', str(MODEL_PATH),
        '--output_dir', str(out_dir),
        '--no_timestamp'
    ]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        summary = out_dir / 'ByteTrack_summary.csv'
        total_line = None
        valid_traj = None
        if summary.exists():
            with open(summary, encoding='utf-8') as f:
                row = list(csv.DictReader(f))[0]
                total_line = int(row['total_line'])
                valid_traj = int(row['valid_traj'])
        m = re.search(r'-(\d+)头', video.stem)
        actual = int(m.group(1)) if m else None
        error = (total_line - actual) if (total_line is not None and actual is not None) else None
        results.append({'video': video.name, 'actual': actual, 'total_line': total_line,
                        'valid_traj': valid_traj, 'error_line': error, 'elapsed': round(elapsed, 1)})
        print(f'  actual={actual} total_line={total_line} valid_traj={valid_traj}')
    except Exception as e:
        print(f'  失败: {e}')
        results.append({'video': video.name, 'actual': None, 'total_line': None,
                        'valid_traj': None, 'error_line': None, 'elapsed': None})

out_csv = OUTPUT_BASE / 'batch_rerun_group5_results.csv'
with open(out_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video','actual','total_line','valid_traj','error_line','elapsed'])
    writer.writeheader()
    writer.writerows(results)

print(f'\n结果已保存: {out_csv}')
errors = [r for r in results if r['error_line'] and r['error_line'] != 0]
print(f'有误差: {len(errors)}/{len(results)}')
for r in errors:
    print(f"  {r['video']}: actual={r['actual']} detected={r['total_line']} error={r['error_line']:+d}")
