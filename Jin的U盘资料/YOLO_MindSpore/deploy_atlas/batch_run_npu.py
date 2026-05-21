#!/usr/bin/env python3
"""Batch NPU inference for pig counting on Atlas 200I DK A2."""
import re, csv, time
from pathlib import Path
import subprocess, sys

SCRIPT_DIR = Path(__file__).parent
TRACK_SCRIPT = SCRIPT_DIR / 'track_and_count_npu.py'
OM_PATH = SCRIPT_DIR / 'models' / 'yolov8n_pig_fp16.om'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--track_thresh', type=float, default=0.5)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob('*.mp4'), key=lambda p: p.stem)
    # Try numeric sort if possible
    try:
        videos = sorted(video_dir.glob('*.mp4'),
                        key=lambda p: int(re.match(r'(\d+)', p.stem).group(1)))
    except:
        pass

    results = []
    for video in videos:
        out_dir = output_base / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nProcessing: {video.name}')

        cmd = [
            sys.executable, str(TRACK_SCRIPT),
            '--video', str(video),
            '--om', str(OM_PATH),
            '--output_dir', str(out_dir),
            '--conf_thres', str(args.conf_thres),
            '--track_thresh', str(args.track_thresh),
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
            print(f'  actual={actual} total_line={total_line} valid_traj={valid_traj} ({elapsed:.1f}s)')
        except Exception as e:
            print(f'  Failed: {e}')
            results.append({'video': video.name, 'actual': None, 'total_line': None,
                            'valid_traj': None, 'error_line': None, 'elapsed': None})

    out_csv = output_base / 'batch_results.csv'
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video', 'actual', 'total_line', 'valid_traj', 'error_line', 'elapsed'])
        writer.writeheader()
        writer.writerows(results)

    print(f'\nResults saved: {out_csv}')
    errors = [r for r in results if r['error_line'] and r['error_line'] != 0]
    print(f'Errors: {len(errors)}/{len(results)}')
    for r in errors:
        print(f"  {r['video']}: actual={r['actual']} detected={r['total_line']} error={r['error_line']:+d}")


if __name__ == '__main__':
    main()
