#!/usr/bin/env python3
"""Local smoke tests for the pig-counting agents."""

from __future__ import annotations

import csv
import json
import sys
import types
import tempfile
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DEPLOY_ATLAS = ROOT / "deploy_atlas"
if str(DEPLOY_ATLAS) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ATLAS))

from review_agent import HumanReviewAgent
from diagnosis_agent import DiagnosisAgent
from autonomous_agent import AutonomousOpsAgent


def _import_web_monitor():
    """Import web_monitor with mocked hardware dependencies."""
    for mod_name in ['cv2', 'numpy', 'npu_detector',
                     'trackers', 'trackers.byte_tracker',
                     'trackers.byte_tracker.byte_tracker',
                     'track_and_count_npu']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    np_mock = sys.modules['numpy']
    np_mock.empty = lambda *a, **k: []
    np_mock.array = lambda *a, **k: []
    np_mock.mean = lambda *a, **k: 0
    np_mock.random = types.ModuleType('numpy.random')
    np_mock.random.seed = lambda x: None
    np_mock.random.randint = lambda *a, **k: [100, 100, 100]
    sys.modules['numpy.random'] = np_mock.random

    cv2_mock = sys.modules['cv2']
    cv2_mock.VideoCapture = lambda *a, **k: None
    cv2_mock.CAP_FFMPEG = 0
    cv2_mock.CAP_PROP_BUFFERSIZE = 0
    cv2_mock.CAP_PROP_FRAME_WIDTH = 0
    cv2_mock.CAP_PROP_FRAME_HEIGHT = 0
    cv2_mock.CAP_PROP_FPS = 0
    cv2_mock.CAP_PROP_POS_FRAMES = 0
    cv2_mock.IMWRITE_JPEG_QUALITY = 0

    npu_mod = sys.modules['npu_detector']
    class FakeDetector:
        def __init__(self, *a, **k): pass
        def detect(self, *a): return []
    npu_mod.NPUDetector = FakeDetector

    tc_mod = sys.modules['track_and_count_npu']
    class FakeZone:
        def __init__(self, *a, **k):
            self.split_0 = self.split_1 = self.split_2 = 0
            self.line_counters = {'line0': 0, 'line1': 0, 'line2': 0}
            self.trajectories = {}
        def update(self, *a, **k): pass
        def finalize(self, *a, **k): pass
    tc_mod.ZoneAnalyzer = FakeZone

    bt_mod = sys.modules['trackers.byte_tracker.byte_tracker']
    class FakeBYTE:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): return []
    bt_mod.BYTETracker = FakeBYTE

    import web_monitor
    return web_monitor


def test_review_agent():
    agent = HumanReviewAgent(ROOT / "review_registry.json")
    corrected = agent.assess("17-37头.mp4", 37, 32)
    assert corrected["review_status"] == "human_reviewed"
    assert corrected["paper_actual"] == 31
    assert corrected["paper_count"] == 32
    assert corrected["paper_error"] == 1

    raw = agent.assess("15-47头.mp4", 47, 46)
    assert raw["review_status"] == "needs_review"
    assert raw["paper_error"] == -1
    print("OK review_agent")


def test_autonomous_agent():
    agent = AutonomousOpsAgent(
        log_dir=None,
        stale_frame_seconds=0.01,
        reconnect_failure_threshold=2,
        low_fps_threshold=4.0,
        drift_threshold=6,
    )
    agent.note_stream_started("rtsp://demo")
    agent.note_frame(frame_idx=10, infer_fps=3.0, total_count=100, valid_traj=92, total_ids=120)
    agent.note_waiting_for_frame(wait_seconds=0.02, failure_streak=2)
    snapshot = agent.snapshot()
    actions = agent.consume_actions()

    assert snapshot["status"] == "RECOVERING"
    assert snapshot["reconnect_requests"] == 1
    assert actions["reconnect"] is True
    print("OK autonomous_agent")


def test_manual_review_store():
    web_monitor = _import_web_monitor()

    review_dir = Path(tempfile.mkdtemp(prefix="pig_agent_test_"))
    review_file = review_dir / "manual_reviews.jsonl"
    web_monitor.review_store["path"] = review_file
    with web_monitor.app_state["lock"]:
        web_monitor.app_state["manual_reviews"] = []

    entry = web_monitor.append_manual_review("stale_frame", "confirmed", "operator accepted restart")
    assert entry["subject"] == "stale_frame"
    assert review_file.exists()

    content = review_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    stored = json.loads(content[0])
    assert stored["decision"] == "confirmed"

    with web_monitor.app_state["lock"]:
        assert web_monitor.app_state["manual_reviews"][0]["note"] == "operator accepted restart"

    shutil.rmtree(review_dir)
    print("OK manual_review_store")


def test_generated_review_csv():
    csv_path = ROOT.parent.parent / "output" / "batch_rerun_group4_mindspore_review" / "batch_rerun_group4_results.csv"
    if not csv_path.exists():
        csv_path = ROOT.parent.parent / "output" / "batch_rerun_group4_review" / "batch_rerun_group4_results.csv"
    if not csv_path.exists():
        print("SKIP review_csv (missing generated file)")
        return

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["video"].startswith("17-37"))
    assert row["paper_actual"] == "31"
    assert row["paper_count"] == "32"
    assert row["paper_error"] == "1"
    print("OK generated_review_csv")


def test_diagnosis_agent():
    out_dir = ROOT.parent.parent / "output" / "batch_rerun_group4_mindspore_review" / "17-37头"
    if not out_dir.exists():
        out_dir = ROOT.parent.parent / "output" / "batch_rerun_group4_review" / "17-37头"
    if not out_dir.exists():
        print("SKIP diagnosis_agent (missing rerun output)")
        return
    agent = DiagnosisAgent()
    diagnosis = agent.analyze(out_dir, "17-37头.mp4", actual=37, paper_actual=31, total_line=32, valid_traj=30)
    assert diagnosis["primary_cause"] == "标签定义与统计方向不一致"
    assert "反向进入目标或非统计方向目标混入" in diagnosis["secondary_causes"]
    assert diagnosis["suspect_windows"]
    print("OK diagnosis_agent")


def test_unified_agent():
    """Test that a single PigCountingAgent instance can do all three jobs."""
    from pig_counting_agent import PigCountingAgent

    agent = PigCountingAgent(
        registry_path=ROOT / "review_registry.json",
        tracker_name="ByteTrack",
        log_dir=None,
        stale_frame_seconds=0.01,
        reconnect_failure_threshold=2,
    )

    # Review capability
    result = agent.assess("17-37头.mp4", 37, 32)
    assert result["paper_actual"] == 31
    assert result["paper_error"] == 1

    # Ops monitoring capability
    agent.note_stream_started("rtsp://test")
    agent.note_frame(frame_idx=1, infer_fps=10.0, total_count=5, valid_traj=5, total_ids=5)
    assert agent.snapshot()["status"] == "HEALTHY"
    agent.note_waiting_for_frame(wait_seconds=0.02, failure_streak=2)
    assert agent.snapshot()["status"] == "RECOVERING"
    assert agent.consume_actions()["reconnect"] is True

    # Diagnosis capability (if data available)
    out_dir = ROOT.parent.parent / "output" / "batch_rerun_group4_review" / "17-37头"
    if out_dir.exists():
        diagnosis = agent.analyze(out_dir, "17-37头.mp4", actual=37, paper_actual=31,
                                  total_line=32, valid_traj=30)
        assert diagnosis["primary_cause"] == "标签定义与统计方向不一致"

    print("OK unified_agent")


def main():
    test_review_agent()
    test_diagnosis_agent()
    test_autonomous_agent()
    test_unified_agent()
    test_manual_review_store()
    test_generated_review_csv()
    print("ALL AGENT SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
