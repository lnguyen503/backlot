"""talking-qc pose checks: thresholds calibrated on six human-approved
talking segments (natural per-frame pose delta peaks at 7.7 deg @25fps;
a hard splice of two takes measures 10.7)."""
import numpy as np

from backlot.qc.talking_qc import DEAD_DEG, SNAP_DEG_PER_FRAME, pose_findings


def _rules(findings):
    return {f["rule"] for f in findings}


def _pose(yaw):
    return np.array([yaw, 0.0, 0.0])


def test_natural_motion_passes():
    # gentle wander well past DEAD_DEG, every per-frame delta <= 7.7 (approved max)
    series = [_pose(7.0 * np.sin(i / 6.0)) for i in range(60)]
    assert _rules(pose_findings(series, 25.0)) == set()


def test_single_frame_snap_flagged():
    series = [_pose(0.0)] * 10 + [_pose(SNAP_DEG_PER_FRAME + 1.5)] * 10
    assert "pose-snap" in _rules(pose_findings(series, 25.0))


def test_fast_natural_turn_not_a_snap():
    # 25 deg turn spread over ~6 frames: the false-positive class
    series = [_pose(min(i * 4.5, 25.0)) for i in range(20)]
    assert "pose-snap" not in _rules(pose_findings(series, 25.0))


def test_snap_threshold_scales_with_fps():
    # the same 11.5 deg one-frame jump is natural at 12fps (2x the time passes)
    series = [_pose(0.0)] * 10 + [_pose(SNAP_DEG_PER_FRAME + 1.5)] * 10
    assert "pose-snap" not in _rules(pose_findings(series, 12.0))


def test_dead_take_warns():
    series = [_pose(DEAD_DEG / 4)] * 40
    f = pose_findings(series, 25.0)
    dead = [x for x in f if x["rule"] == "dead-take"]
    assert dead and dead[0]["severity"] == "warn"


def test_lost_face_is_error():
    series = [_pose(0.0)] * 5 + [None] * 3 + [_pose(1.0)] * 5
    f = pose_findings(series, 25.0)
    lost = [x for x in f if x["rule"] == "face-lost"]
    assert lost and lost[0]["severity"] == "error"
    assert "3/13" in lost[0]["message"]
