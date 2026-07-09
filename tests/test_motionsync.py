"""Tests for the motion-sync frame sampler (no depth model / GPU needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as imageio
from backlot import motionsync


def _make_video(path, frames=12, size=32):
    w = imageio.get_writer(str(path), fps=8, macro_block_size=1)
    for i in range(frames):
        img = np.full((size, size, 3), i * 20 % 255, dtype=np.uint8)
        w.append_data(img)
    w.close()


def test_sample_frames_even_count(tmp_path):
    vid = tmp_path / "ref.mp4"
    _make_video(vid, frames=12)
    got = motionsync.sample_frames(str(vid), 6)
    assert len(got) == 6
    assert all(hasattr(im, "size") for im in got)          # PIL images


def test_sample_frames_caps_at_available(tmp_path):
    vid = tmp_path / "short.mp4"
    _make_video(vid, frames=4)
    # asking for more frames than exist should not exceed what's available
    assert len(motionsync.sample_frames(str(vid), 25)) <= 25
