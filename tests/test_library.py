"""Unit tests for the global Characters & Worlds library store."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.storyboard.library import LibraryStore
from backlot.storyboard.models import Asset, AssetCard


def test_library_roundtrip_and_bucket_filter(tmp_path):
    lib = LibraryStore(str(tmp_path))
    pip = AssetCard(bucket="character", name="PIP", prompt="a robot",
                    voice="am_adam", ref=Asset(filename="pip.png"))
    world = AssetCard(bucket="environment", name="Rooftop", prompt="a garden")
    lib.save(pip)
    lib.save(world)

    got = lib.get(pip.id)
    assert got is not None and got.name == "PIP" and got.voice == "am_adam"
    assert got.ref is not None and got.ref.filename == "pip.png"

    assert len(lib.list()) == 2
    chars = lib.list(bucket="character")
    assert len(chars) == 1 and chars[0]["name"] == "PIP"

    assert lib.delete(pip.id) is True
    assert lib.get(pip.id) is None
    assert len(lib.list()) == 1


def test_library_upsert_same_id(tmp_path):
    lib = LibraryStore(str(tmp_path))
    c = AssetCard(bucket="character", name="A")
    lib.save(c)
    c.name = "A-renamed"
    lib.save(c)                       # same id -> upsert, not a duplicate
    assert len(lib.list()) == 1
    assert lib.get(c.id).name == "A-renamed"
