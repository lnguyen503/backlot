"""Unit tests for the storyboard layer (pure logic — no Ollama/ComfyUI needed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.storyboard.agent import _to_board
from backlot.storyboard.models import AssetCard, Panel, Storyboard
from backlot.storyboard.store import StoryboardStore

_RAW = {
    "title": "Dawn Watch",
    "logline": "A keeper and a gull.",
    "style_notes": "cinematic 35mm",
    "aspect": "landscape",
    "assets": [
        {"bucket": "style", "name": "House Style", "description": "warm", "prompt": "warm style"},
        {"bucket": "character", "name": "Silas", "description": "old keeper", "prompt": "full body"},
        {"bucket": "object", "name": "The Gull", "description": "a gull", "prompt": "a gull"},
    ],
    "panels": [
        {"scene": "s1", "image_prompt": "p1", "shot": "wide", "camera": "static",
         "mood": "calm", "motion_prompt": "m1", "assets": ["Silas", "The Gull"]},
        {"scene": "s2", "image_prompt": "p2", "shot": "close", "camera": "push",
         "mood": "warm", "motion_prompt": "m2", "assets": ["silas"]},  # case-insensitive
    ],
}


def test_to_board_resolves_asset_names_to_ids():
    sb = _to_board(_RAW, idea="an idea")
    assert sb.title == "Dawn Watch"
    assert len(sb.assets) == 3 and len(sb.panels) == 2
    silas = next(c for c in sb.assets if c.name == "Silas")
    # panel 1 references Silas + The Gull; panel 2 references 'silas' (lowercased)
    assert silas.id in sb.panels[0].asset_ids
    assert sb.panels[1].asset_ids == [silas.id]
    assert sb.cards_for(sb.panels[0])[0].name in {"Silas", "The Gull"}


def test_primary_character_needs_a_ref():
    sb = _to_board(_RAW, idea="x")
    p = sb.panels[0]
    assert sb.primary_character(p) is None          # no ref yet
    sb.card(p.asset_ids[0])  # ensure lookup works
    char = next(c for c in sb.assets if c.bucket == "character")
    char.ref = __import__("backlot.storyboard.models", fromlist=["Asset"]).Asset(
        filename="ref.png")
    assert sb.primary_character(p) is char           # now anchorable


def test_store_roundtrip(tmp_path):
    store = StoryboardStore(str(tmp_path))
    sb = _to_board(_RAW, idea="x")
    store.save(sb)
    again = store.get(sb.id)
    assert again is not None and again.title == sb.title
    assert len(again.panels) == 2
    listing = store.list()
    assert listing and listing[0]["id"] == sb.id and listing[0]["panels"] == 2
    assert store.delete(sb.id) is True
    assert store.get(sb.id) is None


def test_models_defaults():
    sb = Storyboard()
    assert sb.aspect == "landscape" and sb.panels == [] and sb.assets == []
    assert sb.score is None and sb.narration is None and sb.ambient is None
    c = AssetCard(bucket="character", name="X")
    assert c.id.startswith("card_") and c.ref is None
    p = Panel()
    assert p.id.startswith("panel_") and p.source == "generated"


def test_audio_legs_ducking_and_order():
    """Voice full; music ducks under voice; ambient/SFX always sits low."""
    from backlot.storyboard.models import Asset
    from backlot.storyboard.render import _audio_legs
    a = Asset(type="audio", filename="x.wav")
    sb = Storyboard(narration=a, score=a, ambient=a)
    # with voice present: [narration 1.0, music 0.30, ambient 0.15], in that order
    legs = _audio_legs(sb, has_voice=True, music_vol=0.30, ambient_vol=0.15)
    assert [v for _, v in legs] == [1.0, 0.30, 0.15]
    assert [asset for asset, _ in legs] == [sb.narration, sb.score, sb.ambient]
    # no voice: music leads at full volume, ambient stays low as a texture
    sb2 = Storyboard(score=a, ambient=a)
    legs2 = _audio_legs(sb2, has_voice=False, music_vol=0.30, ambient_vol=0.15)
    assert [v for _, v in legs2] == [1.0, 0.15]
