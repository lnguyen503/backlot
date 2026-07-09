"""Storyboard data model — the editable, ordered board and its consistency cards.

`AssetCard` is the consistency primitive (TODO item 2): a character / environment /
object / style defined ONCE, reused across every panel so identity, wardrobe,
style and setting stay locked. A `Panel` references cards by id and carries its
own frames-first still and (later) its motion clip.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

Bucket = Literal["style", "character", "environment", "object"]
BUCKETS: tuple[Bucket, ...] = ("style", "character", "environment", "object")
Aspect = Literal["portrait", "landscape", "square"]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Asset(BaseModel):
    """A rendered output (image or video) — mirrors the engine's asset shape."""
    type: str = "image"
    filename: str
    subfolder: str = ""
    url: str = ""


class AssetCard(BaseModel):
    id: str = Field(default_factory=lambda: _id("card"))
    bucket: Bucket
    name: str
    description: str = ""
    # image prompt used to GENERATE this card's reference still (characters/objects/envs)
    prompt: str = ""
    ref: Optional[Asset] = None           # the locked reference image
    source: Literal["generated", "photo"] = "generated"
    seed: int = -1
    voice: str = ""                       # Kokoro voice id for a character (talk render)


class Panel(BaseModel):
    id: str = Field(default_factory=lambda: _id("panel"))
    scene: str = ""                        # what happens in this beat
    image_prompt: str = ""                 # full prompt for the still
    shot: str = ""                         # e.g. "wide establishing", "close-up"
    camera: str = ""                       # e.g. "slow push-in", "static"
    mood: str = ""                         # e.g. "tense", "hopeful"
    motion_prompt: str = ""                # action for img2vid
    dialogue: str = ""                     # spoken line for a talking (InfiniteTalk) panel
    asset_ids: list[str] = Field(default_factory=list)   # cards used in this panel
    still: Optional[Asset] = None          # frames-first preview
    line_audio: Optional[Asset] = None     # TTS of `dialogue` (drives the talking clip)
    video: Optional[Asset] = None          # animated clip
    source: Literal["generated", "photo"] = "generated"
    seed: int = -1


class Storyboard(BaseModel):
    id: str = Field(default_factory=lambda: _id("sb"))
    title: str = "Untitled Storyboard"
    idea: str = ""
    logline: str = ""
    style_notes: str = ""
    aspect: Aspect = "landscape"
    assets: list[AssetCard] = Field(default_factory=list)
    panels: list[Panel] = Field(default_factory=list)
    chat: list[dict] = Field(default_factory=list)   # [{role, content}, ...]
    score: Optional[Asset] = None                    # generated music track (audio)
    narration: Optional[Asset] = None                # generated voiceover track (audio)
    ambient: Optional[Asset] = None                  # generated ambient/SFX bed (audio)
    assembled: Optional[Asset] = None                # final stitched sequence (muxed if scored)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # --- lookups -------------------------------------------------------------
    def card(self, card_id: str) -> Optional[AssetCard]:
        return next((c for c in self.assets if c.id == card_id), None)

    def panel(self, panel_id: str) -> Optional[Panel]:
        return next((p for p in self.panels if p.id == panel_id), None)

    def cards_for(self, panel: Panel) -> list[AssetCard]:
        return [c for c in (self.card(i) for i in panel.asset_ids) if c is not None]

    def primary_character(self, panel: Panel) -> Optional[AssetCard]:
        """The first character card on a panel that has a locked reference —
        the identity to anchor the panel to for cross-panel consistency."""
        for c in self.cards_for(panel):
            if c.bucket == "character" and c.ref is not None:
                return c
        return None
