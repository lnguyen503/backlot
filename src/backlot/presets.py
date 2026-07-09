"""Kontext edit presets — one-click Studio actions (OpenArt-style).

Each preset composes a full FLUX-Kontext instruction from a short template + a
user argument; the web/MCP layer runs the result through the `edit_kontext`
workflow. Keeping the wording here (not in the frontend) makes the presets pure,
testable, and reusable by any caller. Add a preset = add a function + registry entry.
"""
from __future__ import annotations

# Shared tail: tell Kontext to preserve the subject so only the intended change lands.
_KEEP_SUBJECT = ("Keep the main subject exactly the same — identical pose, appearance "
                 "and framing.")


def replace_background(arg: str) -> str:
    """Swap the setting behind the subject, subject untouched."""
    bg = arg.strip() or "a clean neutral studio backdrop"
    return (f"Replace the background with {bg}. {_KEEP_SUBJECT} Blend the subject "
            "naturally into the new setting with matching light, colour and shadow.")


# Named camera angles -> a Kontext-friendly phrase; unknown args pass through verbatim.
_ANGLES = {
    "front": "a straight-on front view",
    "left": "a three-quarter view from the left",
    "right": "a three-quarter view from the right",
    "low": "a low camera angle looking up",
    "high": "a high camera angle looking down",
    "profile": "a side profile view",
    "behind": "a view from directly behind",
    "closeup": "a tight close-up",
    "wide": "a wide establishing shot",
}


def camera_angle(arg: str) -> str:
    """Re-frame the same subject from a different camera viewpoint."""
    angle = _ANGLES.get(arg.strip().lower(), arg.strip() or "a straight-on front view")
    return (f"Show the same subject from {angle}. Keep the same character identity, "
            "outfit, colours and art style; change only the camera viewpoint, not the subject.")


def relight(arg: str) -> str:
    """Change only the lighting/mood of a scene, subject and content untouched."""
    light = arg.strip() or "soft cinematic key light from the left, warm golden-hour glow"
    return (f"Relight this image: {light}. Keep the exact same subject, pose, composition and "
            "content; change only the lighting, shadows, highlights and colour temperature.")


PRESETS = {"replace_background": replace_background, "camera_angle": camera_angle,
           "relight": relight}


# Multi-view / turnaround: rotate the SAME subject to a set of angles (Kontext). Not
# 3D-consistent like SV3D, but zero-model and great for character/product view sheets.
_TURN_LOCK = ("Keep the exact same subject — identical identity, colours, materials, "
              "proportions and art style; plain consistent neutral background, centered.")
_TURN_ANGLES = {
    4: ["front", "right side profile", "back", "left side profile"],
    6: ["front", "front three-quarter right", "right side profile",
        "back", "left side profile", "front three-quarter left"],
    9: ["front", "front three-quarter right", "right side profile",
        "back three-quarter right", "back", "back three-quarter left",
        "left side profile", "front three-quarter left", "top-down high angle"],
}


def turnaround_views(n: int = 4) -> list[dict]:
    """N rotated-view Kontext instructions for a turnaround set: [{label, instruction}]."""
    angles = _TURN_ANGLES.get(n, _TURN_ANGLES[4])
    return [{"label": a, "instruction":
             f"Show this exact same subject from a {a} view, rotated in place. {_TURN_LOCK}"}
            for a in angles]


def instruction(name: str, arg: str = "") -> str:
    """Compose the Kontext instruction for preset `name` with user `arg`."""
    fn = PRESETS.get(name)
    if fn is None:
        raise KeyError(f"unknown preset: {name}")
    return fn(arg)
