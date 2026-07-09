"""VLM judge for QC stages - local Ollama vision model (qwen2.5vl), temperature 0,
JSON-schema-constrained output so verdicts are machine-readable.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5vl:32b"

# One schema for every visual judgment: a 0-10 match score + typed issues.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_match": {"type": "integer", "minimum": 0, "maximum": 10},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string",
                             "enum": ["spatial", "physics", "clones", "text",
                                      "anatomy", "style", "contamination", "other"]},
                    "detail": {"type": "string"},
                    "severity": {"type": "string", "enum": ["error", "warn"]},
                },
                "required": ["type", "detail", "severity"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["scene_match", "issues", "summary"],
}


def judge(image_paths: list[Path], prompt: str, model: str = MODEL,
          timeout: float = 300.0) -> dict:
    """Send frame(s) + an instruction; get a schema-valid verdict dict."""
    images = [base64.b64encode(p.read_bytes()).decode() for p in image_paths]
    r = httpx.post(f"{OLLAMA}/api/chat", json={
        "model": model, "stream": False, "format": VERDICT_SCHEMA,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt, "images": images}],
    }, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def available(model: str = MODEL) -> bool:
    try:
        tags = httpx.get(f"{OLLAMA}/api/tags", timeout=5).json()["models"]
        return any(m["name"] == model for m in tags)
    except Exception:
        return False


def orientation_prompt(intended: str) -> str:
    """Who-sees-what check for held/viewed objects (the 009 backwards-Polaroid class,
    recurred in 001R s12): the object's printed/display face must be visible to the
    HOLDER, not the camera."""
    return (
        "You are checking OBJECT ORIENTATION in a film still. A character is holding "
        "or looking at a flat object. Definitions: the object's FRONT is the side "
        "with the picture, image, or text; a plain cardboard, paper, or wooden side "
        "is its BACK.\n"
        "Physical rule: if the character is viewing the object, the FRONT must be "
        "turned toward the CHARACTER's eyes, which means the camera sees the BACK or "
        "the edge. Camera-sees-the-back is CORRECT viewing physics, not an error.\n"
        "Score scene_match 10 when the camera sees the object's BACK or edge while "
        "the character looks at it (correct), or when the intended shot explicitly "
        "wants the front shown to the camera. Score 0-5 ONLY when the picture/text "
        "side faces the CAMERA while the character is simultaneously depicted "
        "looking at that object. Report that case as an issue of type 'spatial' with "
        "severity 'error', quoting which side of the object the camera sees.\n"
        f"\nINTENDED SHOT: {intended}\n"
        "Only report what you actually see."
    )


def crowd_prompt(intended: str, style: str) -> str:
    """Crowd-context check (001R un-soccer fans, seq 48 item 3): background people
    must read as attendees of THIS event in THIS season/mood, not generic extras.
    Garbage-in survives any model tier - context drift is a still-level defect."""
    return (
        "You are checking CROWD CONTEXT in a film still. Background/secondary "
        "people must read as attendees of the specific event and mood described "
        "in the INTENDED SHOT - their clothing, props, and bearing should say "
        "WHERE they just came from and WHAT kind of night this is.\n"
        "Score scene_match 10 when the people clearly read as that event's "
        "attendees (e.g. sports fans in team colors/scarves after a match); score "
        "0-5 when they read as generic extras who could be anywhere. Report "
        "context drift as an issue of type 'contamination' with severity 'error', "
        "naming what the people actually read as.\n"
        f"\nINTENDED SHOT: {intended}\nEPISODE LOOK: {style}\n"
        "Only report what you actually see."
    )


def still_prompt(intended: str, style: str) -> str:
    return (
        "You are a film-still QC inspector on an AI-video pipeline. Judge the image "
        "against the INTENDED SHOT below. Score scene_match 0-10 (10 = shows exactly "
        "this shot; below 6 = wrong scene). Report issues:\n"
        "- spatial: people/objects placed illogically (someone inside a structure who "
        "should be outside, impossible viewpoints)\n"
        "- physics: floating/disconnected objects, levitating vehicles, impossible "
        "poses or object states\n"
        "- clones: two or more people in frame with the SAME face\n"
        "- text: garbled, misspelled, or nonsense writing on signs/labels (quote it)\n"
        "- anatomy: extra/missing/deformed limbs, hands, or faces\n"
        "- style: image breaks the episode's look\n"
        "- contamination: content that clearly belongs to a DIFFERENT story "
        "(unrelated food truck, wrong setting, wrong era)\n"
        f"\nINTENDED SHOT: {intended}\nEPISODE LOOK: {style}\n"
        "Only report what you actually see. An empty issues list is a valid answer."
    )
