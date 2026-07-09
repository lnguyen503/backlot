"""beats.json lint rules - every rule below is a defect class that AI video
models reliably produce (learned from real episode re-rolls).
Pure functions over the loaded beats dict; no I/O, no models. When a shot keeps
failing, change the SHOT, not the seed - these rules change the shot pre-render.

A finding: {"rule", "severity" ("error"|"warn"), "loc", "message", "fix"}.
"""
from __future__ import annotations

import re

# --- word lists (lowercase; matched on word boundaries) -------------------------

CROWD = ["crowd", "crowds", "fans", "villagers", "onlookers", "bystanders",
         "neighbors", "neighbours", "congregation", "mob", "audience",
         "protesters", "group of people", "people"]
CROWD_SAFE = ["silhouette", "in the distance", "blurred", "out of focus", "bokeh",
              "from behind", "backs of", "texture", "far below", "distant"]

TEXT_TRIGGERS = ["sign", "signage", "banner", "poster", "label", "labeled",
                 "labelled", "headline", "newspaper", "chalkboard", "menu",
                 "lettering", "marquee", "billboard", "storefront", "graffiti",
                 "inscription", "placard", "text"]

FLOATERS = ["flag", "flags", "confetti", "balloon", "balloons", "streamers",
            "bunting", "slipping", "fluttering papers", "flying papers"]

ARTICULATED = ["diving", "dive", "dancing", "dances", "fighting", "fight",
               "jumping", "leaps", "leaping", "sprinting", "kicking", "throwing",
               "catching", "climbing", "swimming", "backflip", "wrestling",
               "athletes", "playing soccer", "playing football"]

HANDOFF = ["passing", "passes", "hands over", "handing", "handoff", "hand off",
           "exchange", "exchanging", "gives", "giving", "receives", "receiving"]
FINE_HANDS = ["typing", "writing", "tying", "buttoning", "stirring", "chopping",
              "knitting", "threading", "sewing", "shuffling cards", "counting",
              "flipping pages", "turning pages"]
CAMERA_POS = ["over her shoulder", "over his shoulder", "over the shoulder",
              "from inside", "from outside", "through the window", "through the",
              "behind the", "pov", "close-up", "extreme close-up", "from above",
              "from below", "framed"]

PERSON = ["man", "woman", "boy", "girl", "child", "children", "kid", "customer",
          "worker", "vendor", "librarian", "baker", "farmer", "fisherman",
          "teacher", "nurse", "firefighter", "driver", "owner", "keeper",
          "grandmother", "grandfather", "mother", "father", "couple"]
STRUCTURE = ["truck", "counter", "window", "doorway", "door", "room", "shop",
             "kitchen", "booth", "stall", "porch", "church", "library", "store"]
SPATIAL = ["outside", "inside", "behind", "through", "beside", "across from",
           "in front of", "at the window", "on the steps", "over the", "next to"]

VEHICLES = ["car", "cars", "truck", "trucks", "van", "bus", "tractor",
            "pickup", "jeep", "motorcycle"]
VEHICLE_PIN = ["stays perfectly still", "stationary", "parked and still",
               "does not move", "remains still"]

WIDE = ["wide shot", "wide view", "aerial", "establishing", "panorama",
        "panoramic", "high angle view of the town", "street scene"]
GRAY_LIGHT = ["gray", "grey", "overcast", "rain-gray", "rain-grey", "drizzle",
              "dull light", "flat light"]

CAMERA_MOVE = ["push-in", "push in", "pull-back", "pull back", "drift", "orbit",
               "pan", "zoom", "dolly", "tilt", "tracking", "handheld", "crane",
               "glide", "sweep"]
DYNAMIC = ["rain", "streaks", "pulses", "drips", "flickers", "flicker", "rises",
           "falls", "drifts", "swirls", "billows", "crackles", "sways",
           "trembles", "breathes", "smoke", "steam", "embers", "dust", "ash",
           "snow", "wind", "waves", "ripples", "glows brighter", "passes",
           "walks", "runs", "pours", "spins", "turns", "opens", "closes",
           "hammers", "climbs", "rustles", "flows"]
STATIC_WORDS = ["stillness", "motionless", "static", "frozen"]

# trap-word lexicon: physics asks the current model tier cannot render -
# every generation re-mints the artifact (water in glasses, weightless paper
# plates, "blazing" fire). Blocked at the read gate.
LIQUID_PHYSICS = ["ripples", "ripple", "rippling", "sloshes", "slosh", "sloshing",
                  "pours", "pouring", "splashes", "splashing", "spills", "spilling"]
LIGHTWEIGHT_OBJ = ["paper plate", "paper plates", "napkin", "napkins", "paper cup",
                   "paper cups", "playing card", "playing cards", "streamer",
                   "streamers", "confetti piece", "loose paper"]
MICRO_MOTION = ["stir", "stirs", "stirring", "flutter", "flutters", "fluttering",
                "tremble", "trembles", "trembling", "rustle", "rustles", "rustling",
                "shiver", "shivers", "shivering"]
BLAZE_WORDS = ["blazing", "ablaze", "inferno", "engulfed in light", "engulfed"]
HAND_WORDS = ["hand", "hands", "finger", "fingers", "fingertip", "fingertips"]
FLAME_WORDS = ["candle", "candles", "flame", "flames", "lit wick", "wicks"]
# aquarium-jar class: glass vessels attract liquid physics -
# FLUX fills an unspecified jar with water/bubbles/floating debris
JAR_VESSEL = ["jar", "jars", "fishbowl", "glass bowl", "glass vessel", "carafe",
              "pitcher"]
JAR_DRY_GUARD = ["dry", "empty", "no liquid", "no water"]

SCORE_NOISE = ["rain", "static", "noise", "wind", "storm", "thunder",
               "field recording", "tape", "vinyl", "lo-fi", "lofi", "hiss",
               "crackle", "nature sounds", "water", "birds", "ambient sounds"]

MOJIBAKE = ["â€", "Ã©", "Ã¨", "â", "â€"]

_WS = re.compile(r"\s+")


def _has(text: str, words: list[str]) -> list[str]:
    """Words from the list present in text (word-boundary, case-insensitive)."""
    low = _WS.sub(" ", text.lower())
    return [w for w in words if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low)]


def _quoted(text: str) -> list[str]:
    # quote marks must not touch a letter on the OUTSIDE, so contractions inside
    # signage ("COULDN'T") don't split the match
    single = re.findall(r"(?<![A-Za-z])'(.{2,}?)'(?![A-Za-z])", text)
    double = re.findall(r'"(.{2,}?)"', text)
    return single + double


def _f(rule: str, severity: str, loc: str, message: str, fix: str) -> dict:
    return {"rule": rule, "severity": severity, "loc": loc,
            "message": message, "fix": fix}


# --- per-prompt rules ------------------------------------------------------------

def check_still(loc: str, still: str) -> list[dict]:
    out = []
    hits = _has(still, CROWD)
    if hits and not _has(still, CROWD_SAFE):
        out.append(_f("crowd-clones", "error", loc,
                      f"group words {hits} render the same face N times",
                      "specify each person distinctly (age/build/hair/one detail) "
                      "or keep the crowd as texture (distant/blurred/from behind)"))
    quoted = _quoted(still)
    text_hits = _has(still, TEXT_TRIGGERS)
    if text_hits and not quoted:
        out.append(_f("incidental-text", "error", loc,
                      f"text surface {text_hits} with no quoted string - the model "
                      "invents garble to fill it",
                      "either quote the exact SHORT text ('LOSERS EAT FREE') or drop "
                      "the text surface from the prompt"))
    for q in quoted:
        if len(q) > 20:
            out.append(_f("long-text", "error", loc,
                          f"quoted text '{q}' is {len(q)} chars - garbles beyond ~20",
                          "shorten to <=20 chars or split across shots"))
        elif any(c.isdigit() for c in q):
            out.append(_f("digits-in-text", "warn", loc,
                          f"'{q}' contains digits - numbers garble more than letters",
                          "prefer words over digits, or OCR-check the still"))
    hits = _has(still, FLOATERS)
    if hits:
        out.append(_f("floaters", "warn", loc,
                      f"object words {hits} spawn floating artifacts in stills",
                      "omit the object word + add it to the negative prompt "
                      "(the 'flag slipping' class)"))
    persons = _has(still, PERSON)
    if (len(persons) >= 2 or _has(still, CROWD)) and _has(still, STRUCTURE) \
            and not _has(still, SPATIAL):
        out.append(_f("spatial-logic", "error", loc,
                      "multiple people + a structure with no explicit spatial relation "
                      "- models place people where the vibe suggests (fans rendered "
                      "cooking INSIDE the taco truck)",
                      "state relations explicitly: 'OUTSIDE the truck', 'through the "
                      "window', 'behind the counter'"))
    veh = _has(still, VEHICLES)
    if veh:
        out.append(_f("vehicle", "warn", loc,
                      f"vehicle {veh} in a still - parked vehicles fly in i2v",
                      "pin it in the motion prompt ('the parked car stays perfectly "
                      "still') or crop it out of frame"))
    if _has(still, WIDE) and (persons or _has(still, CROWD)) and _has(still, GRAY_LIGHT):
        out.append(_f("gray-wide", "warn", loc,
                      "gray-light wide with people reads as AI slop locally",
                      "recompose closer / people as distant texture, or flag the shot "
                      "for the frontier pass"))
    hits = _has(still, ARTICULATED)
    if hits:
        out.append(_f("articulated-motion", "error", loc,
                      f"complex body motion {hits} breaks in BOTH local and frontier "
                      "models (4 attempts, 2 models, all broken)",
                      "recompose: show the aftermath, the reaction, or partial framing "
                      "(hands, feet, shadows)"))
    return out


def check_trap_words(loc: str, motion: str, still: str = "") -> list[dict]:
    """seq-22 lexicon: unrenderable physics asks. Rerolls re-send the request and
    re-mint the artifact - route these beats to the deterministic lane instead."""
    out = []
    hits = _has(motion, LIQUID_PHYSICS)
    if hits:
        out.append(_f("liquid-physics", "error", loc,
                      f"liquid-physics ask {hits} - this tier mints distortion "
                      "artifacts every generation (the 010 water-glass class)",
                      "cut the liquid motion or route the beat to the deterministic "
                      "Ken Burns lane"))
    jar_hits = _has(still, JAR_VESSEL)
    if jar_hits and not _has(still, JAR_DRY_GUARD):
        out.append(_f("aquarium-jar", "error", loc,
                      f"glass vessel {jar_hits} with unspecified fill state - FLUX "
                      "renders it as an aquarium (water, bubbles, floating debris; "
                      "the 001R tip-jar class)",
                      "state the vessel is DRY glass and name its solid contents "
                      "('dry glass jar, folded bills and coins, no liquid')"))
    if _has(motion, LIGHTWEIGHT_OBJ) and _has(motion, MICRO_MOTION):
        out.append(_f("lightweight-micro-motion", "error", loc,
                      f"micro-motion on lightweight objects "
                      f"({_has(motion, LIGHTWEIGHT_OBJ)} x {_has(motion, MICRO_MOTION)}) "
                      "- the 010 paper-plates class",
                      "keep lightweight props static; put the motion on light or camera"))
    if _has(motion, HAND_WORDS) and _has(motion, FLAME_WORDS):
        out.append(_f("hand-near-flame", "error", loc,
                      "hand-object interaction near flames in the motion prompt - "
                      "reads as finger-snap/merge artifacts",
                      "separate the hand beat from the flame beat, or go deterministic"))
    blaze = _has(motion, BLAZE_WORDS) + _has(still, BLAZE_WORDS)
    if blaze:
        out.append(_f("blaze-light", "error", loc,
                      f"'blazing'-class light word {blaze} renders literal fire "
                      "(the 011 tunnel-inferno class)",
                      "describe the light source and color instead ('brilliant "
                      "cool-white floodlight glare')"))
    return out


def check_motion(loc: str, motion: str, still: str = "") -> list[dict]:
    out = []
    if _has(motion, ARTICULATED):
        out.append(_f("articulated-motion", "error", loc,
                      f"complex body motion {_has(motion, ARTICULATED)} in motion prompt",
                      "recompose to aftermath/reaction/partial framing"))
    handoff = _has(motion, HANDOFF) + _has(still, HANDOFF)
    if handoff and not (_has(motion, CAMERA_POS) or _has(still, CAMERA_POS)):
        out.append(_f("handoff-staging", "error", loc,
                      f"handoff action {handoff} without an explicit camera position",
                      "stage the camera ('over her shoulder from inside') - actor "
                      "descriptions alone put the camera in the wrong place"))
    hits = _has(motion, FINE_HANDS)
    if hits:
        out.append(_f("fine-hands", "warn", loc,
                      f"fine hand action {hits} - fingers break under motion",
                      "prefer the object's motion over the fingers', or extreme "
                      "close-up on the OBJECT"))
    if _has(motion, TEXT_TRIGGERS):
        out.append(_f("text-motion", "warn", loc,
                      "motion prompt references a text surface - i2v degrades "
                      "inherited text further",
                      "keep text surfaces still; never animate signage"))
    # single-action phrasing: i2v loops actions to fill the clip duration
    clauses = [c for c in re.split(r",| and | then ", motion.lower()) if
               any(re.search(rf"(?<![a-z]){re.escape(v)}(?![a-z])", c) for v in
                   HANDOFF + FINE_HANDS + ["opens", "closes", "picks", "places",
                                           "lifts", "sets", "grabs", "takes"])]
    if len(clauses) >= 2:
        out.append(_f("multi-action", "error", loc,
                      f"{len(clauses)} discrete actions in one motion prompt - i2v "
                      "loops actions to fill duration (the jar/plate ping-pong class)",
                      "ONE action per clip + ambient motion; split extra actions "
                      "into their own beats/inserts"))
    if not (_has(motion, CAMERA_MOVE) or _has(motion, DYNAMIC)):
        out.append(_f("low-motion", "warn", loc,
                      "no camera move and no dynamic element - near-static i2v "
                      "renders dead frames / freeze flags",
                      "add real camera motion ('steady lateral drift', 'slow orbit') "
                      "or mark the beat for a Ken Burns zoompan over the still"))
    if _has(motion, STATIC_WORDS) and not _has(motion, DYNAMIC):
        out.append(_f("static-word", "warn", loc,
                      f"{_has(motion, STATIC_WORDS)} with no dynamic element",
                      "give the frame something alive (rain, steam, light shift) "
                      "even in a contemplative beat"))
    return out


# --- whole-file rules -------------------------------------------------------------

def lint_beats(b: dict) -> list[dict]:
    out = []
    items = b.get("beats", []) + b.get("inserts", [])
    for item in items:
        kind = "insert" if item in b.get("inserts", []) else "beat"
        loc = f"{kind} {item.get('i', '?')}"
        still = item.get("still", "")
        if still:
            out += check_still(f"{loc}.still", still)
        if item.get("motion"):
            out += check_motion(f"{loc}.motion", item["motion"], still)
            out += check_trap_words(f"{loc}.motion", item["motion"],
                                    still or item.get("kontext", ""))
        if item.get("kontext"):
            out += check_still(f"{loc}.kontext", item["kontext"])
        vo_words = len(item.get("vo", "").split())
        if vo_words > 45 and "insert" not in item:
            out.append(_f("vo-length", "error", loc,
                          f"{vo_words}-word VO on a single shot - narration outruns "
                          "footage into freeze tails",
                          "cut the VO <=30 words or assign an insert cutaway"))
        elif vo_words > 30 and "insert" not in item:
            out.append(_f("vo-length", "warn", loc,
                          f"{vo_words}-word VO with no insert - stretch risk",
                          "trim the VO or add an insert"))
        # identity: a specific person in a text-to-image still regenerates a NEW person
        if kind == "beat" and still and not item.get("kontext") \
                and _has(still, PERSON):
            out.append(_f("identity-from-text", "warn", loc,
                          "named/specific person in a text-only still - every fresh "
                          "generation invents a new face",
                          "derive character appearances via kontext from the master "
                          "ref; text-only people are for strangers/texture"))
    tags = b.get("score_tags", "")
    hits = _has(tags, SCORE_NOISE)
    if hits:
        out.append(_f("score-noise", "error", "score_tags",
                      f"environment words {hits} - ACE-Step renders them literally "
                      "as noise (the 'rain-gentle' static bug)",
                      "keep tags instrumental/production words only (sparse, no "
                      "drums, clean high-fidelity studio recording)"))
    if tags and "instrumental" not in tags.lower():
        out.append(_f("score-vocals", "warn", "score_tags",
                      "no 'instrumental' tag - ACE-Step may add vocals under VO",
                      "add 'instrumental, no vocals'"))
    outro = " ".join(s.get("text", "") for s in b.get("host_outro", []))
    if outro and "read every comment" not in outro.lower():
        out.append(_f("outro-line", "warn", "host_outro",
                      "locked outro line missing",
                      "end on \"I'll read every comment\" (future tense - the locked "
                      "channel signature)"))
    blob = " ".join(str(v) for v in _all_text(b))
    for seq in MOJIBAKE:
        if seq in blob:
            out.append(_f("mojibake", "error", "file",
                          f"mojibake sequence {seq!r} in text - wrong encoding will "
                          "reach TTS and stills as garbage",
                          "rewrite beats.json as clean UTF-8 (plain ASCII dashes are "
                          "safest for TTS)"))
            break
    return out


def _all_text(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [t for v in node.values() for t in _all_text(v)]
    if isinstance(node, list):
        return [t for v in node for t in _all_text(v)]
    return []
