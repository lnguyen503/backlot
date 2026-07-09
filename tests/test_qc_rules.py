"""beats-lint rules: each test pins a defect class we paid for on eps 001-004."""
from backlot.qc.rules import check_motion, check_still, lint_beats


def _rules(findings):
    return {f["rule"] for f in findings}


def test_crowd_clones_flagged():
    assert "crowd-clones" in _rules(check_still("x", "a crowd of fans outside a bar"))


def test_crowd_as_texture_ok():
    assert "crowd-clones" not in _rules(
        check_still("x", "distant crowd, blurred, from behind, bokeh"))


def test_incidental_text_needs_quote():
    assert "incidental-text" in _rules(check_still("x", "an old bakery storefront"))
    assert "incidental-text" not in _rules(
        check_still("x", "a storefront with a sign 'BREAD'"))


def test_long_quoted_text():
    assert "long-text" in _rules(
        check_still("x", "a banner reading 'WE REBUILD KITCHENS TOO'"))


def test_contraction_inside_signage_stays_one_quote():
    f = check_still("x", "chalkboard reading 'WHAT THE RIVER COULDN'T TAKE'")
    long = [x for x in f if x["rule"] == "long-text"]
    assert long and "COULDN'T TAKE" in long[0]["message"]


def test_articulated_motion():
    assert "articulated-motion" in _rules(
        check_still("x", "a goalkeeper diving for the ball"))


def test_handoff_needs_camera_position():
    assert "handoff-staging" in _rules(
        check_motion("x", "she is passing the box to him"))
    assert "handoff-staging" not in _rules(
        check_motion("x", "over her shoulder from inside, passing the box"))


def test_multi_action_loops():
    assert "multi-action" in _rules(
        check_motion("x", "she opens the jar, places it on the shelf and takes another"))


def test_low_motion_freeze_risk():
    assert "low-motion" in _rules(check_motion("x", "a quiet shelf of books"))
    assert "low-motion" not in _rules(
        check_motion("x", "steam rises, slow push-in"))


def test_spatial_logic():
    assert "spatial-logic" in _rules(check_still(
        "x", "a vendor and a customer at a food truck at night"))
    assert "spatial-logic" not in _rules(check_still(
        "x", "a vendor inside the food truck, a customer outside at the window"))


def test_score_noise_and_outro_line():
    b = {"beats": [], "inserts": [],
         "score_tags": "gentle rain, soft piano",
         "host_outro": [{"text": "Good night."}]}
    rules = _rules(lint_beats(b))
    assert "score-noise" in rules and "outro-line" in rules


def test_vo_length_without_insert():
    b = {"beats": [{"i": 1, "vo": " ".join(["word"] * 50),
                    "still": "a hillside", "motion": "slow push-in"}],
         "inserts": []}
    assert any(f["rule"] == "vo-length" and f["severity"] == "error"
               for f in lint_beats(b))


def test_mojibake():
    b = {"beats": [], "inserts": [], "cold_open": {"vo": "the river â€” rising"}}
    assert "mojibake" in _rules(lint_beats(b))


def test_clean_beat_passes():
    b = {"beats": [{"i": 1, "vo": "Dawn over the valley.",
                    "still": "a misty Appalachian valley at dawn, cinematic",
                    "motion": "mist drifts, slow push-in"}],
         "inserts": [],
         "score_tags": "sparse acoustic guitar, instrumental, no drums",
         "host_outro": [{"text": "I'll read every comment."}]}
    assert lint_beats(b) == []
