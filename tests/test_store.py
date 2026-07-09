from backlot.web.store import RunStore


def test_save_and_list_newest_first(tmp_path):
    s = RunStore(str(tmp_path))
    s.save("r1", "txt2img_flux", {"positive_prompt": "x"},
           {"state": "completed", "outputs": [
               {"type": "image", "filename": "a.png", "url": "u", "node_id": "9", "subfolder": ""}]},
           100.0)
    s.save("r2", "txt2img_flux", {"positive_prompt": "y"},
           {"state": "completed", "outputs": []}, 200.0)
    items = s.list()
    assert len(items) == 2
    assert items[0]["run_id"] == "r2"           # newest first
    assert items[1]["outputs"][0]["filename"] == "a.png"


def test_session_filter(tmp_path):
    s = RunStore(str(tmp_path))
    s.save("r1", "txt2img_flux", {}, {"state": "completed", "outputs": []}, 1.0, session_id="S1")
    s.save("r2", "txt2img_flux", {}, {"state": "completed", "outputs": []}, 2.0, session_id="S2")
    s.save("r3", "txt2img_flux", {}, {"state": "completed", "outputs": []}, 3.0)
    assert {m["run_id"] for m in s.list(session_id="S1")} == {"r1"}
    assert len(s.list()) == 3


def test_get_and_missing(tmp_path):
    s = RunStore(str(tmp_path))
    s.save("r1", "txt2img_sdxl", {}, {"state": "completed"}, 1.0)
    assert s.get("r1")["workflow"] == "txt2img_sdxl"
    assert s.get("nope") is None
