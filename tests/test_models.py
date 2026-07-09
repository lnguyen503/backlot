from backlot.engine.models import (
    Asset, Capability, InjectSpec, JobState, JobStatus, ParamType, Progress,
)


def test_asset_dedupe_key():
    a = Asset(type="image", filename="a.png", subfolder="s", url="u", node_id="9")
    assert a.dedupe_key() == ("9", "a.png", "s")


def test_progress_percent():
    assert Progress(value=5, max=20).percent() == 25.0
    assert Progress(value=0, max=0).percent() == 0.0


def test_jobstatus_public_dict():
    js = JobStatus(run_id="r", state=JobState.RUNNING)
    d = js.public_dict()
    assert d["state"] == "running"
    assert d["progress"]["percent"] == 0.0
    assert d["outputs"] == []


def test_capability_params_and_info():
    cap = Capability(
        name="c", title="C", kind="image", template_path="t",
        inject=[InjectSpec(name="p", api=["1", "inputs", "x"],
                           type=ParamType.STRING, required=True)],
    )
    param = cap.params()[0]
    assert param.name == "p" and param.required
    assert cap.public_info()["name"] == "c"
