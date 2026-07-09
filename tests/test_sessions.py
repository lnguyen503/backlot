from backlot.web.sessions import SessionStore


def test_crud_and_order(tmp_path):
    s = SessionStore(str(tmp_path))
    assert s.list() == []
    a = s.create("Project A", 100.0)
    b = s.create("Project B", 200.0)
    assert a["id"] != b["id"]
    assert [x["name"] for x in s.list()] == ["Project B", "Project A"]  # newest first
    assert s.rename(a["id"], "Renamed")["name"] == "Renamed"
    assert s.rename("nope", "x") is None
    assert s.delete(b["id"]) is True
    assert s.delete("nope") is False
    assert len(s.list()) == 1 and s.list()[0]["name"] == "Renamed"
