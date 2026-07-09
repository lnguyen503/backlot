"""Web app smoke test: importing builds the app and registers the routes.

Avoids triggering the lifespan (which would need ComfyUI), so this stays a unit test.
"""
from backlot.web.app import GenBody, app


def test_routes_registered():
    paths = {getattr(r, "path", None) for r in app.routes}
    for p in ["/api/workflows", "/api/workflows/{name}", "/api/generate",
              "/api/jobs/{run_id}", "/api/stream/{run_id}", "/api/assets", "/",
              "/api/sessions", "/api/sessions/{sid}", "/api/edit", "/api/animate"]:
        assert p in paths


def test_genbody_defaults():
    b = GenBody(name="txt2img_flux")
    assert b.name == "txt2img_flux" and b.params == {}
