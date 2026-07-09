"""Headless Blender runner — execute a bpy script in a clean Blender process.

The base primitive of the Blender bridge: hand Blender a Python (bpy) script, it
runs `blender --background --python <script>`, and you get back stdout/stderr/rc.
Because each call is its own process the scene state is not persistent across calls
(fine for build->render tasks; a socket server handles interactive iteration).

Blender exe resolution: $BLENDER_EXE, else the newest install under
"C:/Program Files/Blender Foundation/*/blender.exe".
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


def find_blender() -> str:
    env = os.environ.get("BLENDER_EXE")
    if env and Path(env).exists():
        return env
    hits = sorted(glob.glob(r"C:/Program Files/Blender Foundation/*/blender.exe"), reverse=True)
    if hits:
        return hits[0]
    # Linux/mac fallbacks
    for cand in ("/usr/bin/blender", "/usr/local/bin/blender",
                 "/Applications/Blender.app/Contents/MacOS/Blender"):
        if Path(cand).exists():
            return cand
    raise RuntimeError("blender executable not found; set BLENDER_EXE")


@dataclass
class BlenderResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        # Blender background can exit 0 even on an uncaught script exception, so
        # also treat a Python traceback in stderr as failure.
        return self.returncode == 0 and "Traceback (most recent call last)" not in self.stderr

    def tagged(self, tag: str) -> list[str]:
        """Lines the script emitted as `print(f'{tag} ...')` — a simple result channel."""
        return [ln[len(tag):].strip() for ln in self.stdout.splitlines()
                if ln.strip().startswith(tag)]


def run_script(script: str, *, args: Optional[Sequence[str]] = None,
               blend: Optional[str] = None, factory_startup: bool = True,
               timeout: float = 600) -> BlenderResult:
    """Run a bpy script headless. `args` arrive in the script's sys.argv after '--'.

    `blend` opens an existing .blend first; `factory_startup` ignores user addons/
    prefs for a reproducible environment (set False if you need installed addons).
    """
    exe = find_blender()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(script)
        tmp.close()
        cmd = [exe, "--background"]
        if factory_startup:
            cmd.append("--factory-startup")
        if blend:
            cmd.append(str(blend))
        cmd += ["--python", tmp.name, "--python-exit-code", "1"]
        if args:
            cmd += ["--", *map(str, args)]
        # stdin=DEVNULL: Blender must not inherit the host's stdin — when the host
        # is an MCP stdio server, the inherited protocol pipe blocks Blender at
        # startup (observed: 0 CPU, no frames, forever).
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        return BlenderResult(proc.returncode, proc.stdout, proc.stderr)
    finally:
        os.unlink(tmp.name)


def run_script_file(path: str, **kw) -> BlenderResult:
    return run_script(Path(path).read_text(encoding="utf-8"), **kw)


def version() -> str:
    r = run_script("import bpy; print('VER', bpy.app.version_string)", timeout=120)
    vs = r.tagged("VER")
    return vs[0] if vs else ""
