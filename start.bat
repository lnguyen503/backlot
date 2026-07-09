@echo off
REM Backlot Studio launcher. Double-click to start, then open http://127.0.0.1:8765
REM Requires ComfyUI running on 127.0.0.1:8188.
cd /d "%~dp0"
echo Starting Backlot Studio on http://127.0.0.1:8765 ...
.venv\Scripts\python.exe -m backlot.web.app
pause
