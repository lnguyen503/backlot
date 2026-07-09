"""Blender integration: drive Blender (bpy) from the engine/agent.

Foundation for the TODO "3D / Blender integration" lane — Blender supplies
geometry/motion/cameras, ComfyUI supplies the AI look. Start with a headless
`bpy` runner (one Blender process per task, clean + robust); a persistent socket
server can layer on top for interactive use.
"""
