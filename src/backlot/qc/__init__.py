"""Staged machine-QC for the episode pipeline (workflow redesign, 2026-07-03).

Every stage is analyzed automatically and auto-fixed before advancing so one-off
defects never reach the human. Stages, cheapest first:
  1. beats_lint  — pure-code prompt lint of beats.json (pre-render, $0)
  2. still_qc    — OCR / face-ID / VLM per still (fix here = $0)
  3. clip_qc     — freeze + loop detector + VLM spot-check on motion previs
  4. talking_qc  — frame-check host segments (pose variance, arc-tag fit)
  5. gate A      — deterministic assembly checks (freeze/loudness/drift)
Human gates shrink to: creative-go on a clean cut + upload approval.
"""
