# Production notes — lessons learned the expensive way

Everything below was learned producing real multi-episode video with this engine. The QC rules
in `src/backlot/qc/` encode most of it as machine checks; this page is the narrative version so
you understand *why* before you hit the same walls.

## Storyboard / character consistency

- **Anchor, never chain.** Deriving each shot's start frame from the PREVIOUS shot's last frame
  compounds drift — identity collapses within ~8 shots. Instead, anchor every panel to ONE locked
  character reference (FLUX Kontext instruction edits from the same master). This is what the
  storyboard's asset-card → panel linking does; keep it that way.
- **The reference is the moat.** Save characters/worlds to the library (`runs/library/`) and
  re-import them per project, so a recurring cast stays the SAME cast instead of being re-drafted.
- **Frames first.** Always render per-panel stills and fix the bad ones BEFORE animating.
  Regenerating one still costs seconds; regenerating a video costs minutes. This is the single
  biggest hit-rate lever.
- **i2v inherits still defects.** Garbled signage, extra limbs, wrong props in the still will be
  faithfully animated. Repair the STILL, then re-roll the clip — never try to prompt a video model
  out of a defect the still already has.
- **Prompt-authoring traps** (all encoded in `qc/rules.py::beats-lint`):
  - crowd words ("fans", "villagers") render the same face N times — describe individuals, or
    keep crowds distant/blurred/from-behind;
  - a text surface (sign/banner/poster) with no quoted string gets invented garble — quote the
    exact SHORT text (≤20 chars) or drop the surface;
  - parked vehicles fly, glasses of water ripple forever, paper plates float — avoid physics asks
    the model tier can't hold;
  - articulated action (dancing, diving, fighting) and fine hand work (typing, tying) are the
    weakest classes — write around them;
  - spatial shots need an explicit CAMERA POSITION ("over her shoulder, from inside the room"),
    not actor descriptions;
  - single-action phrasing for handoffs — i2v loops actions to fill the duration, so "she hands
    him the cup" becomes an endless boomerang if the clip is longer than the action.
- **Aspect-ratio discipline.** i2v models follow the input still's AR. A portrait ref fed to a
  landscape edit stretches faces ~2x. All assembly scaling should be scale-to-cover + center-crop
  (that is what `storyboard/assemble.py` does); never flat-scale across ARs.

## Character mastering / expression sheets

The engine's signature capability — one character, same face and voice, across every scene and
episode — rests on a "master + derivations" discipline:

- **One master reference rules everything.** Cast the character once (generate several takes,
  pick for *expression plasticity*, not just looks — a face that can move reads better animated).
  Every downstream still is a Kontext edit anchored to this master.
- **Expression sheets derive nearest-neighbor.** Kontext cannot make a big semantic jump in one
  hop (it will not turn a wide smile into deadpan). Build the sheet stepwise: warm → neutral →
  deadpan; neutral → surprised; etc. Use GEOMETRIC wording ("lips together, mouth closed,
  eyebrows level"), not emotion words, and keep the eyes open in every derivation.
- **Verify sheet identity with a face-embedding check.** Expression derivations drift — measure
  cosine similarity of every sheet still against the master (`qc/faceid.py`); anything below
  ~0.9 will read as a DIFFERENT PERSON when two clips driven from different stills are cut
  together. Re-lock drifted stills (FaceLocker) or re-derive.
- **Stitching clips from different expression stills exposes that drift.** In episode work,
  junctions between per-emotion talking segments are hidden by b-roll cutaways; in a standalone
  talking-head piece every seam is naked. For continuous no-cutaway pieces, drive ONE still with
  the full (emotion-performed) voice track instead of stitching per-emotion clips.
- **Voice is half the identity.** Lock a voice per character (library `voice` field): a Kokoro
  id, or an expressive Chatterbox clone of a synthetic reference. Emotion enters through
  per-line TTS intensity, not by changing voices. A consistent face with a wandering voice
  breaks the character just as badly as face drift.
- **Motion re-invents faces.** i2v regenerates the face during motion even from a perfect
  keyframe. For non-talking shots the identity re-lock (insightface swap + GFPGAN restore,
  `pipelines/consistent_video.py`) seats the reference face back onto every frame. For TALKING
  shots, skip the re-lock — see below.

## Stitching / assembly

- **ffmpeg 7.x cannot decode ComfyUI's animated WEBP** (`SaveAnimatedWEBP` from SVD/Wan).
  `assemble._to_mp4` transcodes webp → mp4 via imageio first. Don't remove that path.
- **Wan-style savers can emit animated WEBP *named* `.mp4`**, and its container duration probes
  as N/A — probe streams / count frames, don't trust the container.
- **A/V drift in chained crossfades (KNOWN ISSUE).** `assemble._concat_xfade` places video
  junctions from measured durations while chained `acrossfade` places audio by each clip's own
  audio length. When a clip's audio and video stream lengths differ (talking clips: video
  quantized to 25fps frames vs. WAV length), every junction adds offset and lips drift across a
  multi-segment stitch. The proven fix — keep the xfade video chain, but place each segment's
  audio ABSOLUTELY on the final timeline (`adelay` to the segment's video start + `amix
  normalize=0`) — is not yet ported into `_concat_xfade`. Until it is, prefer hard cuts
  (`crossfade=0`) for talking content.
- **Check STREAM durations, not the container.** A muxed file's container length equals the
  longest stream — a video that freezes at 20s under a 33s audio track still probes as 33s.
  Verify `ffprobe -select_streams v:0` frame count ≈ audio duration × fps.
- **Stretch, never freeze.** When voiceover overruns video, stretch the video
  (`setpts` + motion interpolation) instead of padding with a freeze frame — freeze tails are the
  #1 source of "it stutters" feedback.
- **Duck audio under dialogue.** `mux_audio` mixes music/ambient/narration as separate legs with
  dialogue at full volume. Keep underscores sparse (no drums for quiet scenes) — a literal
  "rain" score tag renders as noise, not mood.

## Talking heads (InfiniteTalk)

- **The locked recipe:** Kontext still → InfiniteTalk → **no face re-lock** → expressive TTS →
  assemble + score. A post-hoc face-swap re-lock overwrites and closes the mouth, fighting the
  lip-sync — raw InfiniteTalk output is the good output.
- **`audio_scale` trades articulation vs. head stability.** 1.0 = full lip articulation (default,
  right for near-frontal); ~0.5 calms head motion but under-drives the mouth. Never ship 0.5 for
  a frontal host.
- **The warp-free grammar:** the driver warps jaws on big head ROTATION, and in a two-shot the
  LISTENER is driven by the speaker's audio. So: calm near-frontal driven SINGLES for talking,
  static-still + slow push-in for holds, and shot/reverse-shot singles instead of driven
  two-shots. The multi-person workflow (`talkhost_infinitetalk_multi`) works, but singles read
  more real — use the two-shot as the exception.
- **Watch the frame-count constant.** The InfiniteTalk graphs carry an `INTConstant` frame cap
  (e.g. 500 frames ≈ 20s at 25fps) that silently truncates longer audio — raise it to
  `audio_seconds * 25 + pad` or the tail freezes. (See "check stream durations" above — this
  failure mode hides in the container.)
- **Synthetic voice references are the privacy-safe default.** Cloning needs a reference voice;
  never default to a real person's. The engine synthesizes a Kokoro reference and clones THAT
  (expressive but never a real human) — keep that default.

## VRAM / operations

- **Free VRAM before every heavy run.** ComfyUI holds the last model set; the second heavy job in
  a row gets silently OS-OOM-killed (no traceback). `POST /free {"unload_models":true,
  "free_memory":true}` first (the engine's heavy paths do this).
- **Phase heavy stages into separate processes** where possible; clear ComfyUI's queue of
  orphaned jobs — they pile up and steal the GPU.
- **Checkpoint long renders.** The Director checkpoints after every step and resumes via
  `board_id`; per-segment work dirs let an interrupted batch resume instead of restarting.
- **Blender headless must have a GPU device enabled.** Under `--factory-startup` no compute
  device is configured — Cycles silently CPU-renders (hundreds of times slower). The bridge's
  `enable_gpu()` forces OptiX/CUDA in the prelude; keep it in any new scene code.

## Model-specific notes

- **FLUX Kontext** edits are nearest-neighbor: it cannot make a big semantic jump in one hop
  (e.g. remove a wide smile) — derive expressions stepwise, use geometric wording ("mouth
  closed, lips together") over emotion words, and keep eyes open in references.
- **Wan 14B > 5B** for faces and anatomy, but motion artifacts (extra limbs) are reduced, not
  gone — higher res + calmer motion prompts help.
- **SVD** makes 25 frames/clip; interpolate to 30fps (`minterpolate`) or prefer Wan/LTX for
  montage work.
- **Stylized non-human characters look far better than photoreal humans** on local models — they
  sidestep the uncanny valley entirely. If you control the creative, this is the sweet spot.
