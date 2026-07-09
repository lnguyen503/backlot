"""Talking-host POC runner: portrait + voice -> Sonic lip-synced talking head.

POC RESULT (2026-06-26): works end-to-end, ~4.5 min for a 10s clip at 512x768,
~$0 (local on RTX 5090). BUT the result was UNCANNY/creepy -- not release-ready.

Diagnosed causes of the uncanniness (-> see TODO talkhost roadmap):
  1. Used a FULL-BODY portrait -> face too small to animate well. Use a
     HEAD-AND-SHOULDERS close-up instead (biggest single fix).
  2. 512 resolution -> soft mouth/teeth. Raise min_resolution to 768-1024.
  3. SVD backbone warps faces subtly. Consider Hallo2/Hallo3 or lip-sync onto a
     real base video (MuseTalk) for naturalness.
  4. Stiff default motion. Lower dynamic_scale; add expression control.

ONE-TIME SETUP (so this is reproducible):
  - ComfyUI custom node: custom_nodes/ComfyUI_Sonic (git clone)
  - Sonic deps into the ComfyUI RUNTIME venv (NOT standalone-env):
      ".../ComfyUI (1)/ComfyUI/.venv/Scripts/python.exe" -m pip install \
        omegaconf librosa diffusers opencv-python-headless imageio imageio-ffmpeg einops
  - Models in <ComfyUI install>/models/sonic/ (we junctioned it to ComfyUI-Shared/models/sonic):
      unet.pth, audio2token.pth, audio2bucket.pth, yoloface_v5m.pt  (FLAT, not nested)
      RIFE/flownet.pkl, whisper-tiny/   ; SVD svd_xt.safetensors in models/checkpoints
  - Stage inputs in ComfyUI input/: host_face.png, host_line.wav (Kokoro TTS)

Voice: generated with Kokoro (CPU, Apache-licensed). See PROGRESS.md.
"""
import json, time, urllib.request

SERVER = "http://127.0.0.1:8188"
GRAPH = {
    "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": "svd_xt.safetensors"}},
    "2": {"class_type": "SONICTLoader", "inputs": {
        "model": ["1", 0], "sonic_unet": "unet.pth",
        "ip_audio_scale": 1.0, "use_interframe": True, "dtype": "fp16"}},
    "3": {"class_type": "LoadImage", "inputs": {"image": "host_face.png"}},
    "4": {"class_type": "LoadAudio", "inputs": {"audio": "host_line.wav"}},
    "5": {"class_type": "SONIC_PreData", "inputs": {
        "clip_vision": ["1", 1], "vae": ["1", 2], "audio": ["4", 0], "image": ["3", 0],
        "weight_dtype": ["2", 1], "min_resolution": 512, "duration": 11.0, "expand_ratio": 0.5}},
    "6": {"class_type": "SONICSampler", "inputs": {
        "model": ["2", 0], "data_dict": ["5", 0], "seed": 12345,
        "inference_steps": 25, "dynamic_scale": 1.0, "fps": 25.0}},
    "7": {"class_type": "CreateVideo", "inputs": {"images": ["6", 0], "fps": ["6", 1], "audio": ["4", 0]}},
    "8": {"class_type": "SaveVideo", "inputs": {
        "video": ["7", 0], "filename_prefix": "backlot/talkinghost", "format": "mp4", "codec": "h264"}},
}


def main():
    data = json.dumps({"prompt": GRAPH}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data, headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    print("submitted:", pid, flush=True)
    while True:
        time.sleep(8)
        h = json.load(urllib.request.urlopen(f"{SERVER}/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                print("DONE:", json.dumps(h[pid].get("outputs", {}))[:400], flush=True); break
            if st.get("status_str") == "error":
                print("ERROR:", json.dumps(st)[:600], flush=True); break
        print("  ...rendering", flush=True)


if __name__ == "__main__":
    main()
