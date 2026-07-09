"""Clone a voice and speak text. Backends: chatterbox (expressive) | f5 (faithful).

Run from the venv that has the backend:
  .venv-cbx\\Scripts\\python tests\\voice_clone.py --backend chatterbox --ref REF.wav --text "..." --out OUT.wav
  .venv-tts\\Scripts\\python tests\\voice_clone.py --backend f5 --ref REF.wav --ref-text "..." --text "..." --out OUT.wav

torchaudio.load/save are redirected to soundfile (this box's torchaudio routes I/O through
torchcodec, whose prebuilt libs are ABI-incompatible with the installed torch).
"""
import argparse

import soundfile as sf
import torch
import torchaudio

torchaudio.load = lambda p, *a, **k: (
    torch.from_numpy(sf.read(str(p), dtype="float32", always_2d=True)[0].T.copy()),
    sf.read(str(p), dtype="float32", always_2d=True)[1],
)
def _save(p, t, sr, *a, **k):
    arr = t.detach().cpu().numpy()
    sf.write(str(p), arr.T if arr.ndim == 2 else arr, int(sr))
torchaudio.save = _save


def _write(out, arr, sr):
    sf.write(out, arr.T if arr.ndim == 2 else arr.squeeze(), int(sr))


def gen_chatterbox(args, text):
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device=args.device)
    wav = m.generate(text, audio_prompt_path=args.ref,
                     exaggeration=args.exaggeration, cfg_weight=args.cfg)
    _write(args.out, wav.detach().cpu().numpy(), m.sr)


def gen_f5(args, text):
    from f5_tts.api import F5TTS
    t = F5TTS(model="F5TTS_v1_Base", device=args.device)
    t.infer(ref_file=args.ref, ref_text=args.ref_text, gen_text=text,
            file_wave=args.out, remove_silence=True, seed=args.seed)


def _transcribe(ref):
    """Auto-transcribe the reference clip (whisper) when no --ref-text given."""
    import whisper
    return whisper.load_model("base").transcribe(str(ref))["text"].strip()


def gen_qwen3tts(args, text):
    """Voicebox's cloning engine: Qwen3-TTS (Apache-2.0), zero-shot voice clone."""
    from qwen_tts import Qwen3TTSModel
    dev = args.device if args.device != "cpu" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_text = args.ref_text or _transcribe(args.ref)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=dev,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    wavs, sr = model.generate_voice_clone(
        text=text, language=args.language, ref_audio=str(args.ref), ref_text=ref_text)
    _write(args.out, wavs[0], sr)


def main():
    ap = argparse.ArgumentParser(description="Clone a voice and speak text.")
    ap.add_argument("--backend", choices=["chatterbox", "f5", "qwen3tts"], default="chatterbox",
                    help="chatterbox=expressive; f5=faithful; qwen3tts=Voicebox's Qwen3-TTS (Apache-2.0).")
    ap.add_argument("--ref", required=True, help="Reference voice WAV to clone.")
    ap.add_argument("--ref-text", default="", help="Transcript of --ref (f5/qwen3tts; auto-whisper if empty).")
    ap.add_argument("--language", default="English", help="Target language (qwen3tts).")
    ap.add_argument("--text", help="Text to speak.")
    ap.add_argument("--text-file", help="File with text to speak (overrides --text).")
    ap.add_argument("--out", required=True, help="Output WAV path.")
    ap.add_argument("--exaggeration", type=float, default=0.6, help="Chatterbox expressiveness.")
    ap.add_argument("--cfg", type=float, default=0.4, help="Chatterbox cfg_weight (lower = more expressive).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    text = open(args.text_file, encoding="utf-8").read().strip() if args.text_file else args.text
    {"chatterbox": gen_chatterbox, "f5": gen_f5, "qwen3tts": gen_qwen3tts}[args.backend](args, text)
    print(f"VOICE_DONE {args.out} {sf.info(args.out).duration:.2f}s", flush=True)


if __name__ == "__main__":
    main()
