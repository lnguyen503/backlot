"""Parameterized cinematic soundscape synth for the mini-movie runner.

synth_shot(kind, dur, out_path) writes a 48k stereo WAV designed for a shot type:
  - "tremor" : dread. rain + forest + birds + distant swelling BOOM footfalls, no roar.
  - "reveal" : the beast appears. footfalls close, birds scatter, a building mid roar.
  - "attack" : climax. full terrifying roar (chest+growl+snarl) building to the end.

All local, deterministic. Shared DSP with the standalone T-rex roar (v2).
"""
from __future__ import annotations
import wave
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve

SR = 48000


def _mk(dur, seed):
    N = int(dur * SR)
    return N, np.arange(N) / SR, np.random.default_rng(seed)


def bp(x, lo, hi, o=2):
    b, a = butter(o, [lo / (SR / 2), hi / (SR / 2)], btype="band"); return lfilter(b, a, x)
def lp(x, hi, o=2):
    b, a = butter(o, hi / (SR / 2), btype="low"); return lfilter(b, a, x)
def hp(x, lo, o=2):
    b, a = butter(o, lo / (SR / 2), btype="high"); return lfilter(b, a, x)
def ss(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def shape(x, drive):
    x = x + 0.28 * np.sign(x) * x * x
    return np.tanh(drive * x)


def voiced(t, N, rng, f0base, contour, nh, drive, nlo, nhi, namt, t0, tclimax):
    f0 = f0base + contour * ss((t - t0) / max(1e-6, tclimax - t0))
    f0 += 0.02 * f0base * np.sin(2 * np.pi * 5.5 * t)
    f0 += 0.05 * f0base * (rng.standard_normal(N).cumsum() / np.sqrt(N))
    fm = 1.0 + 0.18 * np.sin(2 * np.pi * (f0base * 0.37) * t)
    ph = 2 * np.pi * np.cumsum(f0 * fm) / SR
    s = sum((1.0 / k) * np.sin(k * ph) for k in range(1, nh + 1))
    s += namt * bp(rng.standard_normal(N), nlo, nhi)
    return shape(s, drive)


def roar(t, N, rng, renv, tclimax, aggression=1.0):
    """Three stacked voiced layers -> deep, weighty, aggressive roar."""
    chest = voiced(t, N, rng, 58, 14, 16, 2.6, 150, 1200, 0.30, 0.4, tclimax) * 1.35
    mid = voiced(t, N, rng, 110, 22, 12, 3.0, 350, 2200, 0.40, 0.4, tclimax) * (0.85 * aggression)
    snl = voiced(t, N, rng, 226, 36, 9, 3.4, 1000, 4200, 0.55, 0.4, tclimax) * (0.30 * aggression)
    r = chest + mid + snl
    r = (bp(r, 70, 200) * 1.3 + bp(r, 400, 1000) * 0.85 + bp(r, 1500, 3000) * (0.40 * aggression) + r * 0.4)
    r = r + 0.6 * lp(r, 160)
    r = shape(r, 1.7) * renv
    return hp(r, 40)


def footfall(t, N, rng, c, g):
    e = np.exp(-np.clip(t - c, 0, None) * 8.0) * (t >= c)
    thud = (np.sin(2 * np.pi * 46 * t) + 0.6 * np.sin(2 * np.pi * 31 * t)) * e
    click = hp(rng.standard_normal(N), 1600) * np.exp(-np.clip(t - c, 0, None) * 45) * (t >= c)
    deb = bp(rng.standard_normal(N), 200, 1800) * np.exp(-np.clip(t - c, 0, None) * 13) * (t >= c)
    return (1.7 * thud + 0.22 * click + 0.4 * deb) * g


def rain_bed(t, N, rng, level=1.0):
    r = hp(rng.standard_normal(N), 2000) * 0.028 + bp(rng.standard_normal(N), 400, 1400) * 0.022
    return lp(r, 7000) * (0.9 + 0.1 * np.sin(2 * np.pi * 7 * t)) * level


def forest(t, N, rng, level=1.0):
    wind = lp(rng.standard_normal(N), 700) * 0.12 * (0.8 + 0.2 * np.sin(2 * np.pi * 0.4 * t))
    ins = bp(rng.standard_normal(N), 4000, 8000) * (0.5 + 0.5 * np.sin(2 * np.pi * 22 * t)) * 0.05
    return (wind + ins) * level


def bird(t, c, flo, fhi, dur=0.26, g=0.05):
    e = np.exp(-((t - c) ** 2) / (2 * (dur / 3) ** 2))
    warb = flo + (fhi - flo) * (0.5 + 0.5 * np.sin(2 * np.pi * 18 * t))
    return np.sin(2 * np.pi * np.cumsum(warb) / SR) * e * g


def reverb(mix, N, rng, wet=0.2, decay=0.30):
    L = int(1.3 * SR)
    ir = rng.standard_normal(L) * np.exp(-np.arange(L) / (decay * SR))
    ir = hp(lp(ir, 6000), 250)
    w = fftconvolve(mix, ir)[:N]; w /= (np.abs(w).max() + 1e-9)
    mix = mix / (np.abs(mix).max() + 1e-9)
    return (1 - wet) * mix + wet * w


def _write(out_path, L, R, N):
    st = np.stack([L, R], 1); st /= (np.abs(st).max() + 1e-9)
    st = np.tanh(1.4 * st); st /= (np.abs(st).max() + 1e-9); st *= 0.98
    fi, fo = int(0.02 * SR), int(0.18 * SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]
    st[-fo:] *= np.linspace(1, 0, fo)[:, None]
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def synth_shot(kind, dur, out_path):
    N, t, rng = _mk(dur, {"tremor": 3, "reveal": 5, "attack": 21}.get(kind, 1))

    if kind == "tremor":
        rain = rain_bed(t, N, rng, 1.0)
        amb = forest(t, N, rng, 1.0)
        birds = bird(t, 0.5, 2600, 3400) + bird(t, 1.0, 3200, 2400, 0.22)
        # distant footfalls, spaced and swelling; birds scatter on the big one
        boom = (footfall(t, N, rng, dur * 0.45, 0.4) + footfall(t, N, rng, dur * 0.72, 0.6) +
                footfall(t, N, rng, dur * 0.94, 0.95))
        boom = lp(boom, 220)  # distant = muffled
        rumble = lp(rng.standard_normal(N), 90) * (0.1 + 0.5 * ss(t / dur)) * 0.3
        scatter = (bird(t, dur * 0.95, 3000, 4200, 0.18, 0.06) + bird(t, dur * 0.97, 3600, 2800, 0.15, 0.05))
        mix = rain + amb + birds + 1.0 * boom + rumble + scatter
        mix = reverb(mix, N, rng, 0.18)
        L = mix + 0.4 * np.concatenate([amb[130:], np.zeros(130)])
        R = mix + 0.4 * amb

    elif kind == "reveal":
        rain = rain_bed(t, N, rng, 1.0)
        amb = forest(t, N, rng, 1.0) * (1 - 0.8 * ss((t - 0.5) / 1.5))  # forest hushes
        steps = footfall(t, N, rng, dur * 0.25, 0.9) + footfall(t, N, rng, dur * 0.6, 1.0)
        rumble = lp(rng.standard_normal(N), 110) * (0.2 + 0.8 * ss(t / dur)) * 0.4
        # a building, mid-strength roar in the back half
        renv = ss((t - dur * 0.45) / (dur * 0.45)) * (1 - 0.5 * ss((t - dur * 0.92) / (dur * 0.1)))
        rr = roar(t, N, rng, np.clip(renv, 0, 1) * 0.8, dur * 0.85, aggression=0.7)
        mix = rain + amb + steps + rumble + 0.8 * rr
        mix = reverb(mix, N, rng, 0.22)
        L = mix + 0.4 * np.concatenate([rain[130:], np.zeros(130)])
        R = mix + 0.4 * rain

    else:  # attack
        tclimax = dur - 0.55
        attack = ss((t - 0.33) / 0.06)
        inten = 0.55 + 0.45 * ss((t - 0.5) / (tclimax - 0.5))
        surge = 0.30 * np.exp(-((t - tclimax) ** 2) / (2 * 0.32 ** 2))
        tail = 1.0 - 0.9 * ss((t - (tclimax + 0.1)) / (dur - (tclimax + 0.1)))
        renv = np.clip(attack * (inten + surge) * tail, 0, 1.3)
        rr = roar(t, N, rng, renv, tclimax, aggression=1.0)
        sub = (np.sin(2 * np.pi * 34 * t) + 0.7 * np.sin(2 * np.pi * 27 * t)) * lp(np.clip(renv, 0, 1), 30) * 1.05
        inh = bp(rng.standard_normal(N), 500, 4000) * np.exp(-((t - 0.24) ** 2) / (2 * 0.09 ** 2)) * 0.5
        steps = (footfall(t, N, rng, 0.5, 0.7) + footfall(t, N, rng, 1.0, 0.85) +
                 footfall(t, N, rng, 1.6, 1.0) + footfall(t, N, rng, tclimax, 1.0))
        rain = rain_bed(t, N, rng, 1.0)
        rumble = lp(rng.standard_normal(N), 110) * (0.2 + 0.8 * ss(t / tclimax)) * 0.45
        mix = 1.15 * rr + sub + 0.9 * steps + inh + rain + rumble
        mix = lp(mix, 9000) + 0.35 * lp(mix, 140)
        mix = reverb(mix, N, rng, 0.20)
        L = mix + 0.5 * np.concatenate([rain[130:], np.zeros(130)])
        R = mix + 0.5 * rain

    _write(out_path, L, R, N)
    return out_path


if __name__ == "__main__":
    import sys
    synth_shot(sys.argv[1], float(sys.argv[2]), sys.argv[3])
    print("wrote", sys.argv[3])
