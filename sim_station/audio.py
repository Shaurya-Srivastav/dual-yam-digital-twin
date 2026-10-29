"""Beat analysis of the songs in assets/ (the user's own files, read locally, never uploaded or committed).

Decode with ffmpeg, then: log-mel spectral-flux onset envelope -> tempo from a log-prior-weighted autocorrelation ->
dynamic-programming beat tracker (Ellis 2007) -> downbeats from the bar phase with the most low-frequency energy -> a smoothed
loudness envelope. The choreography reads `beat_pos(t)`, a fractional beat counter that follows the real beats (so tempo drift
and the song's own pushes stay in time), and `energy(t)` to size the moves with the song's dynamics.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import signal

ASSETS = Path(__file__).resolve().parent / "assets"
CACHE = ASSETS / "analysis_cache.json"
SR = 22050
HOP = 512
NFFT = 2048

SONGS: Dict[str, dict] = {
    "rick": {"file": "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp3", "title": "Never Gonna Give You Up", "artist": "Rick Astley", "style": "rick"},
    "aha": {"file": "a-ha - Take On Me (Official Video) [4K].mp3", "title": "Take On Me", "artist": "a-ha", "style": "aha"},
    "beegees": {"file": "Bee Gees - Stayin' Alive (Official Music Video).mp3", "title": "Stayin' Alive", "artist": "Bee Gees", "style": "disco"},
    "click": {"file": "", "title": "Click track", "artist": "generated", "style": "click", "generated": True},
}


@dataclass
class Analysis:
    bpm: float
    beats: np.ndarray                 # beat times in seconds
    downbeats: np.ndarray             # every 4th beat, bar-aligned
    env_t: np.ndarray                 # seconds
    energy: np.ndarray                # 0..1, smoothed loudness
    low: np.ndarray                   # 0..1, smoothed bass energy
    duration: float
    half_time: bool = False           # dance every second beat only when the tracked tempo is above 140 BPM
    meta: dict = field(default_factory=dict)

    def beat_pos(self, t: float) -> float:
        """Fractional beat index at audio time t, interpolated through the tracked beats (extrapolated at the ends)."""
        if not np.isfinite(self.bpm):
            return 0.0
        b = self.beats
        if len(b) < 2:
            return t * self.bpm / 60.0
        if t <= b[0]:
            return (t - b[0]) * self.bpm / 60.0
        if t >= b[-1]:
            return len(b) - 1 + (t - b[-1]) * self.bpm / 60.0
        return float(np.interp(t, b, np.arange(len(b))))

    def bar_offset(self) -> float:
        """Index (0..3) of the first downbeat within the beats array."""
        if len(self.downbeats) == 0:
            return 0.0
        return float(np.argmin(np.abs(self.beats - self.downbeats[0])))

    def energy_at(self, t: float) -> float:
        return float(np.interp(t, self.env_t, self.energy)) if len(self.env_t) else 0.6

    def low_at(self, t: float) -> float:
        return float(np.interp(t, self.env_t, self.low)) if len(self.env_t) else 0.5

    def to_json(self) -> dict:
        return dict(bpm=float(self.bpm), beats=[round(float(x), 4) for x in self.beats], downbeats=[round(float(x), 4) for x in self.downbeats],
                    env_t=[round(float(x), 3) for x in self.env_t], energy=[round(float(x), 4) for x in self.energy], low=[round(float(x), 4) for x in self.low],
                    duration=float(self.duration), half_time=bool(self.half_time))

    @staticmethod
    def from_json(d: dict) -> "Analysis":
        return Analysis(bpm=d["bpm"], beats=np.array(d["beats"]), downbeats=np.array(d["downbeats"]), env_t=np.array(d["env_t"]), energy=np.array(d["energy"]),
                        low=np.array(d["low"]), duration=d["duration"], half_time=d.get("half_time", False))


def decode(path: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _mel_filter(n_mels: int = 64, fmin: float = 30.0, fmax: float = 8000.0) -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.linspace(0, SR / 2, NFFT // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)))
    return fb


def click_track(bpm: float = 120.0, seconds: float = 8.0, sr: int = SR) -> np.ndarray:
    """A generated click track. No recording. Used so beat tests do not need a song file."""
    if bpm <= 0 or seconds <= 0:
        raise ValueError("bpm and seconds must be positive")
    n = int(seconds * sr)
    y = np.zeros(n, dtype=float)
    step = int(round(sr * 60.0 / bpm))
    width = max(int(0.008 * sr), 8)
    window = np.hanning(width)
    for k in range(0, n - width, step):
        y[k:k + width] += window
    return y


def _quiet(duration: float, reason: str) -> Analysis:
    return Analysis(bpm=float("nan"), beats=np.zeros(0), downbeats=np.zeros(0), env_t=np.zeros(0), energy=np.zeros(0),
                    low=np.zeros(0), duration=float(duration), half_time=False, meta={"ok": False, "reason": reason})


def wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    """16-bit mono WAV. The click track is generated; it is not a recording."""
    import io
    import wave
    pcm = np.clip(np.asarray(y, dtype=float), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
    return buf.getvalue()


def analyze(y: np.ndarray) -> Analysis:
    y = np.nan_to_num(np.asarray(y, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if y.size < NFFT:
        return _quiet(y.size / SR, "shorter than one analysis window")
    f, t, Z = signal.stft(y, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    mel = np.log1p(_mel_filter() @ P * 1e3)
    flux = np.maximum(0.0, np.diff(mel, axis=1))
    onset = flux.mean(axis=0)
    onset = np.concatenate([[0.0], onset])
    onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    fps = SR / HOP
    # tempo: autocorrelation weighted by a log-normal prior around 118 BPM
    ac = signal.correlate(onset, onset, mode="full")[len(onset) - 1:]
    lags = np.arange(len(ac))
    bpms = 60.0 * fps / np.maximum(lags, 1)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpms, 1e-3) / 118.0) / 0.55) ** 2)
    ok = (bpms >= 70) & (bpms <= 185)
    score = np.where(ok, ac * prior, -np.inf)
    lag0 = int(np.argmax(score))
    # parabolic refinement of the lag
    if 1 <= lag0 < len(ac) - 1:
        a, b, c = ac[lag0 - 1], ac[lag0], ac[lag0 + 1]
        lag_f = lag0 + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        lag_f = float(lag0)
    period = lag_f                                                       # frames per beat
    bpm = 60.0 * fps / period
    # dynamic programming beat tracker
    n = len(onset)
    alpha = 100.0
    best = onset.copy()
    prev = -np.ones(n, dtype=int)
    lo, hi = int(0.5 * period), int(2.0 * period)
    offs = np.arange(lo, hi + 1)
    pen = -alpha * (np.log(offs / period)) ** 2
    for i in range(n):
        js = i - offs
        valid = js >= 0
        if not valid.any():
            continue
        cand = best[js[valid]] + pen[valid]
        k = int(np.argmax(cand))
        # take the best predecessor even when its score is negative; `or True` used to make the old gate dead
        if np.isfinite(cand[k]):
            best[i] = onset[i] + cand[k]
            prev[i] = js[valid][k]
    # start from the best score in the last two beats
    tail = np.arange(max(0, n - int(2 * period)), n)
    i = int(tail[np.argmax(best[tail])])
    idx = []
    while i >= 0:
        idx.append(i)
        i = prev[i]
    idx = np.array(idx[::-1])
    beats = idx / fps
    # downbeat phase: the one of four whose beats carry the most bass energy
    bass = P[(f >= 40) & (f <= 160)].sum(axis=0)
    bass = np.concatenate([[0.0], bass])[: n]
    bass_at = np.array([bass[min(int(b * fps), n - 1)] for b in beats])
    phase = int(np.argmax([bass_at[p::4].sum() for p in range(4)])) if len(beats) >= 8 else 0
    downbeats = beats[phase::4]
    # loudness envelope at 10 Hz
    hop10 = SR // 10
    m = len(y) // hop10
    if m < 2 or not np.isfinite(bpm) or bpm <= 0:
        return _quiet(len(y) / SR, "not enough audio to track")
    rms = np.sqrt(np.mean(y[: m * hop10].reshape(m, hop10) ** 2, axis=1))
    lowband = signal.sosfilt(signal.butter(4, 150.0, btype="low", fs=SR, output="sos"), y)
    lw = np.sqrt(np.mean(lowband[: m * hop10].reshape(m, hop10) ** 2, axis=1))
    sm = lambda x, k: np.convolve(x, np.ones(k) / k, mode="same")
    e = sm(rms, 15)
    e = np.clip((e - np.percentile(e, 5)) / (np.percentile(e, 95) - np.percentile(e, 5) + 1e-9), 0.0, 1.0)
    lo_n = sm(lw, 8)
    lo_n = np.clip((lo_n - np.percentile(lo_n, 5)) / (np.percentile(lo_n, 95) - np.percentile(lo_n, 5) + 1e-9), 0.0, 1.0)
    return Analysis(bpm=float(bpm), beats=beats, downbeats=downbeats, env_t=np.arange(m) / 10.0, energy=e, low=lo_n, duration=len(y) / SR, half_time=bool(bpm > 140.0))


_cache: Dict[str, Analysis] = {}
_lock = threading.Lock()
status = {"ready": False, "working": "", "error": ""}


def load(song: str) -> Optional[Analysis]:
    return _cache.get(song)


def analyze_all(force: bool = False) -> None:
    """Analyse every song (once; results are cached next to the files as numbers only, never audio)."""
    with _lock:
        disk = {}
        if CACHE.exists() and not force:
            try:
                disk = json.loads(CACHE.read_text())
            except Exception:
                disk = {}
        changed = False
        for sid, meta in SONGS.items():
            if meta.get("generated"):
                stamp = 1
                if not force and sid in disk and disk[sid].get("stamp") == stamp:
                    _cache[sid] = Analysis.from_json(disk[sid]["a"])
                    continue
                status["working"] = meta["title"]
                try:
                    a = analyze(click_track())
                    _cache[sid] = a
                    disk[sid] = {"stamp": stamp, "a": a.to_json()}
                    changed = True
                except Exception as exc:
                    status["error"] = f"{sid}: {type(exc).__name__}: {exc}"
                continue
            path = ASSETS / meta["file"]
            stamp = path.stat().st_mtime if path.exists() else 0
            if sid in disk and disk[sid].get("stamp") == stamp:
                _cache[sid] = Analysis.from_json(disk[sid]["a"])
                continue
            if not path.exists():
                continue
            status["working"] = meta["title"]
            try:
                a = analyze(decode(path))
                _cache[sid] = a
                disk[sid] = {"stamp": stamp, "a": a.to_json()}
                changed = True
            except Exception as exc:
                status["error"] = f"{sid}: {type(exc).__name__}: {exc}"
        if changed:
            CACHE.write_text(json.dumps(disk))
        status.update(ready=True, working="")


if __name__ == "__main__":
    analyze_all(force=True)
    for sid, a in _cache.items():
        print(f"{sid}: {a.bpm:.1f} BPM, {len(a.beats)} beats, first beat {a.beats[0]:.2f}s, first downbeat {a.downbeats[0]:.2f}s, "
              f"duration {a.duration:.0f}s, half_time={a.half_time}, beat spacing std {np.std(np.diff(a.beats)) * 1000:.0f} ms")
