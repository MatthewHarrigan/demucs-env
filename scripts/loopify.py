#!/usr/bin/env python3
"""
Auto-extract musically aligned drum loops.

v2 strategy:
- Detect beat grid with librosa beat_track
- Optionally detect coarse song sections via agglomerative segmentation
- Pick N-bar candidate windows per section
- Rank by seam quality (start/end similarity) + energy
- Export one clean loop WAV + repeated MP3 practice loop (or one per section)
"""

from __future__ import annotations

import argparse
import json
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: loopify requires librosa. Install in this repo venv:\n"
        "  source .venv/bin/activate && uv pip install librosa\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    sys.exit(2)


def seam_error(seg: np.ndarray, sr: int) -> float:
    """Lower is better. Compare first/last 80ms to estimate loop click risk."""
    if len(seg) < sr // 5:
        return float("inf")
    w = min(int(sr * 0.08), len(seg) // 4)
    if w <= 128:
        return float("inf")
    a = seg[:w]
    b = seg[-w:]
    a = a / (np.max(np.abs(a)) + 1e-9)
    b = b / (np.max(np.abs(b)) + 1e-9)
    return float(np.mean((a - b) ** 2))


def detect_sections(mono: np.ndarray, sr: int, section_count: int) -> list[tuple[int, int]]:
    """Return coarse sections as [(start_sample, end_sample), ...]."""
    if section_count <= 1:
        return [(0, len(mono))]

    # Feature matrix for segmentation
    y_harm, y_perc = librosa.effects.hpss(mono)
    feat = np.vstack(
        [
            librosa.feature.mfcc(y=y_harm, sr=sr, n_mfcc=12),
            librosa.feature.chroma_cqt(y=y_harm, sr=sr),
            librosa.feature.rms(y=y_perc),
        ]
    )

    # Ensure section_count is feasible
    n_frames = feat.shape[1]
    k = max(1, min(section_count, max(1, n_frames // 32)))
    if k == 1:
        return [(0, len(mono))]

    bounds = librosa.segment.agglomerative(feat, k=k)
    # bounds are frame indices; include start/end
    bounds = sorted(set([0, *bounds.tolist(), n_frames - 1]))
    samples = librosa.frames_to_samples(np.array(bounds))
    samples[0] = 0
    samples[-1] = len(mono)

    sections: list[tuple[int, int]] = []
    for i in range(len(samples) - 1):
        s = int(samples[i])
        e = int(samples[i + 1])
        if e - s >= int(1.5 * sr):
            sections.append((s, e))

    if not sections:
        sections = [(0, len(mono))]
    return sections


def pick_loop_samples(
    mono: np.ndarray,
    sr: int,
    bars: int,
    beats_per_bar: int,
    min_seconds: float,
    max_seconds: float,
    base_offset: int = 0,
) -> tuple[int, int, float, float]:
    """Return (start_sample_abs, end_sample_abs, tempo_bpm, seam_score)."""
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, trim=False)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if tempo.size else 0.0
    beat_samples = librosa.frames_to_samples(beat_frames)

    if len(beat_samples) < bars * beats_per_bar + 2:
        raise RuntimeError(
            f"Not enough detected beats ({len(beat_samples)}) for {bars} bars."
        )

    window_beats = bars * beats_per_bar
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]

    best = None
    for i in range(0, len(beat_samples) - window_beats):
        s = int(beat_samples[i])
        e = int(beat_samples[i + window_beats])
        if e <= s:
            continue
        dur = (e - s) / sr
        if dur < min_seconds or dur > max_seconds:
            continue

        seg = mono[s:e]
        seam = seam_error(seg, sr)
        if not np.isfinite(seam):
            continue

        rf_s = max(0, int((s / sr) * sr / 512))
        rf_e = min(len(rms), max(rf_s + 1, int((e / sr) * sr / 512)))
        energy = float(np.mean(rms[rf_s:rf_e])) if rf_e > rf_s else 0.0

        score = seam - (0.05 * energy)
        if best is None or score < best[0]:
            best = (score, s, e, seam)

    if best is None:
        raise RuntimeError("Failed to find a valid loop window.")

    _, s, e, seam = best
    return base_offset + int(s), base_offset + int(e), float(tempo), float(seam)


def ffmpeg_repeat_mp3(src_wav: Path, dst_mp3: Path, repeats: int, bitrate: str) -> None:
    loops = max(0, repeats - 1)
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        str(loops),
        "-i",
        str(src_wav),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(dst_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")


def export_one(
    data: np.ndarray,
    sr: int,
    in_path: Path,
    outdir: Path,
    base_name: str,
    bars: int,
    repeats: int,
    bitrate: str,
    s_abs: int,
    e_abs: int,
    tempo: float,
    seam: float,
    section_index: int | None = None,
) -> dict:
    loop_data = data[s_abs:e_abs]
    loop_seconds = len(loop_data) / sr

    suffix = f"_section{string.ascii_uppercase[section_index]}" if section_index is not None else ""
    loop_wav = outdir / f"{base_name}{suffix}_loop_{bars}bar.wav"
    loop_mp3 = outdir / f"{base_name}{suffix}_loop_{bars}bar_x{repeats}.mp3"
    meta_json = outdir / f"{base_name}{suffix}_loop_{bars}bar.json"

    sf.write(str(loop_wav), loop_data, sr)
    ffmpeg_repeat_mp3(loop_wav, loop_mp3, repeats=repeats, bitrate=bitrate)

    meta = {
        "input": str(in_path),
        "sample_rate": sr,
        "tempo_bpm": round(tempo, 2),
        "bars": bars,
        "start_sample": int(s_abs),
        "end_sample": int(e_abs),
        "start_seconds": round(s_abs / sr, 3),
        "end_seconds": round(e_abs / sr, 3),
        "loop_seconds": round(loop_seconds, 3),
        "seam_error": round(seam, 6),
        "loop_wav": str(loop_wav),
        "practice_mp3": str(loop_mp3),
        "section": None if section_index is None else string.ascii_uppercase[section_index],
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="Auto-extract bar-aligned loops")
    p.add_argument("input", help="Input audio path (wav/mp3/aif/etc)")
    p.add_argument("--bars", type=int, default=2, help="Bars per loop (default: 2)")
    p.add_argument("--beats-per-bar", type=int, default=4, help="Time signature beats (default: 4)")
    p.add_argument("--repeats", type=int, default=16, help="How many times to repeat for practice mp3 (default: 16)")
    p.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k)")
    p.add_argument("--outdir", default="output/loops", help="Output directory (default: output/loops)")
    p.add_argument("--name", default=None, help="Output base name (default: derived from input)")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=16.0)
    p.add_argument("--sections", type=int, default=1, help="Generate one loop per detected section (default: 1)")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    base_name = args.name or in_path.stem

    data, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)

    sections = detect_sections(mono, sr, args.sections)
    results: list[dict] = []

    for idx, (sec_s, sec_e) in enumerate(sections):
        seg = mono[sec_s:sec_e]
        if len(seg) < int(args.min_seconds * sr):
            continue
        try:
            s_abs, e_abs, tempo, seam = pick_loop_samples(
                mono=seg,
                sr=sr,
                bars=args.bars,
                beats_per_bar=args.beats_per_bar,
                min_seconds=args.min_seconds,
                max_seconds=args.max_seconds,
                base_offset=sec_s,
            )
            meta = export_one(
                data=data,
                sr=sr,
                in_path=in_path,
                outdir=outdir,
                base_name=base_name,
                bars=args.bars,
                repeats=args.repeats,
                bitrate=args.bitrate,
                s_abs=s_abs,
                e_abs=e_abs,
                tempo=tempo,
                seam=seam,
                section_index=(idx if args.sections > 1 else None),
            )
            results.append(meta)
        except Exception as exc:
            # skip weak sections silently-ish
            print(f"Skipping section {idx+1}: {exc}", file=sys.stderr)

    if not results:
        print("No valid loops generated.", file=sys.stderr)
        return 1

    print("==========================================")
    print(f"Input: {in_path}")
    print(f"Sections requested: {args.sections} | generated: {len(results)}")
    for r in results:
        tag = f" [{r['section']}]" if r.get("section") else ""
        print(f"- Loop{tag}: {r['start_seconds']}s -> {r['end_seconds']}s ({r['loop_seconds']}s), tempo {r['tempo_bpm']} bpm")
        print(f"  seam={r['seam_error']} | mp3={r['practice_mp3']}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
