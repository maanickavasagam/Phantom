"""ECHO audio ingestion (Day 14).

Validates and normalizes permitted audio recordings: format conversion to
16kHz mono WAV (via ffmpeg), duration checks against the project's 10-minute
limit, and basic silence/corruption detection, with clear errors instead of
crashes on bad input.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a"}
MAX_DURATION_S = 600  # 10 minutes, per docs/ECHO_SCOPE.md
TARGET_SAMPLE_RATE = 16000
SILENCE_RMS_THRESHOLD = 1e-4


class IngestError(ValueError):
    pass


@dataclass
class AudioQualityReport:
    duration_s: float
    sample_rate: int
    rms: float
    issues: list[str]


@dataclass
class IngestedAudio:
    path: Path  # path to the normalized 16kHz mono WAV
    source_format: str
    quality: AudioQualityReport


def _ffmpeg_convert(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"ffmpeg conversion timed out for {src.name}") from e
    if proc.returncode != 0 or not dst.exists():
        raise IngestError(f"ffmpeg could not decode {src.name}: {proc.stderr.strip()[:300]}")


def _quality_check(wav_path: Path) -> AudioQualityReport:
    try:
        data, sr = sf.read(wav_path, dtype="float32")
    except Exception as e:  # noqa: BLE001 - surfacing any decode failure as IngestError
        raise IngestError(f"could not read normalized audio: {e}") from e
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    issues = []
    if duration == 0:
        issues.append("empty_audio")
    elif duration > MAX_DURATION_S:
        issues.append("exceeds_duration_limit")
    if rms < SILENCE_RMS_THRESHOLD:
        issues.append("likely_silent")
    return AudioQualityReport(round(duration, 2), sr, round(rms, 6), issues)


def ingest(path: str | Path, work_dir: str | Path = "/tmp/echo_ingest") -> IngestedAudio:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise IngestError(
            f"unsupported audio format: {ext} (supported: {sorted(SUPPORTED_FORMATS)})"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized = work_dir / f"{path.stem}.norm.wav"
    _ffmpeg_convert(path, normalized)
    quality = _quality_check(normalized)
    if "exceeds_duration_limit" in quality.issues:
        raise IngestError(
            f"{path.name} is {quality.duration_s}s, exceeds the {MAX_DURATION_S}s limit"
        )
    return IngestedAudio(path=normalized, source_format=ext, quality=quality)
