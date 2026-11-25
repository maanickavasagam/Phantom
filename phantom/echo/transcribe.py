"""ECHO speech recognition (Day 15).

Engine decision (docs/ECHO_SCOPE.md, resolved Week 4): local Whisper via
faster-whisper (CTranslate2, int8 on CPU). Audio never leaves the machine, no
API key or per-call cost. English-only for v1 (`base.en` by default).

Every word keeps its start/end time and Whisper's probability, so a claim
extracted later can point to the exact seconds of audio it came from.

Configuration (environment):
  ECHO_WHISPER_MODEL   model name or local path (default "base.en")
  ECHO_MODEL_DIR       where models are downloaded/cached (default: HF cache)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import soundfile as sf

DEFAULT_MODEL = "base.en"
VAD_THRESHOLD = 0.2  # chosen on the dev split's phone-band copies, see docs/ECHO_EVAL.md


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    model: str
    language: str
    duration_s: float
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def mean_word_probability(self) -> float:
        ws = self.words
        return sum(w.probability for w in ws) / len(ws) if ws else 0.0

    def to_dict(self) -> dict:
        return {"model": self.model, "language": self.language, "duration_s": self.duration_s,
                "text": self.text, "segments": [asdict(s) for s in self.segments]}


class TranscriptionError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_model(name: str, download_root: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - dependency missing
        raise TranscriptionError(
            "faster-whisper is not installed (pip install faster-whisper)"
        ) from e
    return WhisperModel(name, device="cpu", compute_type="int8", download_root=download_root)


def model_name() -> str:
    return os.environ.get("ECHO_WHISPER_MODEL", DEFAULT_MODEL)


def transcribe(wav_path: str | Path, model: str | None = None) -> Transcript:
    """Transcribe a normalized 16 kHz mono WAV (see echo.ingest)."""
    name = model or model_name()
    whisper = _load_model(name, os.environ.get("ECHO_MODEL_DIR"))
    # Pass samples, not a path: ingestion already produced 16 kHz mono, and this
    # avoids faster-whisper's PyAV decoder (version-sensitive on Windows).
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise TranscriptionError(f"expected 16 kHz audio from ingestion, got {sr} Hz")
    try:
        segments, info = whisper.transcribe(
            audio,
            language="en",
            beam_size=5,
            word_timestamps=True,
            # Skip long silences (reduces hallucinated text), but with a low speech
            # threshold: at the default 0.5 the VAD discarded whole sentences of
            # phone-band (300-3400 Hz) speech (dev phone WER 41.8% -> 5.1%).
            vad_filter=True,
            vad_parameters={"threshold": VAD_THRESHOLD, "min_silence_duration_ms": 1000},
            condition_on_previous_text=False,  # limits error carry-over between segments
        )
        segs = []
        for s in segments:
            segs.append(Segment(
                id=int(s.id), start=round(float(s.start), 2), end=round(float(s.end), 2),
                text=s.text.strip(), avg_logprob=round(float(s.avg_logprob), 3),
                no_speech_prob=round(float(s.no_speech_prob), 3),
                words=[Word(w.word.strip(), round(float(w.start), 2), round(float(w.end), 2),
                            round(float(w.probability), 3)) for w in (s.words or [])],
            ))
    except Exception as e:  # noqa: BLE001 - surface any engine failure as one error type
        raise TranscriptionError(f"speech recognition failed: {e}") from e
    return Transcript(model=name, language=info.language, duration_s=round(info.duration, 2),
                      segments=segs)
