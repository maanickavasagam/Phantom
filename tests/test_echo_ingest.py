import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from phantom.echo.ingest import MAX_DURATION_S, IngestError, ingest


def _write_tone(path, seconds=2.0, sr=44100, freq=440, amp=0.3, silent=False):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = np.zeros_like(t) if silent else (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sr)


@pytest.fixture
def tone_wav(tmp_path):
    p = tmp_path / "tone.wav"
    _write_tone(p)
    return p


@pytest.fixture
def silent_wav(tmp_path):
    p = tmp_path / "silent.wav"
    _write_tone(p, silent=True)
    return p


def test_wav_ingest_normalizes_rate(tone_wav, tmp_path):
    result = ingest(tone_wav, work_dir=tmp_path / "work")
    assert result.quality.sample_rate == 16000
    assert 1.8 < result.quality.duration_s < 2.2
    assert result.quality.issues == []


def test_silent_audio_flagged(silent_wav, tmp_path):
    result = ingest(silent_wav, work_dir=tmp_path / "work")
    assert "likely_silent" in result.quality.issues


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "clip.xyz"
    p.write_bytes(b"not real audio")
    with pytest.raises(IngestError):
        ingest(p, work_dir=tmp_path / "work")


def test_missing_file(tmp_path):
    with pytest.raises(IngestError):
        ingest(tmp_path / "nope.wav", work_dir=tmp_path / "work")


def test_corrupt_file_errors_cleanly(tmp_path):
    p = tmp_path / "bad.mp3"
    p.write_bytes(b"this is not an mp3 file at all")
    with pytest.raises(IngestError):
        ingest(p, work_dir=tmp_path / "work")


def test_mp3_conversion(tone_wav, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    mp3_path = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tone_wav), str(mp3_path)],
        check=True,
    )
    result = ingest(mp3_path, work_dir=tmp_path / "work")
    assert result.source_format == ".mp3"
    assert result.quality.sample_rate == 16000


def test_duration_limit_enforced(tmp_path):
    p = tmp_path / "long.wav"
    _write_tone(p, seconds=1.0)  # short source file
    # Patch the limit low to exercise the check without generating 10+ min of audio.
    import phantom.echo.ingest as echo_ingest

    original = echo_ingest.MAX_DURATION_S
    echo_ingest.MAX_DURATION_S = 0.5
    try:
        with pytest.raises(IngestError, match="exceeds"):
            ingest(p, work_dir=tmp_path / "work")
    finally:
        echo_ingest.MAX_DURATION_S = original
    assert MAX_DURATION_S == 600
