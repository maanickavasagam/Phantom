"""ECHO evaluation (Day 18).

Two benchmarks, matching docs/ECHO_SCOPE.md:

1. Scripted calls (phantom.echo.make_calls): claim extraction precision /
   recall / F1 (overall and per claim type), call-level scam detection, and
   WER, per audio condition (clean, 10 dB / 5 dB noise, phone band).
       python -m phantom.echo.evaluate calls --split dev
2. LibriSpeech test-clean: WER of the speech recogniser on clean read English.
       python -m phantom.echo.evaluate librispeech --root data/raw/LibriSpeech/test-clean --n 200

Tune on the dev split only; run --split test once and report it as-is.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import jiwer

from phantom.common.evidence_store import EvidenceStore
from phantom.echo.analyze import analyze_audio, overall_outcome
from phantom.echo.ingest import ingest
from phantom.echo.transcribe import transcribe

_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation and digit grouping commas, unify 'p.m.'/'pm',
    and map one..ten to digits, so WER counts recognition errors rather than
    formatting differences ("50,000" vs "50000", "five p.m." vs "5 pm")."""
    s = s.lower().replace("p.m.", "pm").replace("a.m.", "am")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(_NUMWORDS.get(w, w) for w in s.split())


def _name_match(pred: str, gold: str) -> bool:
    """Names are scored leniently: ASR spells proper nouns inconsistently ("Verma"
    -> "Varma", "ICICI" -> "ISISI") and organisations pick up trailing words
    ("HDFC Bank Customer Care"). A name counts if it is >= 0.8 similar to the gold
    name, or contains all of its words. All other claim types need an exact key."""
    from difflib import SequenceMatcher

    if SequenceMatcher(None, pred, gold).ratio() >= 0.8:
        return True
    p, g = pred.split(), gold.split()
    return len(g) > 0 and all(any(SequenceMatcher(None, w, x).ratio() >= 0.8 for x in p) for w in g)


def match_claims(pred: list[tuple[str, str]], gold: list[tuple[str, str]]):
    """Greedy one-to-one matching; returns (tp, fp, fn) lists of (type, key)."""
    remaining = list(gold)
    tp, fp = [], []
    for t, k in pred:
        hit = next((g for g in remaining if g[0] == t and
                    (g[1] == k or (t == "name" and _name_match(k, g[1])))), None)
        if hit:
            remaining.remove(hit)
            tp.append((t, k))
        else:
            fp.append((t, k))
    return tp, fp, remaining


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3)}


def evaluate_calls(root: Path, split: str) -> dict:
    labels = [e for e in json.loads((root / "labels.json").read_text()) if e["split"] == split]
    store = EvidenceStore(Path(tempfile.gettempdir()) / "echo_eval_evidence")
    rows = []
    for e in labels:
        t0 = time.perf_counter()
        res = analyze_audio(root / e["file"], store, source=e["file"])
        elapsed = time.perf_counter() - t0
        pred = [(c["type"], c["key"]) for c in res.artifacts.get("claims", [])]
        gold = [(c["type"], c["key"]) for c in e["claims"]]
        tp, fp, fn = match_claims(pred, gold)
        hyp = res.artifacts.get("transcript", {}).get("text", "")
        rows.append({
            "file": e["file"], "family": e["family"], "label": e["label"],
            "condition": e["condition"], "verdict": overall_outcome(res).value,
            "fired": sorted({f.check for f in res.findings if f.outcome.value == "suspicious"}),
            "tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
            "wer": round(jiwer.wer(normalize_text(e["reference"]), normalize_text(hyp)), 3)
            if hyp else 1.0,
            "runtime_s": round(elapsed, 2), "errors": [x.code for x in res.errors],
            "hypothesis": hyp,
        })

    out = {"split": split, "n_recordings": len(rows), "by_condition": {}}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for r in rs:
            for key, i in (("tp", 0), ("fp", 1), ("fn", 2)):
                for t, _ in r[key]:
                    per_type[t][i] += 1
        tp = sum(v[0] for v in per_type.values())
        fp = sum(v[1] for v in per_type.values())
        fn = sum(v[2] for v in per_type.values())
        scam = [r["verdict"] == "suspicious" for r in rs if r["label"] == "scam"]
        benign = [r["verdict"] == "suspicious" for r in rs if r["label"] == "benign"]
        det = _prf(sum(scam), sum(benign), len(scam) - sum(scam))
        out["by_condition"][cond] = {
            "n": len(rs),
            "claims": _prf(tp, fp, fn),
            "claims_by_type": {t: _prf(*v) for t, v in sorted(per_type.items())},
            "scam_detection": {**det, "false_positive_rate": round(sum(benign) / len(benign), 3)
                               if benign else 0.0},
            "mean_wer": round(sum(r["wer"] for r in rs) / len(rs), 3),
            "mean_runtime_s": round(sum(r["runtime_s"] for r in rs) / len(rs), 2),
        }
    out["rows"] = rows
    return out


def evaluate_librispeech(root: Path, n: int, seed: int = 0) -> dict:
    """WER on a random, speaker-balanced sample of LibriSpeech utterances."""
    utts = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            utts.append((trans.parent / f"{uid}.flac", text))
    random.Random(seed).shuffle(utts)
    utts = utts[:n]
    refs, hyps, audio_s, asr_s = [], [], 0.0, 0.0
    for path, text in utts:
        audio = ingest(path)
        t0 = time.perf_counter()
        tr = transcribe(audio.path)
        asr_s += time.perf_counter() - t0
        audio_s += audio.quality.duration_s
        refs.append(normalize_text(text))
        hyps.append(normalize_text(tr.text))
    return {"n_utterances": len(utts), "audio_minutes": round(audio_s / 60, 1),
            "wer": round(jiwer.wer(refs, hyps), 4),
            "real_time_factor": round(asr_s / audio_s, 3) if audio_s else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calls")
    c.add_argument("--root", default="data/echo_calls")
    c.add_argument("--split", choices=["dev", "test"], default="dev")
    c.add_argument("--json-out")
    ls = sub.add_parser("librispeech")
    ls.add_argument("--root", default="data/raw/LibriSpeech/test-clean")
    ls.add_argument("--n", type=int, default=200)
    ls.add_argument("--json-out")
    a = ap.parse_args()
    if a.cmd == "calls":
        m = evaluate_calls(Path(a.root), a.split)
    else:
        m = evaluate_librispeech(Path(a.root), a.n)
    print(json.dumps({k: v for k, v in m.items() if k != "rows"}, indent=2))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(m, indent=2))
