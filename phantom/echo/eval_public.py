"""Evaluate ECHO's scam detection on public scam-dialogue datasets (Week 4).

Independent of our own scripted calls: transcripts written by other
researchers, so the rules were never tuned on their wording.

  BothBosu/single-agent-scam-conversations   1,600 (Llama-3-70B), official train/test split
  BothBosu/multi-agent-scam-conversation     1,600 (two AI agents), official train/test split
  BothBosu/Scammer-Conversation              1,000 (Gretel generator)
  BothBosu/youtube-scam-conversations          20 real scam calls (scam-baiter videos)

All Apache-2.0, from huggingface.co/BothBosu. These are text transcripts, so this
tests claim extraction + indicators + the verdict rule, not speech recognition
(LibriSpeech covers that). Both speakers' turns are kept, as in a real ECHO
transcript (no diarization).

    python -m phantom.echo.eval_public --root data/raw/bothbosu --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from phantom.echo.claims import extract_claims
from phantom.echo.indicators import decide, detect_indicators
from phantom.echo.transcribe import Segment, Transcript, Word

csv.field_size_limit(10**8)

DATASETS = {
    "single_agent": ("single-agent-scam-conversations", "single-agent-scam-dialogue_{}.csv"),
    "multi_agent": ("multi-agent-scam-conversation", "agent_conversation_{}.csv"),
    "scammer_conversation": ("Scammer-Conversation", "gen_conver_noIdentifier_1000.csv"),
    "youtube_real": ("youtube-scam-conversations", "youtube_scam_conversations.csv"),
}
_SPEAKER = re.compile(r"(?:^|\s|\|)\s*(Suspect|Innocent|Person [AB]|Caller|Receiver)\s*:\s*")


def dialogue_to_transcript(text: str) -> Transcript:
    """One segment per speaker turn, with synthetic word timings (0.3 s/word)."""
    parts = _SPEAKER.split(text.replace("\n", " "))
    turns = [p.strip() for i, p in enumerate(parts) if i % 2 == 0 and p.strip()]
    segs, t = [], 0.0
    for i, turn in enumerate(turns):
        words = []
        for w in turn.split():
            words.append(Word(w, round(t, 2), round(t + 0.25, 2), 1.0))
            t += 0.3
        if words:
            segs.append(Segment(i, words[0].start, words[-1].end, turn, 0.0, 0.0, words))
        t += 0.5
    return Transcript("text", "en", round(t, 2), segs)


def predict(text: str) -> tuple[bool, list[str]]:
    tr = dialogue_to_transcript(text)
    d = decide(detect_indicators(tr, extract_claims(tr)))
    fired = [i.name for i in d.strong] + (["pressure_pattern"] if d.pattern else [])
    return d.suspicious, fired


def _rows(root: Path, name: str, split: str | None):
    folder, pattern = DATASETS[name]
    file = root / folder / (pattern.format(split) if "{}" in pattern else pattern)
    with open(file, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            text = r.get("dialogue") or r.get("conversation") or ""
            label = int((r.get("labels") or r.get("label") or "0").strip())
            yield text, label, r.get("type", "unknown")


def evaluate(root: Path, name: str, split: str | None = None) -> dict:
    tp = fp = fn = tn = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    fired_on = {"scam": Counter(), "genuine": Counter()}
    examples = {"missed_scams": [], "false_alarms": []}
    for text, label, typ in _rows(root, name, split):
        pred, fired = predict(text)
        by_type[f"{typ} ({'scam' if label else 'genuine'})"][0] += 1
        by_type[f"{typ} ({'scam' if label else 'genuine'})"][1] += int(pred)
        fired_on["scam" if label else "genuine"].update(fired)
        if label and pred:
            tp += 1
        elif label:
            fn += 1
            if len(examples["missed_scams"]) < 5:
                examples["missed_scams"].append({"type": typ, "text": text[:400]})
        elif pred:
            fp += 1
            if len(examples["false_alarms"]) < 5:
                examples["false_alarms"].append({"type": typ, "fired": fired, "text": text[:400]})
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "dataset": name, "split": split or "all", "n": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 3), "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else None,
        "flagged_rate_by_type": {k: round(v[1] / v[0], 3) for k, v in sorted(by_type.items())},
        "indicators_fired": {k: dict(v.most_common()) for k, v in fired_on.items()},
        "examples": examples,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/bothbosu")
    ap.add_argument("--split", default=None, help="train/test for the two split datasets")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    results = []
    for name in DATASETS:
        split = a.split if "{}" in DATASETS[name][1] else None
        if a.split and not split:
            continue
        has_splits = "{}" in DATASETS[name][1]
        results.append(evaluate(Path(a.root), name, split or ("all" if has_splits else None)))
    for m in results:
        print(f"{m['dataset']:22} {m['split']:5} n={m['n']:5} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['false_positive_rate']}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(results, indent=2))
