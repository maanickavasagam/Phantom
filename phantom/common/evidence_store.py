"""Local, content-addressed evidence store (Day 5)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("phantom.evidence")


class EvidenceStore:
    def __init__(self, root: str | Path = "evidence_store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, kind: str, source: str, meta: dict | None = None) -> str:
        digest = hashlib.sha256(data).hexdigest()
        evidence_id = f"ev_{digest[:16]}"
        folder = self.root / evidence_id
        folder.mkdir(exist_ok=True)
        (folder / "blob").write_bytes(data)
        record = {
            "evidence_id": evidence_id,
            "sha256": digest,
            "kind": kind,
            "source": source,
            "stored_at": datetime.now(UTC).isoformat(),
            "meta": meta or {},
        }
        (folder / "provenance.json").write_text(json.dumps(record, indent=2))
        log.info("stored evidence %s kind=%s source=%s", evidence_id, kind, source)
        return evidence_id

    def get(self, evidence_id: str) -> tuple[bytes, dict]:
        folder = self.root / evidence_id
        if not folder.exists():
            raise KeyError(evidence_id)
        return (folder / "blob").read_bytes(), json.loads((folder / "provenance.json").read_text())

    def verify(self, evidence_id: str) -> bool:
        data, record = self.get(evidence_id)
        return hashlib.sha256(data).hexdigest() == record["sha256"]
