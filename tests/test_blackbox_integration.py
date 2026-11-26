"""Exercise the browser and API together on a local synthetic site."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx


def test_blackbox_browser():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-m", "uvicorn", "phantom.api.main:app",
        "--host", "127.0.0.1", "--port", str(port), "--log-level", "error",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise AssertionError("API process exited before readiness")
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise AssertionError("API did not start")
        response = httpx.post(
            f"{base}/blackbox/investigate",
            json={"scenario": "fees", "strategy": "learned", "action_budget": 1},
            timeout=60,
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["findings"] and result["artifacts"]["trace"]
        assert len(result["evidence_ids"]) >= 3
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
