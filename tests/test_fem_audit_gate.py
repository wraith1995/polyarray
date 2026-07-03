"""The fem-audit gate (fem/AUDITS.md "Gate wiring", decided 2026-07-03).

Shells out to the workspace's mechanical audit (`fem/audit/run_audits.py`) for THIS repo at
`--min-severity ERROR` and fails on any ERROR beyond the recorded BASELINE. The baseline is a
RATCHET — per-rule ceilings owned by queued fem tasks; counts may only fall, and a rule not in
the baseline must stay at zero. A new ERROR needs a fix or a Teo-approved pragma/allowlist
entry (never agent-added — AUDITS.md approval rules).

Skips when the fem harness is not present (a standalone checkout of this repo).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = "polyarray"
# Per-rule ERROR ceilings owned by queued fem tasks (shrink-only).
BASELINE: dict[str, int] = {
    # fem task #16 (mypy configs + annotation cleanup):
    "TYPE-STR-ANNOTATION": 157,
}


def _harness() -> Path | None:
    for anc in Path(__file__).resolve().parents:
        cand = anc / "audit" / "run_audits.py"
        if cand.is_file():
            return cand
    return None


def test_fem_audit_gate() -> None:
    runner = _harness()
    if runner is None:
        pytest.skip("fem audit harness not present (standalone checkout)")
    workspace = Path(__file__).resolve().parents[1].parent      # the dir holding this repo
    out = subprocess.run(
        [sys.executable, str(runner), "--workspace", str(workspace),
         "--repos", REPO, "--min-severity", "ERROR", "--all"],
        capture_output=True, text=True,
    )
    counts = {r: int(n) for r, n in re.findall(r"^\s*ERROR\s+(\S+)\s+(\d+)\s*$", out.stdout, re.M)}
    over = {r: n for r, n in counts.items() if n > BASELINE.get(r, 0)}
    assert not over, (
        f"fem-audit: ERRORs beyond the recorded baseline {BASELINE}: {over}.\n"
        "Fix them or get a Teo-approved pragma (agents never add approvals).\n\n" + out.stdout
    )
