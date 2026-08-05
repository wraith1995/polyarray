"""The fem-audit gate (fem/AUDITS.md "Gate wiring").

Shells out to the workspace's mechanical audit (`fem/audit/run_audits.py`) for THIS repo at
`--min-severity ERROR` and fails on any ERROR beyond the recorded BASELINE. The baseline is a
RATCHET — per-rule ceilings owned by queued fem tasks; counts may only fall, and a rule not in
the baseline must stay at zero. A new ERROR needs a fix or a maintainer-approved pragma/allowlist
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
BASELINE: dict[str, int] = {}                       # ERROR-clean (task #16 cleared the annotations)


def _harness() -> Path | None:
    for anc in Path(__file__).resolve().parents:
        # `savo/audit/workspace/` is the home since the workspace audit moved there (it was in
        # `fem/audit`, an untracked directory). The old path is still probed so a checkout that
        # predates the move keeps working — and so this gate never silently SKIPS, which is what
        # it did in all six repos when the move landed: a gate that skips is not guarding.
        for rel in (("savo", "audit", "workspace", "run_audits.py"),
                    ("audit", "workspace", "run_audits.py"),
                    ("audit", "run_audits.py")):
            cand = anc.joinpath(*rel)
            if cand.is_file():
                return cand
    return None


def _in_a_workspace() -> bool:
    """Whether this checkout sits in a multi-repo workspace (so the harness SHOULD be there).

    The distinction the gate turns on. A repo cloned on its own genuinely has no workspace
    audit and must skip; a repo sitting next to its siblings and finding no harness means the
    harness moved or broke, and that must FAIL. Detected by the siblings the audit needs
    anyway.
    """
    for anc in Path(__file__).resolve().parents:
        if (anc / "grassmann" / "src").is_dir() and (anc / "polyarray" / "src").is_dir():
            return True
    return False


def test_fem_audit_gate() -> None:
    runner = _harness()
    if runner is None:
        # A gate that degrades to SILENCE is indistinguishable from a gate that
        # passes. Moving the workspace audit once left this skipping in all six
        # repos while every suite stayed green and nothing was audited. Skip ONLY
        # for a genuinely standalone checkout; in a workspace, a missing harness
        # is a broken gate and must fail.
        if _in_a_workspace():
            pytest.fail(
                "workspace audit harness NOT FOUND while sitting in a workspace — "
                "expected savo/audit/workspace/run_audits.py. The gate would "
                "otherwise skip silently and this suite would look green with "
                "nothing audited.")
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
        "Fix them or get a maintainer-approved pragma (agents never add approvals).\n\n" + out.stdout
    )
