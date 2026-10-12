# Champion (best-so-far design) derivation from adjacent or baseline-anchored verdicts.
from __future__ import annotations

import re
from typing import Any


def _derive_champion(
    adjacent_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]],
    current_version: int,
) -> tuple[int, bool]:
    """Return (champion_version, arbitration_needed_for_current_version).

    Walk the adjacent verdicts v0..v{current}: the champion advances to the
    newer version only when that version beat the reigning champion DIRECTLY —
    either in its adjacent pair (champion == predecessor) or in a recorded
    champion-arbitration row. If the current version won its adjacent pair
    while the champion is older and no arbitration row exists yet, the second
    element is True: one champion-vs-current judge call is required. Older
    unresolved rounds leave the champion unchanged (conservative).
    """
    adj: dict[int, str | None] = {}
    for row in adjacent_rows:
        m = re.fullmatch(r"v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m and int(m.group(2)) == int(m.group(1)) + 1:
            adj[int(m.group(2))] = row.get("winner")
    arb: dict[tuple[int, int], str | None] = {}
    for row in champion_rows:
        m = re.fullmatch(r"champion_v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m:
            arb[(int(m.group(1)), int(m.group(2)))] = row.get("winner")
    champ = 0
    needs = False
    for k in range(1, current_version + 1):
        if adj.get(k) != "B":
            continue
        if champ == k - 1:
            champ = k
        elif (champ, k) in arb:
            if arb[(champ, k)] == "B":
                champ = k
        elif k == current_version:
            needs = True
    return champ, needs


def _derive_champion_baseline(
    history_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]],
    current_version: int,
) -> tuple[int, bool, bool]:
    """Baseline-anchored champion. Returns (champion, champion_has_baseline_win,
    arbitration_needed_for_current_version).

    Verdict convention: in v{k}_vs_no_harness rows, A = the baseline, B = the
    version, so winner "B" means the version beat the baseline. The champion
    advances only on DIRECT wins: a version's baseline win takes the crown
    outright while no champion has one (valid transitively: champ > baseline >
    others is not needed there); once a champion holds a baseline win, a new
    baseline-winner must beat it in a recorded arbitration. Losses and ties
    leave the champion untouched. With no baseline-winner at all, v0 (the
    seed) holds the crown by default WITHOUT a proven win.
    """
    wins: dict[int, Any] = {}
    for row in history_rows:
        m = re.fullmatch(r"v(\d+)_vs_no_harness", str(row.get("key") or ""))
        if m:
            wins[int(m.group(1))] = row.get("winner")
    arb: dict[tuple[int, int], Any] = {}
    for row in champion_rows:
        m = re.fullmatch(r"champion_v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m:
            arb[(int(m.group(1)), int(m.group(2)))] = row.get("winner")
    champ = 0
    champ_win = wins.get(0) == "B"
    needs = False
    for k in range(1, current_version + 1):
        if wins.get(k) != "B":
            continue
        if not champ_win:
            champ = k
            champ_win = True
        elif (champ, k) in arb:
            if arb[(champ, k)] == "B":
                champ = k
        elif k == current_version:
            needs = True
    return champ, champ_win, needs


def _baseline_scoreboard(
    history_rows: list[dict[str, Any]],
    current_version: int,
    champion: int | None,
    champion_has_win: bool,
) -> str:
    """One-line W/L/T record vs the baseline, with the champion starred."""
    verdicts: dict[int, Any] = {}
    for row in history_rows:
        m = re.fullmatch(r"v(\d+)_vs_no_harness", str(row.get("key") or ""))
        if m:
            verdicts[int(m.group(1))] = row.get("winner")
    marks = {"B": "W", "A": "L", "tie": "T"}
    parts = []
    for k in range(0, current_version + 1):
        mark = marks.get(verdicts.get(k), "?")
        star = "★" if (champion == k and champion_has_win) else ""
        parts.append(f"v{k} {mark}{star}")
    return "Record vs no_harness: " + " · ".join(parts)
