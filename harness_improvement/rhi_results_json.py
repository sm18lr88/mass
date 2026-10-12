# Appendix of results/*.json files (with per-query exclusions and caps) for the optimizer evidence.
from __future__ import annotations

from pathlib import Path


def _build_all_results_json_evidence(
    repo_root: Path,
    query_name: str,
    *,
    per_file_max_chars: int | None = None,
    total_max_chars: int | None = None,
) -> str:
    results_dir = repo_root / "results"
    if not results_dir.is_dir():
        return "- `results/` directory is missing."

    excluded_for_query202 = {
        # Large prediction blobs that frequently push prompts over context limits.
        "curve_data.json",
        "distances_angles_predictions.json",
        "distances_only_predictions.json",
        "run_B0_family_s0.json",
        "run_B0_family_s1.json",
        "run_B0_family_s2.json",
        "run_B0_random_s0.json",
        "run_B1_family_s0.json",
        "run_B1_family_s1.json",
        "run_B1_family_s2.json",
        "run_B1_random_s0.json",
        "run_M1_family_s0.json",
        "run_M1_family_s1.json",
        "run_M1_family_s2.json",
        "run_M1_random_s0.json",
        "run_M2_family_s0.json",
        "run_M2_family_s1.json",
        "run_M2_family_s2.json",
        "run_M2_random_s0.json",
        "run_M3_family_s0.json",
        "run_M3_family_s1.json",
        "run_M3_family_s2.json",
        "run_M3_random_s0.json",
    }
    excluded_for_query207 = {
        # Extremely large file; including full raw JSON can exceed model context window.
        "esmfold_plddts.json",
    }
    excluded_for_query220 = {
        # Large per-sample test result dumps that can dominate prompt token budget.
        "invariant_baseline_test_results.json",
        "se3_gvp_small_test_results.json",
        "se3_gvp_test_results.json",
    }
    excluded_for_query57 = {
        # Omit this JSON from prompt evidence for query57.
        "data_quality_issues.json",
    }
    excluded_globally = {
        # Large per-run raw record dumps that dominate prompt token budget.
        "raw_records.json",
        "pilot_raw_records.json",
    }
    excluded_by_query: dict[str, set[str]] = {
        "query57_ourTeam": excluded_for_query57,
        "query202_ourTeam": excluded_for_query202,
        "query207_ourTeam": excluded_for_query207,
        "query220_ourTeam": excluded_for_query220,
    }

    json_paths = sorted(p for p in results_dir.rglob("*.json") if p.is_file())
    excluded_names = excluded_globally | excluded_by_query.get(query_name, set())
    if excluded_names:
        json_paths = [p for p in json_paths if p.name not in excluded_names]
    if not json_paths:
        return "- No JSON files found under `results/`."

    blocks: list[str] = []
    total_chars = 0
    omitted: list[str] = []
    for path in json_paths:
        rel = path.relative_to(repo_root)
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            body = f"<read error: {exc}>"
        truncated = False
        if per_file_max_chars is not None and len(body) > per_file_max_chars:
            half = per_file_max_chars // 2
            body = (
                body[:half]
                + "\n... [truncated: middle omitted] ...\n"
                + body[len(body) - (per_file_max_chars - half) :]
            )
            truncated = True
        if total_max_chars is not None and total_chars + len(body) > total_max_chars:
            omitted.append(str(rel))
            continue
        total_chars += len(body)
        flag = " [truncated]" if truncated else ""
        blocks.extend(
            [
                f"### File: `{rel}`{flag}",
                "```",
                body,
                "```",
            ]
        )
    if omitted:
        blocks.append(
            "- Omitted due to appendix character budget: "
            + ", ".join(f"`{name}`" for name in omitted)
        )
    return "\n\n".join(blocks)
