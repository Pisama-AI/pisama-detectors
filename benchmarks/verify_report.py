"""Verify the reproducible metrics in the checked-in TRAIL report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path(__file__).with_name("trail.json")
DEFAULT_EVIDENCE = Path(__file__).with_name("evidence.json")
TOLERANCE = 0.0001


def _close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise ValueError(f"{label}: expected {expected:.4f}, recomputed {actual:.4f}")


def _verify_evidence(path: Path, evidence_path: Path) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text())
    artifact = evidence["artifact"]
    evaluation = evidence["evaluation"]

    if artifact["path"] != path.name:
        raise ValueError(
            f"evidence artifact path is {artifact['path']!r}, report path is {path.name!r}"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError("TRAIL report digest does not match the reviewed evidence card")

    if evaluation["held_out"]:
        raise ValueError("archived TRAIL public-split run must not be labeled held out")
    if evaluation["scope"] != "archived_pisama_platform_run":
        raise ValueError("archived report must retain its platform-run scope")
    if evaluation["package_release_evaluated"] is not None:
        raise ValueError("archived platform run must not be attributed to a package release")
    if evaluation["traces"] != 148 or evaluation["calibration_overlap_traces"] != 144:
        raise ValueError("TRAIL trace count or calibration overlap changed without review")
    if evaluation["prediction_level_artifact_available"]:
        raise ValueError("archived report does not include prediction-level evidence")
    if evaluation["negative_candidate_set_available"]:
        raise ValueError("archived report does not include a negative candidate set")

    return evidence


def verify_report(
    path: Path = DEFAULT_REPORT,
    evidence_path: Path = DEFAULT_EVIDENCE,
) -> dict[str, Any]:
    """Recompute category and aggregate metrics from public confusion counts."""
    evidence = _verify_evidence(path, evidence_path)
    report = json.loads(path.read_text())
    categories = report["per_category_f1"]

    tp_total = 0
    fp_total = 0
    fn_total = 0
    f1_values: list[float] = []

    for name, metrics in categories.items():
        tp = int(metrics["tp"])
        fp = int(metrics["fp"])
        fn = int(metrics["fn"])
        support = int(metrics["support"])
        if support != tp + fn:
            raise ValueError(f"{name}: support does not equal tp + fn")

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        _close(precision, float(metrics["precision"]), f"{name} precision")
        _close(recall, float(metrics["recall"]), f"{name} recall")
        _close(f1, float(metrics["f1"]), f"{name} f1")

        tp_total += tp
        fp_total += fp
        fn_total += fn
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values)
    micro_f1 = 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    result = report["result"]
    category_support = tp_total + fn_total
    archived_mapped_annotations = int(result["mapped_annotations"])
    if result["run_id"] != evidence["artifact"]["run_id"]:
        raise ValueError("report run ID does not match the reviewed evidence card")
    _close(macro_f1, float(result["macro_f1"]), "macro_f1")
    _close(micro_f1, float(result["micro_f1"]), "micro_f1")

    if int(result["error_count"]) != 0:
        raise ValueError("archived run contains processing errors")
    if int(result["processed_traces"]) != int(result["total_traces"]):
        raise ValueError("archived run did not process every trace")

    return {
        "categories_verified": len(categories),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "category_support": category_support,
        "archived_mapped_annotations": archived_mapped_annotations,
        "metadata_gap": archived_mapped_annotations - category_support,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "archived_joint_accuracy": float(result["joint_accuracy"]),
        "evaluation_scope": evidence["evaluation"]["scope"],
        "held_out": evidence["evaluation"]["held_out"],
        "calibration_overlap_traces": evidence["evaluation"]["calibration_overlap_traces"],
        "package_release_evaluated": evidence["evaluation"]["package_release_evaluated"],
        "evidence_sha256_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    print(json.dumps(verify_report(args.report, args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
