"""Tests for the public benchmark evidence."""

import json
from pathlib import Path

import pytest

from benchmarks.verify_report import verify_report


def test_trail_report_confusion_counts_reproduce_published_f1() -> None:
    summary = verify_report(Path("benchmarks/trail.json"))
    assert summary == {
        "archived_joint_accuracy": 0.599,
        "archived_mapped_annotations": 813,
        "calibration_overlap_traces": 144,
        "category_support": 808,
        "categories_verified": 14,
        "evaluation_scope": "archived_pisama_platform_run",
        "evidence_sha256_verified": True,
        "fn": 327,
        "fp": 0,
        "held_out": False,
        "macro_f1": 0.7535,
        "metadata_gap": 5,
        "micro_f1": 0.7463,
        "package_release_evaluated": None,
        "tp": 481,
    }


def test_trail_report_digest_prevents_unreviewed_evidence_changes(tmp_path: Path) -> None:
    source = Path("benchmarks/trail.json")
    altered = json.loads(source.read_text())
    altered["result"]["joint_accuracy"] = 1.0
    report = tmp_path / "trail.json"
    report.write_text(json.dumps(altered))

    with pytest.raises(ValueError, match="digest"):
        verify_report(report)


def test_trail_evidence_rejects_held_out_relabeling(tmp_path: Path) -> None:
    evidence = json.loads(Path("benchmarks/evidence.json").read_text())
    evidence["evaluation"]["held_out"] = True
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(ValueError, match="must not be labeled held out"):
        verify_report(Path("benchmarks/trail.json"), evidence_path)
