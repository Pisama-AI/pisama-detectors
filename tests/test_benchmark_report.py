"""Tests for the public benchmark evidence."""

from pathlib import Path

from benchmarks.verify_report import verify_report


def test_trail_report_confusion_counts_reproduce_published_f1() -> None:
    summary = verify_report(Path("benchmarks/trail.json"))
    assert summary == {
        "archived_joint_accuracy": 0.599,
        "archived_mapped_annotations": 813,
        "category_support": 808,
        "categories_verified": 14,
        "fn": 327,
        "fp": 0,
        "macro_f1": 0.7535,
        "metadata_gap": 5,
        "micro_f1": 0.7463,
        "tp": 481,
    }
