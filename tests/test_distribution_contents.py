"""Exercise the real source builder with local build state present."""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def test_sdist_retains_source_and_evidence_but_excludes_local_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "dist", "build", "__pycache__", "*.egg-info",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage*",
        ),
    )
    for name in (".coverage", ".env", "untracked-local-output.txt"):
        (source / name).write_text("local build state, not release content\n")
    output = tmp_path / "artifacts"
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", str(output)],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list(output.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        names = {member.name.split("/", 1)[1] for member in archive if "/" in member.name}
    assert {".coverage", ".env", "untracked-local-output.txt"}.isdisjoint(names)
    assert {
        "LICENSE", "LICENSING.md", "README.md", "pyproject.toml",
        "src/pisama_detectors/py.typed", "src/pisama_detectors/_api.py",
        "benchmarks/evidence.json", "benchmarks/trail.json", "benchmarks/verify_report.py",
        "tests/test_benchmark_report.py", "typing_tests/public_api.py",
    } <= names
