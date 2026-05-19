"""PG.7 (qa-I5 / arch-C2): drive `scripts/check_perf_baselines.sh`
offline in a throwaway git repo over its four branches, including the
load-bearing unconditional bot-PR hard-fail.
"""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_perf_baselines.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "src.py").write_text("x = 1\n")
    (repo / "tests" / "perf" / "_baselines").mkdir(parents=True)
    (repo / "tests/perf/_baselines/cpu-x86.json").write_text("{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run(repo: Path, *, bot: bool = False) -> subprocess.CompletedProcess[str]:
    env = {"PATH": __import__("os").environ["PATH"]}
    if bot:
        env["PR_USER_TYPE"] = "Bot"
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )


def test_bot_authored_baseline_change_hard_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests/perf/_baselines/cpu-x86.json").write_text('{"a": 1}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "PERF_BASELINE_REVIEWED: even with marker")
    res = _run(repo, bot=True)
    assert res.returncode == 1
    assert "bot-authored PR modifying perf baselines is not allowed" in res.stdout


def test_baseline_plus_source_without_marker_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests/perf/_baselines/cpu-x86.json").write_text('{"a": 1}\n')
    (repo / "src.py").write_text("x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "no marker here")
    res = _run(repo)
    assert res.returncode == 1
    assert "without PERF_BASELINE_REVIEWED" in res.stdout


def test_baseline_with_marker_passes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests/perf/_baselines/cpu-x86.json").write_text('{"a": 1}\n')
    (repo / "src.py").write_text("x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "PERF_BASELINE_REVIEWED: re-measured on new torch")
    res = _run(repo)
    assert res.returncode == 0, res.stdout + res.stderr


def test_no_baseline_change_is_noop(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "src.py").write_text("x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "src only")
    res = _run(repo)
    assert res.returncode == 0
    assert "nothing to check" in res.stdout
