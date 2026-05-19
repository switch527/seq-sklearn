#!/usr/bin/env bash
set -euo pipefail

# Perf-baseline CI guard per Phase 11 PE.2 (mirrors check_snapshots.sh).
#   - Bot-authored PRs touching perf baselines fail unconditionally.
#   - Baseline files modified alongside non-baseline source files require
#     a `PERF_BASELINE_REVIEWED:` marker in at least one commit message.
# PR_USER_TYPE is injected by .github/workflows/pr.yml from
# github.event.pull_request.user.type (GitHub Actions does not expose
# $GITHUB_ACTOR_TYPE as a built-in env var).

BASELINE_GLOB="tests/perf/_baselines/"

# Determine base ref. On PR runs, GITHUB_BASE_REF is set. On push runs to
# main, fall back to comparing with the previous commit.
if [ -n "${GITHUB_BASE_REF:-}" ]; then
    BASE="origin/${GITHUB_BASE_REF}"
    git fetch origin "${GITHUB_BASE_REF}" --depth=1 || true
else
    BASE="HEAD~1"
fi

changed=$(git diff --name-only "${BASE}"...HEAD 2>/dev/null || git diff --name-only "${BASE}" HEAD)

if ! echo "$changed" | grep -q "^${BASELINE_GLOB}"; then
    echo "No perf baseline files changed; nothing to check."
    exit 0
fi

if [ "${PR_USER_TYPE:-}" = "Bot" ]; then
    echo "FAIL: bot-authored PR modifying perf baselines is not allowed"
    exit 1
fi

if echo "$changed" | grep -vq "^${BASELINE_GLOB}"; then
    commit_msgs=$(git log "${BASE}..HEAD" --format=%B 2>/dev/null || git log -1 --format=%B HEAD)
    if ! echo "$commit_msgs" | grep -q "^PERF_BASELINE_REVIEWED:"; then
        echo "FAIL: perf baseline files modified without PERF_BASELINE_REVIEWED: marker in any commit message"
        echo "Add a one-line human-written justification in a commit message prefixed with 'PERF_BASELINE_REVIEWED: '."
        exit 1
    fi
fi

echo "Perf baseline guard OK."
