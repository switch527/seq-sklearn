#!/usr/bin/env bash
set -euo pipefail

# Snapshot CI guard per A14 / N1.
#   - Bot-authored PRs touching snapshots fail.
#   - Snapshot files modified alongside non-snapshot source files require
#     a `SNAPSHOT_REVIEWED:` marker in at least one commit message of the PR.
# PR_USER_TYPE is injected by .github/workflows/pr.yml from
# github.event.pull_request.user.type (GitHub Actions does not expose
# $GITHUB_ACTOR_TYPE as a built-in env var).

SNAPSHOT_GLOB="tests/_snapshots/"

# Determine base ref. On PR runs, GITHUB_BASE_REF is set. On push runs to
# main, fall back to comparing with the previous commit.
if [ -n "${GITHUB_BASE_REF:-}" ]; then
    BASE="origin/${GITHUB_BASE_REF}"
    git fetch origin "${GITHUB_BASE_REF}" --depth=1 || true
else
    BASE="HEAD~1"
fi

changed=$(git diff --name-only "${BASE}"...HEAD 2>/dev/null || git diff --name-only "${BASE}" HEAD)

if ! echo "$changed" | grep -q "^${SNAPSHOT_GLOB}"; then
    echo "No snapshot files changed; nothing to check."
    exit 0
fi

if [ "${PR_USER_TYPE:-}" = "Bot" ]; then
    echo "FAIL: bot-authored PR modifying snapshots is not allowed"
    exit 1
fi

if echo "$changed" | grep -vq "^${SNAPSHOT_GLOB}"; then
    commit_msgs=$(git log "${BASE}..HEAD" --format=%B 2>/dev/null || git log -1 --format=%B HEAD)
    if ! echo "$commit_msgs" | grep -q "^SNAPSHOT_REVIEWED:"; then
        echo "FAIL: snapshot files modified without SNAPSHOT_REVIEWED: marker in any commit message"
        echo "Add a one-line human-written justification in a commit message prefixed with 'SNAPSHOT_REVIEWED: '."
        exit 1
    fi
fi

echo "Snapshot guard OK."
