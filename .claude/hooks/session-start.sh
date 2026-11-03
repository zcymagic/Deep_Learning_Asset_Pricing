#!/bin/bash
# Reinstall the academic-research-skills (ARS) Claude Code skills in cloud sessions, whose home
# directory does not persist between sessions. Follows upstream's global install
# (docs/SETUP.md): full clone in ~/academic-research-skills (the skills read its shared/ and
# scripts/), the five skill folders copied to ~/.claude/skills, and the sr-screener reviewer
# agent to ~/.claude/agents. The plugin's hooks and slash commands are not installed.
#
# Pinned to a reviewed commit; to update, review the new upstream version and change ARS_COMMIT.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
	exit 0
fi

ARS_REPO="https://github.com/Imbad0202/academic-research-skills.git"
ARS_COMMIT="2e02041a3b821a160d98c9398f173c484ac759a8"	# v3.24.1
ARS_DIR="$HOME/academic-research-skills"
SKILLS="deep-research academic-paper academic-paper-reviewer academic-pipeline sr-screener"

if [ "$(git -C "$ARS_DIR" rev-parse HEAD 2>/dev/null || true)" != "$ARS_COMMIT" ]; then
	rm -rf "$ARS_DIR"
	git init -q "$ARS_DIR"
	git -C "$ARS_DIR" fetch -q --depth 1 "$ARS_REPO" "$ARS_COMMIT"
	git -C "$ARS_DIR" checkout -q FETCH_HEAD
fi

mkdir -p "$HOME/.claude/skills" "$HOME/.claude/agents"
for s in $SKILLS; do
	rm -rf "$HOME/.claude/skills/$s"
	cp -R "$ARS_DIR/$s" "$HOME/.claude/skills/$s"
done
cp "$ARS_DIR/sr-screener/agents/screening_reviewer_agent.md" "$HOME/.claude/agents/"
