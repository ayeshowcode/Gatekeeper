---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git diff:*), Bash(git log:*)
description: Commit all uncommitted files one by one or in logical batches with self-explanatory messages
---

## Context

- Current git status: !`git status`
- Unstaged diff: !`git diff`
- Untracked files: !`git ls-files --others --exclude-standard`
- Recent commits (for style reference): !`git log --oneline -5`

## Your task

Commit every uncommitted file (modified, new, untracked) — but NEVER commit `docs/openai-code-prompts.md`.

**Grouping rule:** commit files together only if they contribute to the exact same goal and committing them separately would break the code or make no sense in isolation. Max 3 files per commit.

**Commit message format — strict rules:**
- Single line only. No newline, no body, no bullet points after the subject.
- Format: `<what you changed> to <why / the benefit>`
- The "what" names the specific thing changed. The "why" gives the concrete reason, not a vague label.
- Good: `use Redis TTL on session keys to evict stale memory automatically instead of growing unbounded`
- Good: `add PII regex to redact emails and phones before writing to logs to avoid storing personal data`
- Bad: `add security module` — no why
- Bad: `update utils and models` — vague, multi-thing
- Bad: `add xyz\n\nProvides schema validation...` — no body allowed
- Do NOT reference internal planning documents, phase numbers, or prompt numbers.
- Do NOT use "Co-Authored-By" or any trailer lines.

Work through all files sequentially. For each commit: run `git add <specific files>` then `git commit -m "..."`. Do not batch unrelated files just to go faster.
