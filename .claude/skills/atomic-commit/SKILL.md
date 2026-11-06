---
name: atomic-commit
description: How to stage and commit in boinga-translator. Use every time you are about to run git commit, or when the user asks to commit, save progress, or write a commit message.
---
# atomic-commit

## Size
- One logical change per commit: one feature, one fix, or one refactor.
- Target ≤ ~150 lines of non-test code. If larger, split with
  `git add -p` into several commits, each passing tests.
- Never mix formatting-only changes with behaviour changes.

## Stage deliberately
- `git add <explicit paths>`. Never `git add -A` / `git add .`.
- Check `git diff --staged --stat` before committing. No wav/pth/zip files.

## Message format (enforced by .githooks/commit-msg)
```
<type>(<scope>): <imperative summary, ≤ 72 chars, no trailing period>

Chunk: C<n>.<m>[<letter>]
What:
- <concrete change 1, name files/functions>
- <concrete change 2>
Why:
- <reason, link to plan section or decision>
How verified:
- <exact command> -> <real result, e.g. "pytest -q -> 14 passed in 2.1s">
- <Colab cell / manual check and its observed output, if any>
Notes:
- <limitations, follow-ups, or "none">
```
- type ∈ feat, fix, refactor, test, docs, chore, perf, data, train, eval
- scope = the module or area (data-audit, text-frontend, train, repo-graph…)
- "How verified" must contain real output you saw, never "should work".

## Good vs bad
- Bad: `update files`, `fix bug`, `wip`, `more changes`
- Good: `feat(data-audit): compute per-clip duration, rms and clipping ratio`

## After commit
- `git log -1 --stat` and show the hash to the user.
- Push the branch: `git push -u origin HEAD` (if a remote exists).
