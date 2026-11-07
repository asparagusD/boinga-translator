---
name: boinga-workflow
description: The mandatory loop for every subchunk of the boinga-translator project. Use whenever the user asks you to "do subchunk", "start chunk", implement a feature, or change any file in this repo.
---
# boinga-workflow

Follow these steps in order for every subchunk. Do not skip steps.

1. **Orient**
   - Read docs/REPO_GRAPH.md (repo map) and docs/PROGRESS.md.
   - Confirm the chunk branch: `git branch --show-current` must be
     `chunk/C<n>-<slug>`. If you're on main, create the branch.
   - `git status` must be clean. If not, stop and ask.
2. **Plan (short)**
   - List the files you will create/modify and an estimate of non-test lines.
   - If >~150 lines of non-test code or >5 files: split into a/b/c
     sub-subchunks, show the split, and do only the first.
3. **Test first where possible**
   - Write or extend a pytest in tests/ that describes the behaviour, using
     tiny synthetic data (generate wavs with numpy + soundfile in tmp_path).
4. **Implement** the smallest code that makes the test pass.
   - Module docstring on line 1. Type hints. Paths only via paths.py.
5. **Verify**
   - `ruff check .` and `pytest -q` must pass. Paste the real summary lines.
   - If the subchunk needs Colab, use the `colab-handoff` skill, then the
     `colab-verify` skill on the executed notebook before you commit results.
6. **Update the map and logs**
   - `python tools/repo_graph.py` (once it exists) to refresh docs/REPO_GRAPH.md.
   - Tick the subchunk in docs/PROGRESS.md. Add any decision to
     docs/DECISIONS.md (dated, one paragraph).
7. **Review**: ask the `code-reviewer` subagent to review `git diff --staged`.
   Fix blocking findings.
8. **Commit** using the `atomic-commit` skill. One commit.
9. **Report** to the user in ≤6 lines: what changed, test result, commit hash,
   next subchunk.

If any step fails, switch to the `debug-protocol` skill.
