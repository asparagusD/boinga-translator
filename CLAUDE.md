# boinga-translator — rules for Claude Code

## Project
Fine-tune a pretrained Bangla VITS (coqui-tts==0.28.0, model
tts_models/bn/custom/vits-{male,female}, Apache 2.0) into a single-speaker
Chittagonian TTS proof of concept. Data: "A Synchronized Chittagonian Speech
Corpus" (Mendeley n9xttk45df v1, CC BY 4.0). Compute: free Colab T4. Budget $0.
Repo/project name: boinga-translator. Import package: boinga_translator.

## How we work (always)
- Use the Agent Skills in .claude/skills/. Before any task, check which skill
  applies. The `boinga-workflow` skill governs every subchunk.
- Before you plan, read docs/REPO_GRAPH.md (the repo map) and docs/PROGRESS.md.
- One subchunk = one commit. Follow the `atomic-commit` skill. Never batch
  unrelated changes. Never commit with a vague message.
- If a subchunk is over budget (>~150 lines non-test code, >5 files), split it
  into a/b/c before writing code and tell me.
- Any failure -> follow the `debug-protocol` skill. Log it in docs/DEBUG_LOG.md.
- Anything that must run on Colab -> follow the `colab-handoff` skill. You
  cannot see Colab; ask me to paste output.

## Code rules
- Logic lives in src/boinga_translator/*.py. Notebook cells only call functions.
- All paths come from src/boinga_translator/paths.py. No hard-coded paths.
- Every module starts with a one-line docstring describing its purpose.
- Never name a module tts.py, trainer.py, datasets.py, gradio.py, audio.py,
  config.py (they shadow libraries).
- Every data-handling function gets a pytest. Tests use tiny synthetic data,
  never the real corpus.
- Type hints on public functions. ruff must pass.

## Data rules (see `dialect-data-rules` skill)
- Never rewrite, translate or "standardise" Chittagonian transcripts.
- Unicode NFC is the only normalisation by default; any character mapping goes
  in configs/char_map.json and is logged in docs/DECISIONS.md.
- Resample audio to the pretrained config's sample rate; never change the
  model's sample rate.
- Never commit audio, checkpoints or the dataset. They live on Google Drive.

## Training rules (see `vits-finetune` skill)
- Restore from the pretrained checkpoint; resume with continue_path.
- Every long job is resumable and writes logs/checkpoints to Drive.
- Don't promise training times before the smoke test measures throughput.

## Never
- No paid APIs, services or Colab tiers (except I may use the pay as you go plan of colab to access a A100 GPU).
- No force-push to main. No editing of git history that's already pushed.
- No skipping the git hooks (--no-verify) unless I say so.
