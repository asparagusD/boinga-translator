---
name: colab-handoff
description: How to hand work to Google Colab and take results back. Use when a step needs a GPU, the real dataset, Google Drive, or the pretrained checkpoint, or when the user pastes Colab output.
---
# colab-handoff

You cannot see Colab. The user is your hands there.

## Sending a step to Colab
1. Make sure the code is committed and pushed (Colab pulls from GitHub).
2. Give the user ONE code block to paste as a new cell, always starting with:
```python
   %cd /content/boinga-translator
   !git pull -q && git log -1 --oneline
```
   then one call into boinga_translator (no logic in the cell), and printed
   checks that end in a clear `PASS:` or `FAIL:` line.
3. Say what output you expect (shapes, counts, file paths, roughly how long).
4. Ask the user to paste the full output, including warnings.

## Receiving output
- Compare with what you expected. Quote the lines that matter.
- If there's any traceback or FAIL line, switch to `debug-protocol`.
- Only commit "How verified" lines that came from the pasted output.

## Colab facts to respect
- Free tier: GPU not guaranteed; session can drop anytime; ~12 h max.
- Read data from /content (fast); write checkpoints/logs to Drive (safe).
- Drive root: /content/drive/MyDrive/boinga-translator/
