---
name: dialect-data-rules
description: Rules for handling Chittagonian audio and transcripts in boinga-translator. Use for any code that reads, filters, normalises, splits or writes dataset audio or text, or that touches configs/char_map.json.
---
# dialect-data-rules

- The transcripts are Chittagonian written in Bengali script. They are the
  ground truth. Never rewrite, translate, spell-correct or "standardise" them.
- Keep the standard-Bangla translation column as metadata only. Never train
  on it.
- Default text normalisation = Unicode NFC + collapse repeated whitespace +
  strip. Nothing else unless a decision is logged.
- Characters not in the pretrained model's character set: report them with
  counts and examples first. A mapping (e.g. a variant form → its canonical
  codepoint) goes in configs/char_map.json with a comment for each entry, and a
  dated entry in docs/DECISIONS.md. Never silently drop characters.
- Audio: convert to mono, resample to the checkpoint's sample rate, peak-safe
  loudness normalisation, trim only leading/trailing silence (keep ≥ 100 ms).
- Exclusions are counted by reason (too short, too long, clipped, silent,
  empty text, OOV) and written to reports/.
- Splits: no transcript text appears in more than one split. Test split is
  fixed once and never trained on.
- Licence: CC BY 4.0. Every README/model card cites the dataset DOI
  10.17632/n9xttk45df.1 and its authors.
- Never commit audio or the dataset.
