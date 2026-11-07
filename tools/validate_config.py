"""Colab check (C7.1): coqui's VitsConfig loads configs/vits_ctg_finetune.json with every override."""

from __future__ import annotations

import json
import sys

from boinga_translator import paths, train_config

NOT_FIELDS = ("restore_path",)  # a TrainerArgs setting, not a VitsConfig field (coqpit drops it)
DATASET_KEYS = ("formatter", "path", "meta_file_train", "meta_file_val", "language")


def main() -> int:
    """Load with VitsConfig().load_json, check each override took effect, print PASS/FAIL."""
    path = paths.REPO / train_config.CONFIG_REL
    try:
        from TTS.tts.configs.vits_config import VitsConfig  # coqui only on Colab

        cfg = VitsConfig()
        cfg.load_json(str(path))
    except Exception as exc:  # noqa: BLE001 - coqpit raises many types on a bad field
        print(f"FAIL: coqui rejects {path}: {type(exc).__name__}: {exc}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    # coqpit ignores unknown keys, so a misspelled override would load silently: compare each.
    lost = [k for k in train_config.REASONS
            if k not in (*NOT_FIELDS, "datasets") and getattr(cfg, k, None) != data[k]]
    loaded = getattr(cfg, "datasets", None) or [None]
    lost += [f"datasets.0.{k}" for k in DATASET_KEYS
             if getattr(loaded[0], k, None) != data["datasets"][0][k]]
    print(f"sample_rate {cfg.audio.sample_rate}, batch_size {cfg.batch_size}, "
          f"lr_gen {cfg.lr_gen}, lr_disc {cfg.lr_disc}, mixed_precision {cfg.mixed_precision}")
    if lost:
        print(f"FAIL: coqui loaded the config but not these overrides: {lost}")
        return 1
    print("PASS: coqui accepts config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
