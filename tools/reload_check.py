"""Colab check (C9.3): a fresh process loads the promoted best/ model and speaks 3 test rows."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import numpy as np
import soundfile as sf
from tqdm import tqdm

from boinga_translator import paths, prepare, synth

VOICE = "boinga"
N_ROWS = 3
MIN_SECONDS = 0.5
MIN_DBFS = -50.0


def test_rows(n: int = N_ROWS) -> list[tuple[str, str]]:
    """(clip_id, model text) of the first n rows of the prepared test manifest."""
    manifest = paths.data_prepared() / prepare.manifest_names()["test"]
    rows = [tuple(line.split("|")[:2]) for line in
            manifest.read_text(encoding="utf-8").splitlines() if line.strip()][:n]
    if len(rows) < n:
        raise ValueError(f"{manifest} has {len(rows)} rows, the check needs {n}")
    return [(str(cid), str(text)) for cid, text in rows]


def rms_dbfs(wav: np.ndarray) -> float:
    """RMS level in dBFS (full scale 1.0); -inf for silence or an empty wav."""
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) if wav.size else 0.0
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def main(bank: Any | None = None) -> int:
    """Speak the test rows with the promoted voice, check length + level, print PASS/FAIL."""
    try:
        out = synth.reload_dir()
        (out / synth.RELOAD_LIST).unlink(missing_ok=True)  # never play an older run's wavs
        if bank is None:
            bank = synth.VoiceBank()
        if VOICE not in bank.available():
            print(f"FAIL: reload check: no promoted model in {paths.best_ckpt()} "
                  "(run `boinga promote` first)")
            return 1
        rows: list[dict[str, Any]] = []
        bad: list[str] = []
        out.mkdir(parents=True, exist_ok=True)
        for i, (clip_id, text) in enumerate(tqdm(test_rows(), desc="reload check", unit="wav")):
            rate, wav = bank.speak(VOICE, text)
            path = out / f"{i}_{clip_id}.wav"
            sf.write(path, wav, rate)  # the raw model output, not peak-normalised
            seconds, level = wav.size / rate, rms_dbfs(wav)
            ok = seconds > MIN_SECONDS and level > MIN_DBFS and bool(np.isfinite(wav).all())
            print(f"{path.name}: {seconds:.2f} s, RMS {level:.1f} dBFS, "
                  f"{'ok' if ok else 'too short, quiet or not finite'}: {text}")
            rows.append({"wav": str(path), "clip_id": clip_id, "text": text,
                         "seconds": round(seconds, 3), "dbfs": round(level, 1) if math.isfinite(level) else None})
            bad += [] if ok else [path.name]
        (out / synth.RELOAD_LIST).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - any load/synthesis error is a FAIL line
        print(f"FAIL: reload check: {type(exc).__name__}: {exc}")
        return 1
    if bad:
        print(f"FAIL: reload check: {bad} failed the length (> {MIN_SECONDS} s), level "
              f"(> {MIN_DBFS} dBFS) or finite check")
        return 1
    print(f"PASS: reload check ({len(rows)} wavs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
