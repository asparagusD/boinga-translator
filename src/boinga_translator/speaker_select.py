"""Speaker selection: one speaker embedding per clean clip (resemblyzer, cached per clip)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boinga_translator import audit_summary, paths

ENCODER_FILE = "encoder.txt"  # which encoder filled the cache: never mix two in one folder
ERRORS_FILE = "errors.txt"
MAX_FAIL_RATIO = 0.01
RESEMBLYZER = "resemblyzer-0.1.4"
SB_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SB_RATE = 16_000  # the ECAPA model's rate; the corpus is 16 kHz mono (audit_summary.md)


@dataclass(frozen=True)
class Encoder:
    """A speaker encoder: a name for the cache marker and wav path -> 1-D embedding."""

    name: str
    embed: Callable[[str], np.ndarray]


def _resemblyzer() -> Encoder:
    from resemblyzer import VoiceEncoder, preprocess_wav  # webrtcvad needs pkg_resources

    enc = VoiceEncoder(verbose=False)  # cuda when available
    return Encoder(RESEMBLYZER, lambda wav: enc.embed_utterance(preprocess_wav(wav)))


def _speechbrain() -> Encoder:
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source=SB_SOURCE, savedir=str(paths.pretrained() / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device})

    def embed(wav: str) -> np.ndarray:
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        if sr != SB_RATE:
            raise ValueError(f"{wav}: {sr} Hz, the ECAPA encoder needs {SB_RATE}")
        signal = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        return model.encode_batch(signal).squeeze().cpu().numpy()

    return Encoder("speechbrain-ecapa-voxceleb", embed)


def load_encoder() -> Encoder:
    """resemblyzer if it imports, else SpeechBrain ECAPA (DECISIONS "Speaker encoder")."""
    try:
        return _resemblyzer()
    except ImportError as exc:
        print(f"WARN: resemblyzer unavailable ({exc}); falling back to {SB_SOURCE}")
    try:
        return _speechbrain()
    except ImportError as exc:
        raise RuntimeError(f"no speaker encoder: resemblyzer and speechbrain both fail to "
                           f"import ({exc}); pip install -e '.[speaker-fallback]'") from exc


def _npy(cache_dir: Path, clip_id: str) -> Path:
    if not clip_id or any(c in clip_id for c in "/\\") or clip_id in (".", ".."):
        raise ValueError(f"clip_id {clip_id!r} is not a safe file name")
    return cache_dir / f"{clip_id}.npy"


def _save(path: Path, vec: np.ndarray) -> None:
    """Write via a temp file so a dropped Drive/VM never leaves a half .npy behind."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, vec)
    os.replace(tmp, path)


def _claim(cache_dir: Path, name: str) -> None:
    marker = cache_dir / ENCODER_FILE
    if marker.exists() and (old := marker.read_text(encoding="utf-8").strip()) != name:
        raise RuntimeError(f"{cache_dir} holds {old} embeddings, not {name}: install {old} "
                           f"or move the folder away before re-embedding")
    if not marker.exists() and any(cache_dir.glob("*.npy")):
        raise RuntimeError(f"{cache_dir} has embeddings but no {ENCODER_FILE}: their encoder "
                           f"is unknown, move the folder away before re-embedding")
    marker.write_text(name + "\n", encoding="utf-8")


def embed(audit_df: pd.DataFrame, cache_dir: Path | None = None,
          encoder: Encoder | None = None, log_every: int = 500) -> pd.DataFrame:
    """Embed every clean clip once into cache_dir/<clip_id>.npy; returns the cached ones."""
    cache_dir = cache_dir or paths.speaker_emb()
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean = audit_df[audit_summary.clean_mask(audit_df)]
    todo = [r for r in clean.itertuples() if not _npy(cache_dir, str(r.clip_id)).exists()]
    failed: list[str] = []
    max_failed = MAX_FAIL_RATIO * len(clean)
    if todo:
        encoder = encoder or load_encoder()
        _claim(cache_dir, encoder.name)
        print(f"embedding {len(todo)} of {len(clean)} clean clips with {encoder.name}")
        for i, r in enumerate(todo, 1):
            try:
                vec = np.asarray(encoder.embed(str(r.wav_path)), dtype=np.float32).ravel()
                if vec.size == 0 or not np.isfinite(vec).all():
                    raise ValueError("empty or non-finite embedding")
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the rest
                failed.append(f"{r.clip_id}\t{type(exc).__name__}: {exc}")
                if len(failed) > max_failed:  # e.g. CUDA errors: stop now, not after 40 min
                    break
                continue
            _save(_npy(cache_dir, str(r.clip_id)), vec)
            if i % log_every == 0:
                print(f"{i}/{len(todo)} embedded")
    (cache_dir / ERRORS_FILE).write_text("".join(f"{e}\n" for e in failed), encoding="utf-8")
    done = clean.assign(npy=[_npy(cache_dir, str(c)) for c in clean["clip_id"]])
    done = done[[p.exists() for p in done["npy"]]][["clip_id", "split_official", "npy"]]
    if len(failed) > max_failed:
        raise RuntimeError(f"{len(failed)} of {len(clean)} clips failed (see "
                           f"{cache_dir / ERRORS_FILE}; first: {failed[0]})")
    marker = cache_dir / ENCODER_FILE
    name = marker.read_text(encoding="utf-8").strip() if marker.exists() else "no encoder"
    print(f"PASS: {len(done)} embeddings ({name}, {len(failed)} failed)")
    return done.reset_index(drop=True)
