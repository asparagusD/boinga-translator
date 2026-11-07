"""Speaker selection: embed each clean clip's voice (cached per clip), cluster and rank voices."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boinga_translator import audit_summary, paths, report_io, settings

ENCODER_FILE = "encoder.txt"  # which encoder filled the cache: never mix two in one folder
ERRORS_FILE = "errors.txt"
MAX_FAIL_RATIO = 0.01
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)  # cosine distance, average linkage
MIN_MINUTES = 30.0  # the largest cluster must reach this for its silhouette to count
RANKING_REL = "reports/speaker_ranking.md"
TOP_IN_REPORT, IDS_PER_CLUSTER = 20, 5
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


def _load(emb_dir: Path, audit_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Clean clips that have an embedding, in clip order, and their L2-normalised vectors."""
    clean = audit_df[audit_summary.clean_mask(audit_df)]
    have = {p.stem for p in emb_dir.glob("*.npy")}
    meta = clean[clean["clip_id"].astype(str).isin(have)].sort_values("clip_id")
    meta = meta[["clip_id", "split_official", "duration_s"]].reset_index(drop=True)
    if len(meta) < 3:
        raise ValueError(f"{len(meta)} clean clips with an embedding in {emb_dir}: clustering "
                         "needs at least 3 (run `boinga embed-speakers` first)")
    x = np.stack([np.load(emb_dir / f"{c}.npy").ravel() for c in meta["clip_id"]])
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if bad := list(meta["clip_id"][~(np.isfinite(norm[:, 0]) & (norm[:, 0] > 0))]):
        raise ValueError(f"zero or non-finite embeddings for clips {bad[:5]}: delete their "
                         f".npy in {emb_dir} and re-run `boinga embed-speakers`")
    return meta, x / norm


def _rank(meta: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Relabel clusters 1..K by clean minutes (desc), ties by their first clip id."""
    df = meta.assign(raw=raw)
    order = (df.groupby("raw").agg(sec=("duration_s", "sum"), first=("clip_id", "min"))
             .sort_values(["sec", "first"], ascending=[False, True]))
    return df["raw"].map({r: i for i, r in enumerate(order.index, 1)}).to_numpy()


def _sweep(x: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Cluster at every threshold (the tree is built once, cached); one stats row each."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    rows, labels = [], {}
    with tempfile.TemporaryDirectory() as cache:
        for t in THRESHOLDS:
            raw = AgglomerativeClustering(n_clusters=None, distance_threshold=t,
                                          metric="cosine", linkage="average",
                                          memory=cache).fit_predict(x)
            labels[t] = _rank(meta, raw)
            k = int(raw.max()) + 1
            sil = silhouette_score(x, raw, metric="cosine") if 2 <= k < len(x) else np.nan
            top = meta["duration_s"][labels[t] == 1].sum() / 60
            rows.append({"threshold": t, "clusters": k, "largest_min": top, "silhouette": sil})
    return pd.DataFrame(rows), labels


def _choose(table: pd.DataFrame, min_minutes: float) -> float:
    """Best silhouette where the largest cluster has >= min_minutes, else most minutes."""
    ok = table[(table["largest_min"] >= min_minutes) & table["silhouette"].notna()]
    if len(ok):  # idxmax keeps the first (lowest threshold) on ties
        return float(ok.loc[ok["silhouette"].idxmax(), "threshold"])
    return float(table.loc[table["largest_min"].idxmax(), "threshold"])


def _render(table: pd.DataFrame, labels: pd.DataFrame, chosen: float, encoder: str,
            min_minutes: float) -> str:
    """reports/speaker_ranking.md: threshold table, then the top clusters by clean minutes."""
    lines =["# Speaker clusters", "",
             (f"Encoder: {encoder}. Clean clips with an embedding: {len(labels)}. "
              f"Agglomerative clustering, cosine distance, average linkage. Chosen threshold "
              f"{chosen:.2f}: best silhouette among thresholds whose largest cluster has "
              f">= {min_minutes:g} clean minutes, else the one with the most."), "",
             "| threshold | clusters | largest cluster min | silhouette | chosen |",
             "|---|---|---|---|---|"]
    lines += [f"| {r.threshold:.2f} | {r.clusters} | {r.largest_min:.2f} | {r.silhouette:.3f} "
              f"| {'yes' if r.threshold == chosen else ''} |" for r in table.itertuples()]
    lines += ["", f"## Clusters ranked by clean minutes (top {TOP_IN_REPORT})", "",
              "| rank | clips | clean min | train min | val min | test min | clip ids |",
              "|---|---|---|---|---|---|---|"]
    for k, g in list(labels.groupby("cluster"))[:TOP_IN_REPORT]:
        split = g.groupby("split_official")["duration_s"].sum() / 60
        ids = g["clip_id"].sample(min(IDS_PER_CLUSTER, len(g)), random_state=settings.SEED)
        lines.append(f"| {k} | {len(g)} | {g['duration_s'].sum() / 60:.2f} | "
                     + " | ".join(f"{split.get(s, 0.0):.2f}" for s in ("train", "val", "test"))
                     + f" | {', '.join(sorted(ids))} |")
    rest = labels[labels["cluster"] > TOP_IN_REPORT]
    if len(rest):
        lines.append(f"\n{rest['cluster'].nunique()} more clusters: {len(rest)} clips, "
                     f"{rest['duration_s'].sum() / 60:.2f} min.")
    return "\n".join(lines) + "\n"


def cluster(emb_dir: Path, audit_df: pd.DataFrame,
            min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    """Cluster voices over THRESHOLDS, keep the chosen one, save labels, write the ranking."""
    meta, x = _load(emb_dir, audit_df)
    table, all_labels = _sweep(x, meta)
    chosen = _choose(table, min_minutes)
    labels = meta.assign(cluster=all_labels[chosen], threshold=chosen)
    labels = labels.sort_values(["cluster", "clip_id"]).reset_index(drop=True)
    paths.speaker_clusters().parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(paths.speaker_clusters(), index=False)
    marker = emb_dir / ENCODER_FILE
    encoder = marker.read_text(encoding="utf-8").strip() if marker.exists() else "unknown"
    repo_path, _ = report_io.write_report(
        _render(table, labels, chosen, encoder, min_minutes), RANKING_REL)
    report_io.emit(repo_path, RANKING_REL)
    print(f"PASS: ranked {labels['cluster'].nunique()} clusters (threshold {chosen:.2f})")
    return labels


def candidates(n: int = 3, labels_csv: Path | None = None) -> list[int]:
    """The n clusters with the most clean minutes: cluster() already ranked them 1..K."""
    df = pd.read_csv(labels_csv or paths.speaker_clusters(), dtype={"clip_id": str})
    return [int(k) for k in sorted(df["cluster"].unique())[:n]]
