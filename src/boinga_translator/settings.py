"""Pinned facts: pretrained model ids/URLs, dataset file + sha256, seed and inference scales."""

# Sources and dates: docs/DECISIONS.md "Pinned facts". Change a value only with a new entry there.

from __future__ import annotations

from typing import Final

PROJECT_NAME: Final[str] = "boinga-translator"

# Pretrained Bangla VITS (Coqui model registry, Idiap coqui-ai-TTS repo)
COQUI_TTS_VERSION: Final[str] = "0.28.0"
PRETRAINED_IDS: Final[dict[str, str]] = {
    "female": "tts_models/bn/custom/vits-female",
    "male": "tts_models/bn/custom/vits-male",
}
PRETRAINED_LICENSE: Final[str] = "Apache-2.0"
PRETRAINED_RELEASE_URLS: Final[dict[str, str]] = {
    "female": "https://github.com/coqui-ai/TTS/releases/download/v0.13.3_models/"
    "tts_models--bn--custom--vits_female.zip",
    "male": "https://github.com/coqui-ai/TTS/releases/download/v0.13.3_models/"
    "tts_models--bn--custom--vits_male.zip",
}

# Dataset (Mendeley Data files API)
DATASET_NAME: Final[str] = (
    "A Synchronized Chittagonian Speech Corpus for Dialect ASR and Speech-to-Speech Translation"
)
DATASET_DOI: Final[str] = "10.17632/n9xttk45df.1"
DATASET_LICENSE: Final[str] = "CC-BY-4.0"
DATASET_AUTHORS: Final[tuple[str, ...]] = ("Dipto Saha", "Md Ariful Islam", "Md. Milon Islam")
DATASET_ZIP_URL: Final[str] = (
    "https://data.mendeley.com/public-files/datasets/n9xttk45df/files/"
    "4b27c3de-f162-4028-a8aa-b7db252ece1e/file_downloaded"
)
DATASET_ZIP_BYTES: Final[int] = 843_601_272
DATASET_SHA256: Final[str] = "6ae668d22492f725cc3846e809788555d48b1b2c096af66cd8b521b71c0ccb0d"

# Reproducibility: one seed and fixed inference noise/length scales for all evaluation audio
SEED: Final[int] = 1234
INFER_NOISE_SCALE: Final[float] = 0.667
INFER_NOISE_SCALE_W: Final[float] = 0.8
INFER_LENGTH_SCALE: Final[float] = 1.0
