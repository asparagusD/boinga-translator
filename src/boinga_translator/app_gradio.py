"""Gradio demo: type Chittagonian text, hear the fine-tuned (boinga) or the baseline voice."""

# Named app_gradio, never gradio.py (it would shadow the library). gradio is imported only
# inside build_app/make_handler, so the rest of the package and its tests run without it.

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from boinga_translator import data_audit, paths, settings, text_frontend, train_config

TITLE = "boinga-translator — Chittagonian TTS (proof of concept)"
DESCRIPTION = (
    "A single-speaker Chittagonian voice (**boinga**), fine-tuned from a pretrained Bangla VITS, "
    "next to that pretrained voice (**baseline**). This is a research demo, a proof of concept: "
    "the voice may mispronounce words, especially ones unlike the training sentences."
)
MAX_CHARS = 300
N_EXAMPLES = 5
EVAL_PROMPTS_FILE = "eval_prompts.json"
MSG_EMPTY = "Please enter some Bengali text."
MSG_TOO_LONG = f"Please keep it under {MAX_CHARS} characters."
MSG_NO_VOICE = "Please choose a voice."
MSG_UNKNOWN = "The voice cannot say these characters, please remove them: "
DEFAULT_VOICE = "boinga"

log = logging.getLogger(__name__)


class Bank(Protocol):
    """What the demo needs from synth.VoiceBank."""

    def available(self) -> list[str]: ...

    def speak(self, voice: str, text: str) -> tuple[int, np.ndarray]: ...


class DemoError(ValueError):
    """A message for the person using the demo (shown as a gr.Error, never a traceback)."""


def model_charset() -> frozenset[str]:
    """Characters the base voice (and so the fine-tuned one) knows, from pretrained facts."""
    return data_audit.load_model_charset(train_config.base_voice())


def check_input(text: str | None, charset: frozenset[str] | None = None) -> str | None:
    """Error message for text the demo will not synthesise, else None (edge spaces ignored).

    With a charset, characters the voice lacks after prepare_text (NFC + char_map, as the
    voice bank reads it) are listed: coqui would drop them with only a log warning.
    """
    stripped = (text or "").strip()
    if not stripped:
        return MSG_EMPTY
    if len(stripped) > MAX_CHARS:
        return MSG_TOO_LONG
    if charset is not None:
        prepared = text_frontend.prepare_text(stripped)
        if not prepared:  # e.g. only quotes or brackets, which char_map removes
            return MSG_EMPTY
        unknown = [c for c in prepared if c not in charset and not c.isspace()]
        if unknown:
            return MSG_UNKNOWN + ", ".join(dict.fromkeys(unknown))
    return None


def example_texts(path: Path | None = None, n: int = N_EXAMPLES) -> list[str]:
    """ctg_text (verbatim) of the first n held-out eval prompts; [] if the file is missing."""
    path = path or paths.configs_dir() / EVAL_PROMPTS_FILE
    if not path.is_file():
        return []
    try:
        return [row["ctg_text"] for row in json.loads(path.read_text(encoding="utf-8"))[:n]]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"bad prompts file {path}: {type(exc).__name__} {exc}") from exc


def footer_markdown() -> str:
    """Credits: the base models (coqui-tts) and the dataset, with their licences."""
    ids = ", ".join(f"`{i}`" for i in settings.PRETRAINED_IDS.values())
    authors = ", ".join(settings.DATASET_AUTHORS)
    return (
        f"Base models: {ids} by {settings.PRETRAINED_AUTHOR}, via coqui-tts "
        f"{settings.COQUI_TTS_VERSION} ({settings.PRETRAINED_LICENSE}).  \n"
        f"Training data: *{settings.DATASET_NAME}* by {authors}, "
        f"doi:[{settings.DATASET_DOI}](https://doi.org/{settings.DATASET_DOI}) "
        f"({settings.DATASET_LICENSE})."
    )


def synthesize(bank: Bank, text: str | None, voice: str | None,
               charset: frozenset[str] | None = None) -> tuple[int, np.ndarray]:
    """(sample_rate, wav) for the demo; every failure becomes a DemoError with a short message."""
    problem = check_input(text, charset)
    if problem:
        raise DemoError(problem)
    if not voice:
        raise DemoError(MSG_NO_VOICE)
    try:  # the text as typed: VoiceBank's prepare_text does the only normalisation
        return bank.speak(voice, text or "")
    except ValueError as exc:  # VoiceBank's own messages (unknown voice, empty after prep)
        raise DemoError(str(exc)) from exc
    except Exception as exc:  # anything else: log it here, show only its type
        log.exception("synthesis failed")
        message = f"Synthesis failed ({type(exc).__name__}). Please try other text."
        raise DemoError(message) from exc


def make_handler(bank: Bank, charset: frozenset[str] | None = None
                 ) -> Callable[[str, str], tuple[int, np.ndarray]]:
    """Button callback: synthesize(), with DemoError turned into gr.Error for the page.

    charset defaults to model_charset() (loaded once, when the handler is made).
    """
    import gradio as gr

    known = model_charset() if charset is None else charset

    def handler(text: str, voice: str) -> tuple[int, np.ndarray]:
        try:
            return synthesize(bank, text, voice, known)
        except DemoError as exc:
            # no traceback on the page, nor in the server log (the live Colab cell shows it)
            raise gr.Error(str(exc), print_exception=False) from None

    return handler


def build_app(bank: Bank, prompts_path: Path | None = None) -> Any:
    """The demo page (a gr.Blocks); ValueError if the bank has no voice available."""
    voices = bank.available()
    if not voices:
        raise ValueError("no voice available: best/ and the base voice need model.pth + "
                         "config.json")
    import gradio as gr

    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        text = gr.Textbox(label="Chittagonian text (Bengali script)", lines=3,
                          placeholder="Type a sentence, or pick an example below")
        voice = gr.Radio(choices=voices, label="Voice",
                         value=DEFAULT_VOICE if DEFAULT_VOICE in voices else voices[0])
        button = gr.Button("Synthesize", variant="primary")
        audio = gr.Audio(label="Speech", type="numpy", interactive=False)
        examples = example_texts(prompts_path)
        if examples:
            gr.Examples(examples=[[t] for t in examples], inputs=[text],
                        label="Held-out test sentences (never trained on)")
        gr.Markdown(footer_markdown())
        button.click(make_handler(bank), inputs=[text, voice], outputs=[audio])
    return app


def launch(share: bool = False) -> None:
    """Build the app on the real voice bank and serve it (blocks outside a notebook)."""
    from boinga_translator import synth

    build_app(synth.VoiceBank()).launch(share=share)
