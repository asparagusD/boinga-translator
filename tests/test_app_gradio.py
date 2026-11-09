"""Tests for app_gradio: input check, examples, footer, handler errors, Blocks (fake bank)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boinga_translator import app_gradio, cli, settings

SR = 22050


class FakeBank:
    """VoiceBank stand-in: fixed voices, records calls, can be told to fail."""

    def __init__(self, voices: list[str], fail: Exception | None = None) -> None:
        self.voices, self.fail = voices, fail
        self.calls: list[tuple[str, str]] = []

    def available(self) -> list[str]:
        return list(self.voices)

    def speak(self, voice: str, text: str) -> tuple[int, np.ndarray]:
        self.calls.append((voice, text))
        if self.fail is not None:
            raise self.fail
        return SR, np.zeros(SR, dtype=np.float32)


def _prompts(tmp_path: Path, n: int) -> Path:
    rows = [{"id": f"P{i:02d}", "clip_id": str(i), "ctg_text": f"বাক্য {i} ", "bn_text": "x",
             "has_reference": True} for i in range(1, n + 1)]
    path = tmp_path / "eval_prompts.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_check_input_empty(text: str | None) -> None:
    assert app_gradio.check_input(text) == "Please enter some Bengali text."


def test_check_input_length_limit() -> None:
    assert app_gradio.check_input("ক" * 300) is None
    assert app_gradio.check_input("ক" * 301) == "Please keep it under 300 characters."
    assert app_gradio.check_input("  " + "ক" * 300 + "  ") is None  # edge spaces do not count


def test_check_input_ok() -> None:
    assert app_gradio.check_input("হেঁততে উগ্গো পোয়া ছুটি আইসসি।") is None


def test_example_texts_first_five_verbatim(tmp_path: Path) -> None:
    texts = app_gradio.example_texts(_prompts(tmp_path, 7))
    assert texts == [f"বাক্য {i} " for i in range(1, 6)]  # trailing space kept: verbatim


def test_example_texts_missing_file(tmp_path: Path) -> None:
    assert app_gradio.example_texts(tmp_path / "nope.json") == []


def test_footer_names_models_dataset_and_licences() -> None:
    foot = app_gradio.footer_markdown()
    for needle in [*settings.PRETRAINED_IDS.values(), settings.PRETRAINED_LICENSE,
                   settings.PRETRAINED_AUTHOR, "coqui-tts", settings.DATASET_NAME,
                   settings.DATASET_DOI, settings.DATASET_LICENSE, *settings.DATASET_AUTHORS]:
        assert needle in foot


def test_synthesize_ok() -> None:
    bank = FakeBank(["boinga", "baseline"])
    rate, wav = app_gradio.synthesize(bank, " হেঁততে ", "baseline")
    assert rate == SR and wav.size == SR
    assert bank.calls == [("baseline", " হেঁততে ")]


def test_synthesize_rejects_bad_input_before_the_model() -> None:
    bank = FakeBank(["boinga"])
    with pytest.raises(app_gradio.DemoError, match="Please enter some Bengali text."):
        app_gradio.synthesize(bank, "  ", "boinga")
    assert bank.calls == []


def test_synthesize_no_voice_chosen() -> None:
    with pytest.raises(app_gradio.DemoError, match="Please choose a voice."):
        app_gradio.synthesize(FakeBank(["boinga"]), "ক", None)


def test_synthesize_bank_errors_become_messages() -> None:
    bank = FakeBank(["boinga"], fail=ValueError("text is empty after preparation: '“”'"))
    with pytest.raises(app_gradio.DemoError, match="empty after preparation"):
        app_gradio.synthesize(bank, "“”", "boinga")
    bank = FakeBank(["boinga"], fail=RuntimeError("CUDA out of memory /secret/path"))
    with pytest.raises(app_gradio.DemoError) as info:
        app_gradio.synthesize(bank, "ক", "boinga")
    assert "RuntimeError" in str(info.value) and "/secret/path" not in str(info.value)


def test_demo_cli_parses_share() -> None:
    assert cli.build_parser().parse_args(["demo", "--share"]).share is True
    assert cli.build_parser().parse_args(["demo"]).share is False


def test_demo_cli_launches(monkeypatch: pytest.MonkeyPatch,
                           capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(app_gradio, "launch", lambda share=False: seen.append(share))
    assert cli.main(["demo", "--share"]) == 0 and seen == [True]

    def no_voice(share: bool = False) -> None:
        raise ValueError("no voice available")

    monkeypatch.setattr(app_gradio, "launch", no_voice)
    assert cli.main(["demo"]) == 1
    assert "FAIL: demo ValueError: no voice available" in capsys.readouterr().out


def test_example_texts_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "eval_prompts.json"
    bad.write_text("[{\"id\": \"P01\"}]", encoding="utf-8")
    with pytest.raises(ValueError, match="eval_prompts.json"):
        app_gradio.example_texts(bad)


def test_build_app_needs_a_voice() -> None:
    with pytest.raises(ValueError, match="no voice"):
        app_gradio.build_app(FakeBank([]))


def test_build_app_blocks(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    app = app_gradio.build_app(FakeBank(["boinga", "baseline"]), _prompts(tmp_path, 6))
    assert isinstance(app, gr.Blocks)
    assert app.title == app_gradio.TITLE
    blocks = list(app.blocks.values())
    radio = next(b for b in blocks if isinstance(b, gr.Radio))
    assert [c[1] for c in radio.choices] == ["boinga", "baseline"] and radio.value == "boinga"
    assert any(isinstance(b, gr.Audio) for b in blocks)
    assert any(isinstance(b, gr.Button) for b in blocks)
    examples = next(b for b in blocks if isinstance(b, gr.Dataset))
    assert len(examples.samples) == 5
    texts = " ".join(b.value for b in blocks if isinstance(b, gr.Markdown))
    assert settings.DATASET_DOI in texts and "proof of concept" in texts


def test_handler_raises_gradio_error() -> None:
    gr = pytest.importorskip("gradio")
    handler = app_gradio.make_handler(FakeBank(["boinga"]))
    with pytest.raises(gr.Error, match="Please keep it under 300 characters.") as info:
        handler("ক" * 301, "boinga")
    assert info.value.print_exception is False  # no traceback in the live cell's log
    rate, wav = handler("ক", "boinga")
    assert rate == SR and wav.size == SR
