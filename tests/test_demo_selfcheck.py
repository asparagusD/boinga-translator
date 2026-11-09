"""Tests for tools/demo_selfcheck.py: case lines, PASS/FAIL rules, escaped exceptions (fake bank)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from boinga_translator import app_gradio

pytest.importorskip("gradio")  # the self-check calls the app's own gr.Error handler

SR = 22050
PROMPT = "বাক্য এক "


def _tool() -> types.ModuleType:
    import demo_selfcheck

    return demo_selfcheck


class FakeBank:
    """VoiceBank stand-in: a 1 s tone per call, or a chosen exception."""

    def __init__(self, voices: tuple[str, ...] = ("boinga", "baseline"),
                 fail: Exception | None = None, seconds: float = 1.0) -> None:
        self.voices, self.fail, self.seconds = list(voices), fail, seconds
        self.spoken: list[tuple[str, str]] = []

    def available(self) -> list[str]:
        return self.voices

    def speak(self, voice: str, text: str) -> tuple[int, np.ndarray]:
        self.spoken.append((voice, text))
        if self.fail is not None:
            raise self.fail
        return SR, (0.1 * np.ones(int(self.seconds * SR))).astype(np.float32)


def test_cases_cover_the_promptbook() -> None:
    cases = {c.name: c for c in _tool().cases(PROMPT)}
    assert list(cases) == ["normal-boinga", "normal-baseline", "empty", "too-long",
                           "latin-mixed", "bengali-digits"]
    assert cases["normal-boinga"].text == PROMPT  # verbatim, trailing space kept
    assert len(cases["too-long"].text) == 400
    assert cases["empty"].expect_error == app_gradio.MSG_EMPTY
    assert cases["too-long"].expect_error == app_gradio.MSG_TOO_LONG
    assert any("a" <= c <= "z" for c in cases["latin-mixed"].text)
    assert any("০" <= c <= "৯" for c in cases["bengali-digits"].text)


def test_pass(capsys: pytest.CaptureFixture[str]) -> None:
    bank = FakeBank()
    assert _tool().main(bank, prompt=PROMPT) == 0
    out = capsys.readouterr().out
    assert "CASE normal-boinga: ok 1.00 s" in out and "CASE normal-baseline: ok 1.00 s" in out
    assert f"CASE empty: error {app_gradio.MSG_EMPTY}" in out
    assert f"CASE too-long: error {app_gradio.MSG_TOO_LONG}" in out
    assert f"CASE latin-mixed: error {app_gradio.MSG_UNKNOWN}h, e, l, o, w, r, d" in out
    assert "CASE bengali-digits: ok" in out
    assert out.rstrip().endswith("PASS: demo selfcheck (6 cases)")
    assert ("baseline", PROMPT) in bank.spoken and len(bank.spoken) == 3  # bad input never spoken


def test_friendly_error_on_a_normal_case_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert _tool().main(FakeBank(fail=RuntimeError("cuda")), prompt=PROMPT) == 1
    out = capsys.readouterr().out
    assert "CASE normal-boinga: error Synthesis failed (RuntimeError)" in out
    assert "FAIL: demo selfcheck" in out and "PASS:" not in out


def test_short_audio_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert _tool().main(FakeBank(seconds=0.2), prompt=PROMPT) == 1
    assert "FAIL: demo selfcheck" in capsys.readouterr().out


def test_missing_voice_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert _tool().main(FakeBank(voices=("baseline",)), prompt=PROMPT) == 1
    assert "voice 'boinga' not available" in capsys.readouterr().out


def test_missing_prompts_file_fails_clearly(monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(app_gradio, "example_texts", lambda n=5: [])
    assert _tool().main(FakeBank()) == 1
    assert "FAIL: demo selfcheck: no prompt in configs/eval_prompts.json" in capsys.readouterr().out


def test_escaped_exception_is_a_fail_line(monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    def leaky(bank: object):  # a handler that lets a non-gr.Error out
        def handler(text: str, voice: str) -> None:
            raise KeyError("boom")
        return handler

    monkeypatch.setattr(app_gradio, "make_handler", leaky)
    assert _tool().main(FakeBank(), prompt=PROMPT) == 1
    out = capsys.readouterr().out
    assert "CASE empty: escaped KeyError" in out and "Traceback" not in out
