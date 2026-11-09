"""Colab check (C11.2): the demo's own handler on a normal prompt and on bad inputs, no UI."""

# Calls app_gradio.make_handler (the button callback build_app wires up) on the real voice
# bank. A normal prompt must give audio in both voices; empty and 400-character input must
# give the app's friendly message; Latin letters and Bengali digits may give audio or a
# friendly message (check_input does not judge them). Only gr.Error may leave the handler.

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from boinga_translator import app_gradio

VOICES = ("boinga", "baseline")
MIN_SECONDS = 0.5
LONG_CHARS = 400


@dataclass(frozen=True)
class Case:
    """One input; expect_error = the exact message, None = audio, "" = audio or any message."""

    name: str
    voice: str
    text: str
    expect_error: str | None


def cases(prompt: str) -> list[Case]:
    """The fixed case list around one held-out prompt (used verbatim)."""
    return [
        *(Case(f"normal-{v}", v, prompt, None) for v in VOICES),
        Case("empty", "boinga", "", app_gradio.MSG_EMPTY),
        Case("too-long", "boinga", "ক" * LONG_CHARS, app_gradio.MSG_TOO_LONG),
        Case("latin-mixed", "boinga", f"{prompt.strip()} hello world", ""),
        Case("bengali-digits", "boinga", f"{prompt.strip()} ১২৩", ""),
    ]


Handler = Callable[[str, str], tuple[int, np.ndarray]]


def run_case(handler: Handler, case: Case) -> tuple[bool, str]:
    """(passed, outcome text) for one case; an escaped exception is a failed case."""
    import gradio as gr

    try:
        rate, wav = handler(case.text, case.voice)
    except gr.Error as exc:
        message = str(exc.message)
        ok = case.expect_error == "" or case.expect_error == message
        return ok, f"error {message}"
    except Exception as exc:  # noqa: BLE001 - the point of the check: nothing else may escape
        return False, f"escaped {type(exc).__name__}: {exc}"
    seconds = wav.size / rate
    good = seconds > MIN_SECONDS and bool(np.isfinite(wav).all())
    ok = good and case.expect_error in (None, "")
    return ok, f"ok {seconds:.2f} s" + ("" if good else " (too short or not finite)")


def main(bank: app_gradio.Bank | None = None, prompt: str | None = None) -> int:
    """Run every case through the app's handler; print CASE lines and PASS/FAIL."""
    try:
        if bank is None:
            from boinga_translator import synth

            bank = synth.VoiceBank()
        missing = [v for v in VOICES if v not in bank.available()]
        if missing:
            print(f"FAIL: demo selfcheck: voice {missing[0]!r} not available")
            return 1
        if prompt is None:
            found = app_gradio.example_texts(n=1)
            if not found:
                print(f"FAIL: demo selfcheck: no prompt in configs/{app_gradio.EVAL_PROMPTS_FILE}")
                return 1
            prompt = found[0]
        handler = app_gradio.make_handler(bank)
        failed: list[str] = []
        all_cases = cases(prompt)
        for case in tqdm(all_cases, desc="demo selfcheck", unit="case"):
            ok, outcome = run_case(handler, case)
            print(f"CASE {case.name}: {outcome}")
            failed += [] if ok else [case.name]
    except Exception as exc:  # noqa: BLE001 - load errors are a FAIL line, not a traceback
        print(f"FAIL: demo selfcheck: {type(exc).__name__}: {exc}")
        return 1
    if failed:
        print(f"FAIL: demo selfcheck: {failed}")
        return 1
    print(f"PASS: demo selfcheck ({len(all_cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
