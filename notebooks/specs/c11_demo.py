"""Spec: C11 run 1, Gradio demo: automated edge-case checks, then the live demo and your notes."""

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C11",
    run=1,
    slug="demo",
    title="C11 — demo",
    branch="chunk/C11-demo",
    gpu=True,
    cells=[
        md("""
            **Automated demo checks.** `tools/demo_selfcheck.py` loads both voices (fine-tuned
            `boinga` from Drive `best/`, and the pretrained `baseline`) and calls the same
            button handler the demo page uses, without opening the page: the first held-out
            eval prompt in both voices, empty text, 400 characters, the prompt with Latin
            letters added and with Bengali digits added (tqdm bar; loading the models takes a
            little while). Expect one `CASE <name>: ok <seconds> s` or `CASE <name>: error
            <message>` line per case: audio for the two normal cases, the friendly messages
            "Please enter some Bengali text." and "Please keep it under 300 characters." for
            the empty and long ones, "The voice cannot say these characters, please remove
            them: h, e, l, o, w, r, d" for the Latin letters, audio or a message for the
            digits, then `PASS: demo selfcheck (6 cases)`.
        """),
        code("C11.2.check", "Automated demo checks", "!python tools/demo_selfcheck.py",
             expect=[r"PASS: demo selfcheck"]),
        md("""
            **Live demo.** The next cell keeps running while the demo is live. Open the share
            link (`Running on public URL: https://....gradio.live`), try it: the examples, your
            own sentences, both voices. Anyone with the link can use it while the cell runs.
            Then press Stop on the cell, fill the form below and (re)run the form cell
            (Shift+Enter): running it after you fill it is what saves your notes. Errors you
            trigger on the page (empty text, too long) show there
            only; a `Traceback` in this cell's output means a real failure.
        """),
        code("C11.2.live", "Live demo (stop the cell when done)", """
            from boinga_translator.app_gradio import build_app
            from boinga_translator.synth import VoiceBank
            build_app(VoiceBank()).launch(share=True, debug=True)
        """, expect=[], allow_interrupt=True),
        md("""
            **Your observations — demo.** `DEMO_WORKED`: did the page load and speak your text
            in both voices (yes / partly / no)? `NOTES`: what you tried and heard (pronunciation,
            errors, speed); write `none` if there is nothing to say. Both fields are required.
        """),
        form("C11-notes", "Your observations — demo", fields={
            "DEMO_WORKED": ["", "yes", "partly", "no"],
            "NOTES": "",
        }),
    ],
)
