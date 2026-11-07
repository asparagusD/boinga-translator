"""Spec: C2 run 1, pretrained Bangla VITS voices: fetch, inspect and baseline audio."""

from nbkit import Notebook, code, form

NOTEBOOK = Notebook(
    chunk="C2",
    run=1,
    slug="pretrained-baseline",
    title="C2 — pretrained model and baseline",
    branch="chunk/C2-pretrained-baseline",
    gpu=True,
    cells=[
        code("C2.1", "Fetch pretrained voices", "!boinga fetch-pretrained --voice both",
             expect=[r"PASS: female cached", r"PASS: male cached"]),
        code("C2.2", "Inspect pretrained voices", "!boinga inspect-pretrained --voice both",
             expect=[r"PASS: inspected female", r"PASS: inspected male",
                     r"=== BEGIN configs/pretrained_facts.json ==="]),
        code("C2.4", "Baseline smoke synthesis", "!boinga baseline --voice both",
             expect=[r"PASS: baseline female", r"PASS: baseline male"]),
        code("C2.4.play", "Listen: baseline audio", """
            from boinga_translator import baseline

            baseline.play_all()
        """, expect=[r"AUDIO female STD_1", r"AUDIO male STD_1"]),
        form("C2.4-notes", "Your observations — baseline", fields={
            "FEMALE_STD_INTELLIGIBLE": ["", "yes", "partly", "no"],
            "MALE_STD_INTELLIGIBLE": ["", "yes", "partly", "no"],
            "ARTEFACTS": "",
            "PREFERRED_VOICE_QUALITY": ["", "female", "male", "same"],
        }),
    ],
)
