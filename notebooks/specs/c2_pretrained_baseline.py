"""Spec: C2 run 1, pretrained Bangla VITS voices: fetch, inspect and baseline audio."""

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C2",
    run=1,
    slug="pretrained-baseline",
    title="C2 — pretrained model and baseline",
    branch="chunk/C2-pretrained-baseline",
    gpu=True,
    cells=[
        md("""
            **Fetch the pretrained voices.** Downloads the female and male Bangla VITS models
            (~1 GB each) once and caches them on Drive; later runs reuse the cache. coqui shows
            a download progress bar (the GitHub-release fallback has none). Expect `PASS: female
            cached` and `PASS: male cached`.
        """),
        code("C2.1", "Fetch pretrained voices", "!boinga fetch-pretrained --voice both",
             expect=[r"PASS: female cached", r"PASS: male cached"]),
        md("""
            **Inspect the voices.** Reads each voice's config and checkpoint: sample rate,
            character set, discriminator weights. Writes `configs/pretrained_facts.json` and
            `reports/pretrained_inspect.md`. Expect `PASS: inspected female`, `PASS: inspected
            male` and the facts JSON block.
        """),
        code("C2.2", "Inspect pretrained voices", "!boinga inspect-pretrained --voice both",
             expect=[r"PASS: inspected female", r"PASS: inspected male",
                     r"=== BEGIN configs/pretrained_facts.json ==="]),
        md("""
            **Baseline synthesis.** Each voice reads the prompts in
            `configs/baseline_prompts.json` with a fixed seed; a progress bar counts the
            prompts. Each wav must be longer than 0.3 s and louder than -50 dBFS. Expect `PASS:
            baseline female` and `PASS: baseline male`.
        """),
        code("C2.4", "Baseline smoke synthesis", "!boinga baseline --voice both",
             expect=[r"PASS: baseline female", r"PASS: baseline male"]),
        md("""
            **Listen.** Audio players for every baseline wav, labelled with the prompt text.
            Nothing is checked automatically; your judgement goes in the form under "Your
            observations".
        """),
        code("C2.4.play", "Listen: baseline audio", """
            from boinga_translator import baseline

            baseline.play_all()
        """, expect=[r"AUDIO female STD_1", r"AUDIO male STD_1"]),
        md("""
            **Your observations — baseline.** Is each voice intelligible on standard Bangla,
            which artefacts did you hear, and which voice sounds better? Pick from the lists,
            then run this cell again.
        """),
        form("C2.4-notes", "Your observations — baseline", fields={
            "FEMALE_STD_INTELLIGIBLE": ["", "yes", "partly", "no"],
            "MALE_STD_INTELLIGIBLE": ["", "yes", "partly", "no"],
            "ARTEFACTS": "",
            "PREFERRED_VOICE_QUALITY": ["", "female", "male", "same"],
        }),
    ],
)
