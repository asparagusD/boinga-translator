"""Spec: C1 pilot run proving the notebook pipeline (env report, Drive write, a form)."""

from nbkit import Notebook, code, form

NOTEBOOK = Notebook(
    chunk="C1",
    run=1,
    slug="pilot",
    title="Pipeline pilot",
    branch="chunk/C1N-notebook-pipeline",
    cells=[
        code("C1.10.a", "Env report", """
            from boinga_translator import env_report

            env_report.run()
        """, expect=[r"PASS: env recorded"]),
        code("C1.10.b", "Drive write test", """
            import os

            from boinga_translator import paths

            probe = paths.drive_reports() / "pilot.txt"
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("boinga pilot\\n", encoding="utf-8")
            ok = probe.read_text(encoding="utf-8") == "boinga pilot\\n"
            mounted = os.path.ismount(paths.COLAB_DRIVE.parent)  # /content/drive
            on_drive = probe.resolve().is_relative_to(paths.COLAB_DRIVE)
            if ok and mounted and on_drive:
                print(f"PASS: drive ok ({probe})")
            else:
                print(f"FAIL: drive check failed (read back {ok}, Drive mounted {mounted}, "
                      f"under MyDrive {on_drive}): {probe}")
                raise SystemExit(1)
        """, expect=[r"PASS: drive ok"]),
        form("C1.10-notes", "Your observations", {"GPU_SHOWN": "", "ANY_PROBLEMS": "none"}),
    ],
)
