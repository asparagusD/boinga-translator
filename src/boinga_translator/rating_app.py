"""Interactive blind rating in Colab: play clip1/clip2, score 1-5, save the sheet to Drive."""

from __future__ import annotations

import html
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from boinga_translator import colab_runtime
from boinga_translator.evaluate import CRITERIA, SHEET_COLS, blind_dir, ratings_dir

CLIPS = ("clip1", "clip2")
UNSET = "-"
SCORES = ("1", "2", "3", "4", "5")


def _slug(rater: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "", rater.strip().lower())
    if not slug:
        raise ValueError(f"rater name {rater!r} has no letters or digits")
    return slug


def sheet_path(rater: str) -> Path:
    """Drive ratings/<rater>.csv, the file `boinga eval-score` reads."""
    return ratings_dir() / f"{_slug(rater)}.csv"


def load_or_start(rater: str) -> pd.DataFrame:
    """This rater's saved sheet (to resume), else a fresh copy of blind/rating_sheet.csv."""
    path = sheet_path(rater)
    src = path if path.is_file() else blind_dir() / "rating_sheet.csv"
    sheet = pd.read_csv(src, dtype=str, keep_default_na=False).reindex(columns=SHEET_COLS,
                                                                       fill_value="")
    return sheet.assign(rater=_slug(rater))


def record(sheet: pd.DataFrame, prompt_id: str, clip: str, scores: Mapping[str, object],
           comment: str) -> pd.DataFrame:
    """A copy of the sheet with one clip's four scores (1-5 only) and comment filled in."""
    row = (sheet["prompt_id"] == prompt_id) & (sheet["clip"] == clip)
    if not row.any():
        raise KeyError(f"{prompt_id} {clip} is not in the rating sheet")
    out = sheet.copy()
    for c in CRITERIA:
        value = str(scores.get(c)).strip()
        if value not in SCORES:
            raise ValueError(f"{clip}: {c} must be 1-5, got {scores.get(c)!r}")
        out.loc[row, c] = value
    out.loc[row, "comments"] = " ".join(str(comment).split())
    return out


def _complete(sheet: pd.DataFrame) -> pd.Series:
    return sheet[CRITERIA].isin(SCORES).all(axis=1)


def progress(sheet: pd.DataFrame) -> tuple[int, int]:
    """(clips with all four scores, clips in the sheet)."""
    return int(_complete(sheet).sum()), len(sheet)


def next_prompt(sheet: pd.DataFrame) -> str | None:
    """The first prompt with an unscored clip, or None when every clip is rated."""
    todo = sheet.loc[~_complete(sheet), "prompt_id"]
    return None if todo.empty else str(todo.iloc[0])


def save(sheet: pd.DataFrame, rater: str) -> Path:
    """Write the sheet to Drive ratings/<rater>.csv in one step (never half-written)."""
    path = sheet_path(rater)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sheet.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)
    return path


def adopt_blind_sheet(rater: str) -> int:
    """Copy scores typed straight into blind/rating_sheet.csv to ratings/<rater>.csv.

    Only when the blind sheet has more rated clips than the rater's saved sheet; a saved
    sheet it replaces is kept as <rater>.csv.<time>.bak (not *.csv, so it is never scored).
    For one-person evaluation: checking a second rater would adopt the same scores again.
    Returns the number of rated clips copied (0 when nothing was copied).
    """
    src = blind_dir() / "rating_sheet.csv"
    if not src.is_file():
        return 0
    blind = pd.read_csv(src, dtype=str, keep_default_na=False).reindex(columns=SHEET_COLS,
                                                                       fill_value="")
    path = sheet_path(rater)
    have = progress(load_or_start(rater))[0] if path.is_file() else 0
    done = progress(blind)[0]
    if done <= have:
        return 0
    kept = ""
    if path.is_file():
        bak = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        os.replace(path, bak)
        kept = f" (previous sheet kept as {bak.name})"
    save(blind.assign(rater=_slug(rater)), rater)
    print(f"copied {done} rated clips from blind/rating_sheet.csv to {path}{kept}")
    return done


def check(rater: str) -> None:
    """PASS when every clip of the rater's saved sheet has four scores, else FAIL and stop.

    Scores typed into blind/rating_sheet.csv instead of the rating screen are adopted first.
    """
    adopt_blind_sheet(rater)
    path = sheet_path(rater)
    if not path.is_file():
        colab_runtime.fail(f"no ratings saved for {_slug(rater)!r} at {path}")
    sheet = load_or_start(rater)
    done, total = progress(sheet)
    if done < total:
        colab_runtime.fail(f"rated {done} of {total} clips (next: {next_prompt(sheet)})")
    print(f"PASS: rated {done} of {total} clips ({path})")


def rate(rater: str = "me") -> object:
    """The rating screen: one prompt at a time, saved to Drive after each prompt.

    Returns the widget box (header, the two clip columns, buttons, message); as a cell's last
    expression Colab shows it once. "Me" and "me" share one file (names are lower-cased).
    """
    import ipywidgets as w
    from IPython.display import Audio, display

    try:  # Colab needs this for newer ipywidgets; elsewhere it is not there
        from google.colab import output

        output.enable_custom_widget_manager()
    except ImportError:
        pass
    state = {"sheet": load_or_start(rater)}
    pids = list(dict.fromkeys(state["sheet"]["prompt_id"]))
    pos = {"i": pids.index(next_prompt(state["sheet"]) or pids[-1])}
    head, msg = w.HTML(), w.HTML()
    players = {c: w.Output() for c in CLIPS}
    pick = {(c, k): w.ToggleButtons(options=[UNSET, *SCORES], description=k,
                                    style={"description_width": "150px", "button_width": "36px"})
            for c in CLIPS for k in CRITERIA}
    notes = {c: w.Text(placeholder="comments (optional)") for c in CLIPS}
    back, ahead = w.Button(description="◀ Previous"), w.Button(description="Save & next ▶",
                                                                button_style="success")

    def show() -> None:
        sheet, pid = state["sheet"], pids[pos["i"]]
        rows = sheet[sheet["prompt_id"] == pid].set_index("clip")
        done, total = progress(sheet)
        head.value = (f"<h3>{pid} ({pos['i'] + 1} of {len(pids)}) · {done}/{total} clips rated"
                      f"</h3><p style='font-size:1.4em'>{html.escape(rows.iloc[0]['text'])}</p>")
        for c in CLIPS:
            players[c].clear_output()
            with players[c]:
                display(Audio(filename=str(blind_dir() / f"{pid}_{c}.wav")))
            for k in CRITERIA:
                pick[(c, k)].value = rows.at[c, k] if rows.at[c, k] in SCORES else UNSET
            notes[c].value = rows.at[c, "comments"]

    def save_next(_: object) -> None:
        pid, sheet = pids[pos["i"]], state["sheet"]
        try:
            for c in CLIPS:
                sheet = record(sheet, pid, c, {k: pick[(c, k)].value for k in CRITERIA},
                               notes[c].value)
        except ValueError as exc:
            msg.value = f"<b style='color:#c00'>{html.escape(str(exc))}</b>"
            return
        state["sheet"] = sheet
        save(sheet, rater)
        left = next_prompt(sheet)
        msg.value = f"saved {pid}" + ("" if left else ": all clips rated, run the next cell")
        last = pos["i"] == len(pids) - 1
        pos["i"] = pids.index(left) if last and left else min(pos["i"] + 1, len(pids) - 1)
        show()

    def go_back(_: object) -> None:
        pid, sheet = pids[pos["i"]], state["sheet"]
        try:  # keep this prompt's picks when they are complete
            for c in CLIPS:
                sheet = record(sheet, pid, c, {k: pick[(c, k)].value for k in CRITERIA},
                               notes[c].value)
            state["sheet"] = sheet
            save(sheet, rater)
            msg.value = f"saved {pid}"
        except ValueError:
            touched = any(p.value != UNSET for p in pick.values())
            msg.value = f"{pid} not saved (incomplete)" if touched else ""
        pos["i"] = max(0, pos["i"] - 1)
        show()

    back.on_click(go_back)
    ahead.on_click(save_next)
    columns = [w.VBox([w.HTML(f"<b>{c}</b>"), players[c], *(pick[(c, k)] for k in CRITERIA),
                       notes[c]]) for c in CLIPS]
    show()
    return w.VBox([head, w.HBox(columns), w.HBox([back, ahead]), msg])
