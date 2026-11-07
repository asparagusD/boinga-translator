"""Tests for train_runner.check_loader and `boinga check-loader`: batch PASS/FAIL, fake coqui."""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from boinga_translator import cli, paths, train_config, train_runner

VOCAB = 10
SR = 22050


def _batch(xp: Callable[..., object] = np.asarray, **change: object) -> dict[str, object]:
    """A tiny VITS-like batch (2 clips) after format_batch_on_device; xp makes the arrays."""
    batch: dict[str, object] = {
        "tokens": xp(np.array([[1, 2, 3, 0], [4, 9, 0, 0]], dtype=np.int64)),
        "token_lens": xp(np.array([3, 2], dtype=np.int64)),
        "waveform": xp(np.zeros((2, 1, SR), dtype=np.float32)),
        "waveform_lens": xp(np.array([SR, SR // 2], dtype=np.int64)),
        "spec": xp(np.ones((2, 513, 86), dtype=np.float32)),
        "spec_lens": xp(np.array([86, 43], dtype=np.int32)),
        "raw_text": ["কথা", "আমি"],
        "speaker_ids": None,  # single speaker: format_batch sets these to None
        "d_vectors": None,
    }
    batch.update(change)
    return batch


def test_judge_ok_and_stats() -> None:
    lines, problems = train_runner.judge_batch(_batch(), VOCAB, SR)
    assert problems == []
    text = "\n".join(lines)
    assert "tokens: shape (2, 4) int64, ids 0..9 of vocab 10" in text
    assert "token_lens: 2..3" in text
    assert "seconds: 0.50..1.00" in text
    assert "spec: shape (2, 513, 86) float32, nan 0, inf 0" in text
    assert "spec_lens: shape (2,) int32" in text and "speaker_ids" not in text


@pytest.mark.parametrize(("change", "reason"), [
    ({"tokens": np.array([[1, 10]])}, "token id 10 >= vocab 10"),
    ({"tokens": np.array([[-1, 2]])}, "negative token id -1"),
    ({"waveform": np.array([[np.nan, 0.0]], dtype=np.float32)}, "waveform has 1 NaN/inf"),
    ({"spec": np.array([[np.inf, -np.inf]], dtype=np.float32)}, "spec has 2 NaN/inf"),
])
def test_judge_failures(change: dict[str, object], reason: str) -> None:
    _, problems = train_runner.judge_batch(_batch(**change), VOCAB, SR)
    assert reason in problems


def test_judge_empty_tokens() -> None:
    _, problems = train_runner.judge_batch(_batch(tokens=np.zeros((0, 0), np.int64)), VOCAB, SR)
    assert "empty tokens" in problems


def test_judge_missing_tokens() -> None:
    batch = _batch()
    del batch["tokens"]
    _, problems = train_runner.judge_batch(batch, VOCAB, SR)
    assert "batch has no tokens" in problems


def test_judge_torch_tensors() -> None:
    torch = pytest.importorskip("torch")
    assert train_runner.judge_batch(_batch(torch.as_tensor), VOCAB, SR)[1] == []
    bad = _batch(torch.as_tensor, waveform=torch.tensor([[float("nan")]]))
    assert "waveform has 1 NaN/inf" in train_runner.judge_batch(bad, VOCAB, SR)[1]


@pytest.fixture
def fake_coqui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Fake TTS modules (VitsConfig, Vits, load_tts_samples) and a config file in a fake repo."""
    monkeypatch.setattr(paths, "REPO", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / train_config.CONFIG_REL).write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {"batch": _batch()}

    class VitsConfig:
        audio = types.SimpleNamespace(sample_rate=SR)

        def load_json(self, path: str) -> None:
            seen["config_path"] = path

    class Vits:
        def __init__(self, config: object) -> None:
            self.config = config
            self.text_encoder = types.SimpleNamespace(
                emb=types.SimpleNamespace(num_embeddings=VOCAB))

        def get_data_loader(self, config: object, *, is_eval: bool = False,
                            samples: list | None = None, verbose: bool = True) -> list:
            seen["loader_kwargs"] = {"is_eval": is_eval, "samples": samples}
            return [dict(seen["batch"])]  # type: ignore[arg-type]

        def format_batch(self, batch: dict) -> dict:
            return batch

        def format_batch_on_device(self, batch: dict) -> dict:
            return batch

    def load_tts_samples(config: object, *, eval_split: bool = True) -> tuple[list, list]:
        seen["eval_split"] = eval_split
        return [{"text": "a"}] * 3, [{"text": "b"}]

    mods = {
        "TTS.tts.configs.vits_config": {"VitsConfig": VitsConfig},
        "TTS.tts.models.vits": {"Vits": Vits},
        "TTS.tts.datasets": {"load_tts_samples": load_tts_samples},
    }
    for name in ("TTS", "TTS.tts", "TTS.tts.configs", "TTS.tts.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "torch", None)  # check_loader must work without no_grad
    return seen


def test_check_loader_pass(fake_coqui: dict[str, object],
                           capsys: pytest.CaptureFixture[str]) -> None:
    assert train_runner.check_loader() is True
    out = capsys.readouterr().out
    assert out.startswith("get_data_loader(config: 'object', *, is_eval")  # signature first
    assert "samples: 3 train / 1 eval" in out and "PASS: loader ok" in out
    assert fake_coqui["eval_split"] is True
    assert fake_coqui["loader_kwargs"] == {"is_eval": False, "samples": [{"text": "a"}] * 3}
    assert str(fake_coqui["config_path"]).endswith("vits_ctg_finetune.json")


def test_check_loader_fail_and_cli(fake_coqui: dict[str, object],
                                   capsys: pytest.CaptureFixture[str]) -> None:
    fake_coqui["batch"] = _batch(tokens=np.array([[3, 12]]))
    assert train_runner.check_loader() is False
    assert "FAIL: loader token id 12 >= vocab 10" in capsys.readouterr().out
    assert cli.main(["check-loader"]) == 1
    fake_coqui["batch"] = _batch()
    assert cli.main(["check-loader"]) == 0


def test_cli_reports_exceptions(monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    def boom(config_path: Path | None = None) -> bool:
        raise ImportError("No module named 'TTS'")

    monkeypatch.setattr(train_runner, "check_loader", boom)
    assert cli.main(["check-loader"]) == 1
    assert "FAIL: loader ImportError: No module named 'TTS'" in capsys.readouterr().out


def test_signature_filter() -> None:
    def f(config: object, *, is_eval: bool = False, samples: object = None) -> None: ...

    kwargs = train_runner._loader_kwargs(inspect.signature(f), [1])
    assert kwargs == {"is_eval": False, "samples": [1]}
