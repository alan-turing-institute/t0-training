import sys
import types

import t0_training.filters.classifiers as classifiers
from t0_training.filters.classifiers import _predict_label_prob


class _FakeModel:
    def predict(self, text, k=10, threshold=0.0):
        assert "\n" not in text
        assert text == "hello world"
        return ["__label__en", "__label__fr"], [0.9, 0.1]


def test_predict_label_prob_preprocessing():
    model = _FakeModel()
    p = _predict_label_prob(model, "hello\nworld", "__label__en")
    assert p == 0.9


def test_ensure_lid_model_download_failure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(classifiers, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(classifiers, "LID_MODEL", tmp_path / "lid.176.bin")

    def _boom(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr(classifiers.urllib.request, "urlretrieve", _boom)
    assert classifiers.ensure_lid_model() is None


def test_ensure_hf_model_download_failure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(classifiers, "MODEL_CACHE_DIR", tmp_path)

    fake_hf = types.SimpleNamespace(hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    local_path = tmp_path / "missing-model.bin"
    out = classifiers.ensure_hf_model(local_path, "org/repo", ("model.bin", "fallback.bin"))
    assert out is None
