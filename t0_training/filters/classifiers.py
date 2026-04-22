from __future__ import annotations

import urllib.request
from pathlib import Path

from . import FilterResult

MODEL_CACHE_DIR = Path.home() / ".cache" / "t0_training" / "filter_models"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MODEL = MODEL_CACHE_DIR / "lid.176.bin"
QC_MODEL = MODEL_CACHE_DIR / "dolma3_qc_model.bin"
TOPIC_MODEL = MODEL_CACHE_DIR / "weborganizer_model.bin"
QC_REPO = "allenai/dolma3-fasttext-quality-classifier"
TOPIC_REPO = "allenai/dolma3-fasttext-weborganizer-topic-classifier"


def _load_fasttext_model(path: Path):
    import fasttext

    return fasttext.load_model(str(path))


def _predict_label_prob(model, text: str, label: str, k: int = 10) -> float:
    labels, probs = model.predict(text.replace("\n", " "), k=k, threshold=0.0)
    return dict(zip(labels, probs)).get(label, 0.0)


def ensure_lid_model() -> Path | None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if LID_MODEL.exists():
        return LID_MODEL
    try:
        urllib.request.urlretrieve(LID_URL, LID_MODEL)
        return LID_MODEL
    except Exception:
        return None


def ensure_hf_model(local_path: Path, repo_id: str, candidates: tuple[str, ...]) -> Path | None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None

    for filename in candidates:
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODEL_CACHE_DIR)
            src = Path(downloaded)
            if src != local_path:
                src.rename(local_path)
            return local_path
        except Exception:
            continue
    return None


def run_classifiers(text: str) -> list[FilterResult]:
    out: list[FilterResult] = []

    lid_path = ensure_lid_model()
    if lid_path is None:
        out.append(FilterResult("lang_en", "SKIPPED", details="lid.176 model unavailable; run t0-filter-audit --download-models"))
    else:
        try:
            lid_model = _load_fasttext_model(lid_path)
            en_prob = float(_predict_label_prob(lid_model, text, "__label__en", k=10))
            out.append(FilterResult("lang_en", "PASS" if en_prob >= 0.65 else "FAIL", value=en_prob, threshold=">=0.65"))
        except Exception as e:
            out.append(FilterResult("lang_en", "SKIPPED", details=f"failed loading model: {e}"))

    qc_path = ensure_hf_model(QC_MODEL, QC_REPO, ("model.bin", "dolma3_qc_model.bin"))
    if qc_path is not None:
        try:
            qc_model = _load_fasttext_model(qc_path)
            hq = float(_predict_label_prob(qc_model, text, "__label__hq", k=10))
            out.append(FilterResult("quality_score", "INFO", value=hq, details="__label__hq"))
        except Exception as e:
            out.append(FilterResult("quality_score", "SKIPPED", details=f"failed loading model: {e}"))
    else:
        out.append(FilterResult("quality_score", "SKIPPED", details="quality model not found; run t0-filter-audit --download-models"))

    topic_path = ensure_hf_model(TOPIC_MODEL, TOPIC_REPO, ("model.bin", "weborganizer_model.bin"))
    if topic_path is not None:
        try:
            model = _load_fasttext_model(topic_path)
            labels, probs = model.predict(text.replace("\n", " "), k=5, threshold=0.0)
            top_label = labels[0] if labels else ""
            top_prob = float(probs[0]) if probs else 0.0
            out.append(FilterResult("topic", "INFO", value=top_label, details=f"p={top_prob:.4f}"))
        except Exception as e:
            out.append(FilterResult("topic", "SKIPPED", details=f"failed loading model: {e}"))
    else:
        out.append(FilterResult("topic", "SKIPPED", details="topic model not found; run t0-filter-audit --download-models"))

    return out


def gzip_ratio(text: str) -> float:
    import gzip

    encoded = text.encode("utf-8")
    comp_len = len(gzip.compress(encoded))
    raw_len = max(1, len(encoded))
    return comp_len / raw_len


def gzip_compressibility_result(text: str) -> FilterResult:
    return FilterResult("gzip_compressibility", "INFO", value=gzip_ratio(text))
