import os, joblib, pandas as pd
from functools import lru_cache
from .config import STORAGE_BACKEND, GCS_BUCKET, MODEL_PATHS, FEATURE_PATH, LOCAL_ARTEFACT_DIR, EVAL_CSV_PATH
import json

def _download_from_gcs(gcs_path: str, local_path: str) -> None:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(gcs_path).download_to_filename(local_path)

def _ensure_local(rel_path: str) -> str:
    if STORAGE_BACKEND == "gcs":
        local_dir = "/tmp/artefacts"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(rel_path))
        if not os.path.exists(local_path):
            _download_from_gcs(rel_path, local_path)
        return local_path
    else:
        return os.path.join(LOCAL_ARTEFACT_DIR, rel_path)

@lru_cache(maxsize=None)
def load_model(model_name: str):
    rel = MODEL_PATHS[model_name]
    local = _ensure_local(rel)
    return joblib.load(local)

@lru_cache(maxsize=None)
def load_feature_list():
    rel = FEATURE_PATH
    local = _ensure_local(rel)
    return joblib.load(local)

@lru_cache(maxsize=None)
def load_eval_df() -> pd.DataFrame | None:
    """評価用CSVがあれば返す（y列必須、特徴量列はfeature_listと同順/同名想定）"""
    try:
        local = _ensure_local(EVAL_CSV_PATH)
        if os.path.exists(local):
            return pd.read_csv(local)
    except Exception:
        pass
    return None

@lru_cache(maxsize=None)
def load_metrics_json():
    try:
        local = _ensure_local("models/math_predictor/v1/metrics.json")
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None