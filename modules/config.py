import os

# -------- ストレージ設定 --------
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs")  # "gcs" / "local"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your_GCS_BUCKET_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


MODEL_PATHS = {
    "LinearRegression": f"models/math_predictor/{MODEL_VERSION}/LinearRegression.pkl",
    "LightGBM":        f"models/math_predictor/{MODEL_VERSION}/LightGBM.pkl",
}
FEATURE_PATH = f"models/math_predictor/{MODEL_VERSION}/feature_list.pkl"
LOCAL_ARTEFACT_DIR = os.getenv("ARTEFACT_DIR", "./artefacts")

# 評価データ（あればRMSE/R2算出に使用。なければスキップ）
EVAL_CSV_PATH = os.getenv("EVAL_CSV_PATH", f"models/math_predictor/{MODEL_VERSION}/eval.csv")

# -------- ログ/管理/メタ --------
ADMIN_PASS = os.getenv("ADMIN_PASS", "")          # 空ならゲート実質無効
LOG_BACKEND = os.getenv("LOG_BACKEND", "your_log_backend")   # "local" or "gcs"
GCS_EVENTS_PATH = os.getenv("GCS_EVENTS_PATH", "logs/events.csv")

ENV = os.getenv("ENV", "dev")
REGION = os.getenv("REGION", "asia-northeast1")
DEPLOYED_AT = os.getenv("DEPLOYED_AT", "")        # 例: "2025-11-10T23:45:00Z"

# -------- APIドキュメント --------
API_DOCS_URL = os.getenv("API_DOCS_URL", "http://localhost:8000/docs")  # FastAPI 併設時


