import os, io, csv, json, uuid, datetime as dt
import pandas as pd
import streamlit as st
from modules.config import LOG_BACKEND, GCS_BUCKET, GCS_EVENTS_PATH
try:
    from google.cloud import storage
except Exception:
    storage = None


# ユーザーID（セッション単位）
def _ensure_user():
    if "user_session_id" not in st.session_state:
        st.session_state["user_session_id"] = str(uuid.uuid4())
    return st.session_state["user_session_id"]


# A/B割り当て（1セッションで固定）
def assign_ab(key: str = "copy_variant") -> str:
    if key not in st.session_state:
        st.session_state[key] = "A" if (uuid.uuid4().int % 2 == 0) else "B"
    return st.session_state[key]


# --- ローカル書き込み (events.csv) ---
def _write_local(row: dict):
    path = "events.csv"
    file_exists = os.path.exists(path)

    cols = [
        "timestamp",
        "page",
        "variant",
        "predicted",
        "clicked",
        "user_session_id",
        "event",
        "payload",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --- GCS書き込み ---
def _write_gcs(row: dict):
    if storage is None:
        return
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_EVENTS_PATH)
    cols = [
        "timestamp",
        "page",
        "variant",
        "predicted",
        "clicked",
        "user_session_id",
        "event",
        "payload",
    ]
    try:
        data = io.BytesIO(blob.download_as_bytes())
        df_existing = pd.read_csv(data)
    except Exception:
        df_existing = pd.DataFrame(columns=cols)

    df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    buf = io.StringIO()
    df_new.to_csv(buf, index=False)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")

def log_event(
    page: str,
    event: str,
    variant: str | None = None,
    predicted: float | None = None,
    clicked: bool | None = None,
    payload: dict | None = None,
) -> None:
    user = _ensure_user()
    # UTC ISO形式で "2025-11-11T00:00:00Z" みたいな形にする
    ts = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(timespec="seconds")
    ts = ts.replace("+00:00", "Z")

    row = {
        "timestamp": ts,
        "page": page,
        "variant": variant if variant is not None else "",
        "predicted": predicted if predicted is not None else "",
        "clicked": clicked if clicked is not None else "",
        "user_session_id": user,
        "event": event,
        "payload": json.dumps(payload or {}, ensure_ascii=False),
    }

    if LOG_BACKEND == "gcs" and storage is not None and GCS_EVENTS_PATH:
        _write_gcs(row)
    else:
        _write_local(row)


# --- 読み込み側 ---
def load_events_df() -> pd.DataFrame:
    cols = [
        "timestamp",
        "page",
        "variant",
        "predicted",
        "clicked",
        "user_session_id",
        "event",
        "payload",
    ]

    if LOG_BACKEND == "gcs" and storage is not None and GCS_EVENTS_PATH:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_EVENTS_PATH)
        try:
            df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        except Exception:
            df = pd.DataFrame(columns=cols)
    else:
        path = "events.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=cols)

    # 古いフォーマット（ts,user_id,page,event,variant,payload）との互換
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = df["ts"]
    if "user_session_id" not in df.columns and "user_id" in df.columns:
        df["user_session_id"] = df["user_id"]

    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ["clicked"] else False

    return df[cols]
