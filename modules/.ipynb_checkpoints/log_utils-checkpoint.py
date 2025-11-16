import os, io, csv, json, uuid, datetime as dt
import pandas as pd
import streamlit as st
from modules.config import LOG_BACKEND, GCS_BUCKET, GCS_EVENTS_PATH
try:
    from google.cloud import storage
except:
    storage = None

def _ensure_user():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    return st.session_state["user_id"]

def assign_ab(key="copy_variant"):
    if key not in st.session_state:
        st.session_state[key] = "A" if uuid.uuid4().int % 2 == 0 else "B"
    return st.session_state[key]

def _append_local(row:dict, path="events.csv"):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

def _append_gcs(row:dict):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_EVENTS_PATH)
    try:
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))
    except:
        df = pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    blob.upload_from_file(io.BytesIO(buf.getvalue()), content_type="text/csv")

def log_event(page:str, event:str, payload:dict=None, variant:str=None):
    user = _ensure_user()
    row = {
        "ts": dt.datetime.utcnow().isoformat(),
        "user_id": user,
        "page": page,
        "event": event,
        "variant": variant or st.session_state.get("copy_variant",""),
        "payload": json.dumps(payload or {}, ensure_ascii=False)
    }
    if LOG_BACKEND=="gcs":
        _append_gcs(row)
    else:
        _append_local(row)

def load_events_df():
    if LOG_BACKEND=="gcs":
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_EVENTS_PATH)
        try:
            return pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        except:
            return pd.DataFrame(columns=["ts","user_id","page","event","variant","payload"])
    else:
        p="events.csv"
        if os.path.exists(p):
            return pd.read_csv(p)
        return pd.DataFrame(columns=["ts","user_id","page","event","variant","payload"])
