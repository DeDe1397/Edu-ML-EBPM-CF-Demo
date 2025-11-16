import json

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from modules.model_io import (
    load_model,
    load_feature_list,
    load_eval_df,
    load_metrics_json,
)
from modules.log_utils import assign_ab, log_event, load_events_df
from modules.ab_texts import A_COPY, B_COPY
from modules.metrics import rmse, r2
from modules.config import API_DOCS_URL


st.title("予測（Linear / LightGBM + SHAP）")

# --- 共通ヘッダ（Problem → Hypothesis → Metric + API / スキーマ / 但し書き） ---
st.caption(
    "**Problem → Hypothesis → Metric**："
    "成績(StudentsPerformanceInExams)予測に基づく学習提案の有効性を検証する → "
    "A/Bテスト（コピー比較）で行動が変わるなら改善余地あり → "
    "CTRと予測精度（RMSE・R²）で判断"
)
st.markdown(f"[APIドキュメント（FastAPI /docs）（準備中）]({API_DOCS_URL})")

with st.expander("イベントログのスキーマ / 例"):
    st.code(
        "{timestamp, page, variant, predicted, clicked, user_session_id, event, payload}"
    )
    example = {
        "timestamp": "...",
        "page": "predict",
        "variant": "A",
        "predicted": 68.2,
        "clicked": True,
        "user_session_id": "session-xxxx",
        "event": "cta_click",
        "payload": {"accepted": True},
    }
    st.code(json.dumps(example, ensure_ascii=False, indent=2), language="json")

st.caption("出典：公開データ(StudentsPerformanceInExams)のダミー構成。PIIなし。デモ用途。")


# =========================
# モデル・特徴量の読み込み
# =========================
@st.cache_resource
def get_artifacts(model_name: str):
    model = load_model(model_name)
    feature_list = load_feature_list()
    return model, feature_list


def create_background_data(feature_list, n: int = 100) -> pd.DataFrame:
    """SHAP用の背景データを簡易生成（ダミー）"""
    data = {}
    for f in feature_list:
        if f in ["reading_score", "writing_score"]:
            data[f] = np.random.uniform(50, 100, n)
        else:
            data[f] = np.random.choice([0, 1], n, p=[0.8, 0.2])
    return pd.DataFrame(data, columns=feature_list)


def encode_row(d: dict, feature_list: list[str]) -> pd.DataFrame:
    """ユーザー入力をOne-Hot形式にエンコード（学習時の列順に合わせる）"""
    X = pd.DataFrame(0, index=[0], columns=feature_list)
    X["reading_score"] = d["reading_score"]
    X["writing_score"] = d["writing_score"]
    for k, v in d.items():
        if k in ["reading_score", "writing_score"]:
            continue
        col = f"{k}_{v}"
        if col in X.columns:
            X[col] = 1
    return X


# モデル選択（表示用・メトリクス用）
model_choice = st.radio("モデル", ["LinearRegression", "LightGBM"])
model, feature_list = get_artifacts(model_choice)

# 入力フォーム
with st.form("in"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("gender", ["male", "female"])
        race = st.selectbox(
            "race/ethnicity",
            ["group A", "group B", "group C", "group D", "group E"],
        )
        edu = st.selectbox(
            "parental_level_of_education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        )
    with c2:
        lunch = st.selectbox("lunch", ["standard", "free/reduced"])
        prep = st.selectbox("test_preparation_course", ["none", "completed"])
        read = st.slider("reading_score", 0, 100, 70)
        write = st.slider("writing_score", 0, 100, 65)

    submitted = st.form_submit_button("予測")


# モデル精度（RMSE / R²）
st.subheader("モデル精度")
rmse_val = r2_val = None

mx = load_metrics_json()
if mx and model_choice in mx:
    rmse_val = mx[model_choice].get("rmse")
    r2_val = mx[model_choice].get("r2")
else:
    eval_df = load_eval_df()
    if eval_df is not None and "y" in eval_df.columns:
        X_eval = eval_df[feature_list]
        y_eval = eval_df["y"]
        y_hat = model.predict(X_eval)
        rmse_val, r2_val = rmse(y_eval, y_hat), r2(y_eval, y_hat)

c1, c2 = st.columns(2)
c1.metric("RMSE", f"{rmse_val:.3f}" if rmse_val is not None else "—")
c2.metric("R²", f"{r2_val:.3f}" if r2_val is not None else "—")

# 直近の予測結果 + SHAP + A/Bコピー + CTAログ
if submitted:
    d = {
        "gender": gender,
        "race/ethnicity": race,
        "parental_level_of_education": edu,
        "lunch": lunch,
        "test_preparation_course": prep,
        "reading_score": read,
        "writing_score": write,
    }
    X = encode_row(d, feature_list)
    y = float(model.predict(X)[0])

    st.session_state["last_pred"] = {
        "features": d,
        "y": y,
        "model_name": model_choice,
        "variant": assign_ab(),
        "impression_logged": False,
    }

if "last_pred" in st.session_state:
    pred = st.session_state["last_pred"]
    d = pred["features"]
    y = pred["y"]
    model_name = pred["model_name"]
    variant = pred["variant"]

    model_pred, feature_list_pred = get_artifacts(model_name)
    bg = create_background_data(feature_list_pred, 100)
    explainer = (
        shap.TreeExplainer(model_pred, bg)
        if model_name == "LightGBM"
        else shap.LinearExplainer(model_pred, bg)
    )
    st.success(f"**予測スコア：{y:.1f}**")
    if not pred.get("impression_logged", False):
        log_event(
            page="predict",
            event="prediction_shown",
            variant=variant,
            predicted=y,
            clicked=None,
            payload={"features": d},
        )
        pred["impression_logged"] = True
        st.session_state["last_pred"] = pred

    # SHAP Waterfall
    X_for_shap = encode_row(d, feature_list_pred)
    shap_values = explainer(X_for_shap)
    st.write("**予測の理由（SHAP Waterfall）**")
    st.caption(
        "SHAPの背景分布は、本番想定では学習データ（またはそのサンプル）から作成しますが、"
        "このデモでは特徴量の分布を模したダミーデータから生成しています。"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.plots.waterfall(shap_values[0, :], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)
    
    # A/Bコピー
    copy = A_COPY if variant == "A" else B_COPY
    st.divider()
    st.write("**推奨アクション（A/Bテスト中）**")
    st.info(copy)

    cta1, cta2, cta3 = st.columns(3)

    if cta1.button("この提案で進める"):
        log_event(
            page="predict",
            event="cta_click",
            variant=variant,
            predicted=y,
            clicked=True,
            payload={"accepted": True},
        )
        st.toast("記録しました（CTA受諾）📈")

    if cta2.button("別案が欲しい"):
        log_event(
            page="predict",
            event="cta_click",
            variant=variant,
            predicted=y,
            clicked=False,
            payload={"accepted": False},
        )
        st.toast("記録しました（CTA拒否）📝")

    if cta3.button("結果を共有"):
        log_event(
            page="predict",
            event="share",
            variant=variant,
            predicted=y,
            clicked=None,
            payload={},
        )
        st.toast("共有ログを記録しました 🔗")

# A/Bテストの途中経過（デモ集計）
with st.expander("A/Bテストの途中経過（デモ集計）"):
    df_log = load_events_df()
    df_log = df_log[df_log["page"] == "predict"].copy()
    clicks = df_log[df_log["event"] == "cta_click"].copy()

    if clicks.empty:
        st.write("まだデータがありません（CTAボタンが押されていません）。")
    else:
        clicks["clicked_bool"] = (
            clicks["clicked"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
        ctr = (
            clicks.groupby("variant")["clicked_bool"]
            .mean()
            .rename("CTR")
            .to_frame()
        )
        counts = (
            clicks.groupby("variant")["clicked_bool"]
            .count()
            .rename("n")
            .to_frame()
        )
        st.dataframe(ctr.join(counts).style.format({"CTR": "{:.2%}"}))

        st.caption(
            "CTR: clicked / 全CTAイベント数（variant別）。"
            "localログの場合はevents.csv、GCSの場合は設定されたパスに蓄積。"
        )

# モデルカード
with st.popover("モデルカード"):
    st.write("**モデル**：", model_choice)
    st.write("**学習日**：", "（メタがあれば表示）")
    st.write("**特徴量数**：", len(feature_list))
    st.write("**データ出典**：", "公開データ（ダミー）")
    st.write("**ランダムシード**：", 42)
    st.write("**既知の限界**：", "観測バイアス、適用範囲外への結果の当てはめ、説明変数の欠落 等")

    