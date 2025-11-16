import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import timedelta
from modules.guards import admin_gate
from modules.log_utils import load_events_df
from modules.config import ENV, REGION, DEPLOYED_AT

st.title("運用ダッシュボード")
st.caption(
    "Problem → Hypothesis → Metric："
    "このAIデモ(predictsページ）がどれくらい使われているか・安定しているかを把握する → "
    "利用回数とCTR・エラー件数が許容範囲なら運用上の問題は小さいと考えられる → "
    "日別利用回数 / 7日間のCTR / 直近24hのエラー数をモニタリング"
)
with st.expander("イベントログのスキーマ / 例"):
    st.code("{timestamp, page, variant, predicted, clicked, user_session_id, event, payload}")

# 管理者ゲート
if not admin_gate():
    st.stop()

df = load_events_df()
if len(df) == 0:
    st.info("ログがありません。")
    st.stop()

# =========================
# 時刻処理：UTCのTimestampに統一
# =========================
# timestamp（ISO8601 + 'Z'）を tz=UTC の datetime64[ns, UTC] に変換
df["dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

st.write("直近イベント（先頭30件）")
df_safe = df.drop(columns=["payload", "user_session_id"], errors="ignore")
st.dataframe(df_safe.head(30))

st.divider()
st.subheader("KPI（直近7日）")

now = pd.Timestamp.now(tz="UTC")
week_ago = now - pd.Timedelta(days=7)
sub7 = df[df["dt"] >= week_ago].copy()
use_count = int((sub7["event"] == "prediction_shown").sum())

# CTR（predictページのcta_click）
clicks = sub7[(sub7["page"] == "predict") & (sub7["event"] == "cta_click")].copy()
if len(clicks) > 0:
    def _accepted(s):
        try:
            return bool(json.loads(s).get("accepted"))
        except Exception:
            return False

    clicks["accepted"] = clicks["payload"].apply(
        lambda s: _accepted(s) if isinstance(s, str) else False
    )
    ctr = clicks["accepted"].mean()
else:
    ctr = 0.0

c1, c2 = st.columns(2)
c1.metric("利用回数（7日）", f"{use_count}")
c2.metric("CTR（7日）", f"{ctr:.2%}")

# =========================
# 日別推移（利用回数）
# =========================
daily = sub7[sub7["event"] == "prediction_shown"].copy()
if len(daily) > 0:
    daily["date"] = daily["dt"].dt.date
    cnt = daily.groupby("date").size()
    fig, ax = plt.subplots()
    cnt.plot(kind="line", marker="o", ax=ax)
    ax.set_ylabel("count")
    ax.set_title("Daily Accesses")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.subheader("稼働メタ情報")

# 直近24h の error 件数
last24 = now - pd.Timedelta(hours=24)
mask_24h = df["dt"] >= last24
err24 = int(df[mask_24h & (df["event"] == "error")].shape[0])

st.write(f"- ENV：{ENV}")
st.write(f"- REGION：{REGION}")
st.write(f"- 最終デプロイ：{DEPLOYED_AT or '—'}")
st.write(f"- 直近24hのエラー件数：{err24}")
