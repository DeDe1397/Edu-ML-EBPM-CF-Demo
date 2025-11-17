import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from modules.config import API_DOCS_URL
from google.cloud import bigquery


st.title("EBPM：t検定 / PSM / IPW / 回帰調整")

# 共通ヘッダ
st.caption("**Problem → Hypothesis → Metric**：施策は成績を改善する？ → 交絡を統制すれば差が縮退しつつ有意性は保たれるはず → ATT / ATE とバランス指標（SMD）で確認")
st.markdown(f"[APIドキュメント（FastAPI /docs）（準備中）]({API_DOCS_URL})")
st.caption("出典：公開データ（BigQuery）or模したダミー。PIIなし。デモ用途。")

# ========= BigQuery =========
PROJECT_ID_DEFAULT = "your_PROJECT_ID_DEFAULT"
BQ_TABLE_DEFAULT = f"{PROJECT_ID_DEFAULT}.your_PATH"

@st.cache_data(show_spinner=False, ttl=600)
def load_students_performance_from_bq() -> tuple[pd.DataFrame, str]:
    """
    BigQueryから StudentsPerformance を取得。
    期待列（BQ側）: test_preparation_course, reading_score, writing_score, math_score
    戻り値 df_raw は既存処理と合わせるため、スペース入り名にrenameして返却。
    """
    bq_table = os.getenv("BQ_TABLE_PATH", BQ_TABLE_DEFAULT)
    gcp_project = os.getenv("GCP_PROJECT", PROJECT_ID_DEFAULT)

    client = bigquery.Client(project=gcp_project)
    query = f"""
    SELECT
      CAST(test_preparation_course AS STRING) AS test_preparation_course,
      CAST(reading_score AS FLOAT64)         AS reading_score,
      CAST(writing_score AS FLOAT64)         AS writing_score,
      CAST(math_score    AS FLOAT64)         AS math_score
    FROM `{bq_table}`
    """
    dfq = client.query(query).result().to_dataframe()

    df_raw = dfq.rename(columns={
        "test_preparation_course": "test preparation course",
        "reading_score": "reading score",
        "writing_score": "writing score",
        "math_score": "math score",
    })

    for c in ["reading score", "writing score", "math score"]:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    need = {"test preparation course","reading score","writing score","math score"}
    miss = need - set(df_raw.columns)
    if miss:
        raise ValueError(f"BigQueryテーブルに必要な列が不足: {miss}")

    return df_raw, bq_table


# ====================== データ選択 ======================
st.subheader("分析データの選択")

mode = st.radio(
    "データソースを選択",
    ["① 公開データ（BigQuery）", "② CSVアップロード"],
    index=0,
    horizontal=True
)
st.caption(
    "ローカルCSVのみで完結する構成にしつつ、"
    "クラウド環境があれば同じ分析をBigQueryテーブルからも実行できるようにしています。"
)

df = None
if mode.startswith("①"):
    st.info("公開データ（StudentsPerformanceInExams）は BigQuery から読み込んでいます。")
    try:
        df_raw, src_table = load_students_performance_from_bq()
    except Exception as e:
        st.error(f"BigQuery読み込みに失敗: {e}")
        st.stop()

    # treat=補習受講（completed=1）, motivation=reading score, baseline=writing score, y=math score
    df = pd.DataFrame({
        "treat": df_raw["test preparation course"].astype(str).str.strip().str.lower().eq("completed").astype(int),
        "motivation": df_raw["reading score"],
        "baseline":   df_raw["writing score"],
        "y":          df_raw["math score"],
    }).dropna(subset=["motivation","baseline","y"])

else:  
    st.info("アップロードCSVを使う場合は、必ず `treat`, `motivation`, `baseline`, `y` の4列を含めてください。（y が無い場合はデモ用に自動生成）")
    file = st.file_uploader("CSVファイルを選択", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in ["motivation","baseline","treat"] if c not in df.columns]
        if missing:
            st.error(f"次の列が不足しています: {missing}")
            st.stop()
        if "y" not in df.columns:
            st.warning("目的変数 y が存在しないため、ダミーで作成します。")
            rng = np.random.default_rng(42)
            df["y"] = 60 + 5*df["treat"] + 7*df["motivation"] + 0.4*df["baseline"] + rng.normal(0,5,len(df))
    else:
        st.stop()

# --- データ確認用（読み取ったデータのサンプル） ---
st.subheader(" 読み取ったデータの確認")
st.write("以下は分析に使用されるデータの先頭5行です。")
st.dataframe(df.head())

st.markdown(
    """
**カラムの意味**

- `treat` : 補習受講ダミー（1 = 補習を受けた, 0 = 受けていない）
- `motivation` : 学習意欲スコア（例：アンケート回答や読書量などを標準化した指標）
- `baseline` : 事前テスト（施策前）の成績
- `y` : 事後テスト（施策後）の数学スコア

ここでは「補習受講 (treat) がどれくらい y を押し上げているか」を、
t検定 / PSM / IPW / 回帰調整の4通りで推定します。
"""
)
st.caption(f"行数: {len(df):,}  |  列: {', '.join(df.columns)}")
st.divider()

# ====================== 分析 ======================

# t検定（単純平均差）
t_stat, pval = stats.ttest_ind(df.loc[df.treat==1,"y"], df.loc[df.treat==0,"y"])
diff_in_means = df.loc[df.treat==1,"y"].mean() - df.loc[df.treat==0,"y"].mean()

# 傾向スコア推定（motivation, baseline → treat）
X_ps = df[["motivation","baseline"]].values
lr = LogisticRegression(max_iter=1000).fit(X_ps, df["treat"])
ps = lr.predict_proba(X_ps)[:,1]

# PSM（1:1 最近傍）
sc = StandardScaler().fit(ps.reshape(-1,1))
emb = sc.transform(ps.reshape(-1,1))
tr_idx = np.where(df.treat==1)[0]
ct_idx = np.where(df.treat==0)[0]
nbrs = NearestNeighbors(n_neighbors=1).fit(emb[ct_idx])
_, nb = nbrs.kneighbors(emb[tr_idx])
matched_t_idx = tr_idx
matched_c_idx = ct_idx[nb.flatten()]
matched = pd.concat([df.iloc[matched_t_idx], df.iloc[matched_c_idx]], ignore_index=False).copy()

# マッチング後のPS
ps_matched_t = ps[matched_t_idx]
ps_matched_c = ps[matched_c_idx]

# PSM ATT
psm_att = matched.loc[matched.treat==1,"y"].mean() - matched.loc[matched.treat==0,"y"].mean()

# IPW
y_t = ( (df.loc[df.treat==1,"y"]/ps[df.treat==1]).sum() / (1/ps[df.treat==1]).sum() )
y_c = ( (df.loc[df.treat==0,"y"]/(1-ps[df.treat==0])).sum() / (1/(1-ps[df.treat==0])).sum() )
ipw_ate = y_t - y_c

# 回帰調整（重回帰： y ~ treat + motivation + baseline）
reg = LinearRegression().fit(df[["treat","motivation","baseline"]], df["y"])
reg_ate = float(reg.coef_[0])  # treat 係数

# SMD（標準化平均差） before/after（motivation, baseline）
def smd(a: pd.Series, b: pd.Series) -> float:
    mu_a, mu_b = a.mean(), b.mean()
    s = np.sqrt( (a.var(ddof=1)+b.var(ddof=1))/2 )
    return float((mu_a-mu_b)/s) if s>0 else 0.0

before = {
    "motivation": smd(df.loc[df.treat==1,"motivation"], df.loc[df.treat==0,"motivation"]),
    "baseline":   smd(df.loc[df.treat==1,"baseline"],   df.loc[df.treat==0,"baseline"]),
}
after = {
    "motivation": smd(matched.loc[matched.treat==1,"motivation"], matched.loc[matched.treat==0,"motivation"]),
    "baseline":   smd(matched.loc[matched.treat==1,"baseline"],   matched.loc[matched.treat==0,"baseline"]),
}

# ====================== 表示 ======================
c1,c2,c3,c4 = st.columns(4)
c1.metric("t検定：平均差", f"{diff_in_means:.2f}", help=f"t={t_stat:.2f}, p={pval:.3g}")
c2.metric("PSM：ATT", f"{psm_att:.2f}")
c3.metric("IPW：ATE", f"{ipw_ate:.2f}")
c4.metric("回帰調整：ATE(係数)", f"{reg_ate:.2f}")

st.subheader("バランス確認")
c4a,c4b = st.columns(2)
with c4a:
    st.write("SMD（Before/After）")
    smd_df = pd.DataFrame({"Before":before, "After":after})
    st.dataframe(smd_df.T.style.format("{:.2f}"))
with c4b:
    st.write("傾向スコアの重なり（マッチング前）")
    fig1, ax1 = plt.subplots()
    ax1.hist(ps[df.treat==1], bins=20, alpha=0.6, label="Treat")
    ax1.hist(ps[df.treat==0], bins=20, alpha=0.6, label="Control")
    ax1.set_xlabel("Propensity score"); ax1.set_ylabel("Count"); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

st.write("傾向スコアの重なり（マッチング後）")
fig2, ax2 = plt.subplots()
ax2.hist(ps_matched_t, bins=20, alpha=0.6, label="Treat (matched)")
ax2.hist(ps_matched_c, bins=20, alpha=0.6, label="Control (matched)")
ax2.set_xlabel("Propensity score"); ax2.set_ylabel("Count"); ax2.legend()
st.pyplot(fig2); plt.close(fig2)

st.subheader("効果推定の比較")
effect_labels = ["Diff-in-Means (t-test)", "PSM ATT", "IPW ATE", "Regression Adj. ATE"]
effect_values = [diff_in_means, psm_att, ipw_ate, reg_ate]
fig3, ax3 = plt.subplots()
ax3.bar(range(len(effect_values)), effect_values)
ax3.set_xticks(range(len(effect_values)))
ax3.set_xticklabels(effect_labels, rotation=15, ha="right")
ax3.set_ylabel("Estimated effect (y)")
st.pyplot(fig3); plt.close(fig3)

st.info("【前提・課題】：未観測変数によるバイアス、重み発散リスク、共通サポートの確認、モデルの適合性 等")
