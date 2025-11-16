import streamlit as st
from modules.config import API_DOCS_URL

st.set_page_config(page_title="Edu ML / EBPM / CF Demo", layout="centered")
st.title("教育データ × ML × EBPM × CF デモ")

# APIドキュメントへのリンク（ある場合だけ表示）
if API_DOCS_URL:
    st.markdown(f"[APIドキュメント（FastAPI /docs）（準備中）]({API_DOCS_URL})")
else:
    st.caption("APIドキュメント（FastAPI /docs）は別途準備中です。")

st.markdown("""
**このデモで見せたい価値**
- **predicts（予測）**：Linear/LightGBM + SHAP。**A/Bテスト（コピー比較）**と**CTR計測**で「試作→計測→改善」を回す。
- **EBPM（エビデンスベース）**：t検定/PSM/IPW/回帰調整 で施策効果を“公平に”推定・可視化（観測データでの限界も明示）。
- **CF（推薦モデル）**：User-based CF と **Precision@K**（＋拡張でRecall@K）。基礎的な推薦と評価を最小構成で再現。
- **dashboard（運用ダッシュボード）**：predictにおける直近CTR/利用回数などの活用状況を可視化。
""")

st.caption("出典：公開データまたはダミー構成。PIIなし。デモ用途。")
st.info("左のページメニューから各デモを選択してください。")
