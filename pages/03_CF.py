import numpy as np, pandas as pd, streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from modules.metrics import precision_at_k, recall_at_k
from modules.config import API_DOCS_URL

st.title("推薦：User-based CF と Precision/Recall@K")

# 共通ヘッダ
st.caption(
    "**Problem → Hypothesis → Metric**："
    "似たユーザーの嗜好からレコメンド → 近傍ユーザー数を調整すると精度が変わるはず → Precision/Recall@K（評価データ側）で確認"
)
st.markdown(f"[APIドキュメント（FastAPI /docs）（準備中）]({API_DOCS_URL})")
st.caption("出典：デモ用のダミーレーティングデータ。PIIなし。")

# =====================
# 1. データの確認
# =====================
st.subheader("1. デモ用の評価データ（ratings）")
ratings = pd.DataFrame({
    "userId": [
        # user 1
        *[1]*10,
        # user 2
        *[2]*10,
        # user 3
        *[3]*10,
        # user 4
        *[4]*10,
        # user 5
        *[5]*10,
        # user 6
        *[6]*10,
    ],
    "itemId": [
        # user 1: 10〜19
        10,11,12,13,14,15,16,17,18,19,
        # user 2: 11〜20
        11,12,13,14,15,16,17,18,19,20,
        # user 3: 12〜21
        12,13,14,15,16,17,18,19,20,21,
        # user 4: 13〜22
        13,14,15,16,17,18,19,20,21,22,
        # user 5: 14〜23
        14,15,16,17,18,19,20,21,22,23,
        # user 6: 15〜24
        15,16,17,18,19,20,21,22,23,24,
    ],
    "rating": [
        # user 1：前半アイテムほど高評価
        5,5,4,4,3,3,2,2,1,1,
        # user 2：ややフラット寄り
        5,4,4,3,3,2,2,2,1,1,
        # user 3：中〜後半寄りに評価
        3,3,4,4,5,5,3,3,2,2,
        # user 4：全体的に中評価
        3,3,3,4,4,3,3,2,2,2,
        # user 5：後半アイテムがやや高め
        2,2,3,3,4,4,5,5,3,3,
        # user 6：全体的に低め
        2,2,2,3,3,3,2,2,1,1,
    ]
})


st.subheader("推薦デモで使用する疑似データ")
st.dataframe(ratings)
st.caption("userId × itemId × rating の行列。ユーザーごとに評価パターンが少し異なるように固定で作成。")

# =====================
# 2. 学習データと評価データへの分割
# =====================
st.subheader("2. 学習データ / 評価データ（80/20分割）")

st.markdown(
    "- 各ユーザーごとに評価履歴（今回はitemId順）を並べて、**80%を学習(train)、残り20%を評価(test)** に使います。\n"
    "- 実運用では「時系列の後ろ側（最近の行動）」をテストにすることが多く、ここでは簡単のため `itemId` 順にソートして末尾を test にしています。"
)

# 80/20分割（ユーザー別に「簡易的な順序」の末尾20%をtest）
def split_train_test(df):
    train_list, test_list = [], []
    for uid, g in df.groupby("userId"):
        k = max(1, int(len(g)*0.2))
        # 本来は timestamp 等でソートすべきだが、ここでは簡単のため itemId 順
        g = g.sort_values("itemId")
        test_list.append(g.tail(k))
        train_list.append(g.iloc[:-k])
    return pd.concat(train_list), pd.concat(test_list)

train, test = split_train_test(ratings)

c1, c2 = st.columns(2)
with c1:
    st.write("**train（学習用）**")
    st.dataframe(train)
with c2:
    st.write("**test（評価用）**")
    st.dataframe(test)

st.caption(
    "train: 類似度計算とレコメンドの学習に使用\n"
    "test : Precision/Recall@K を計測するために使用（ユーザーごとの hold-out 評価）"
)

# =====================
# 3. User-based CF の準備（行列 & 類似度）
# =====================
st.subheader("3. User-based 協調フィルタリングの準備")

st.markdown(
    "- train データから **ユーザー×アイテム行列**（ピボットテーブル）を作ります。\n"
    "- ユーザー同士の類似度を **コサイン類似度** で計算し、近いユーザーを「近傍」として使います。"
)

mat = train.pivot_table(index="userId", columns="itemId", values="rating")
sim = cosine_similarity(mat.fillna(0))
sim_df = pd.DataFrame(sim, index=mat.index, columns=mat.index)

st.write("ユーザー×アイテム行列（train）")
st.dataframe(mat)

st.write("ユーザー間コサイン類似度（1.0 に近いほど嗜好が似ている）")
st.dataframe(sim_df)

# =====================
# 4. レコメンド関数（User-based CF）
# =====================
st.subheader("4. 近傍ユーザーからのレコメンド生成")

st.markdown(
    "選択したユーザーに対して、\n"
    "1. 類似度の高いユーザー上位 K 名（近傍）を選ぶ\n"
    "2. 近傍の評価の加重平均（類似度を重み）で、未評価アイテムのスコアを推定\n"
    "3. スコアの高い順に Top-N を推薦\n"
)

def predict_for(user_id, k=5, topn=5):
    # 近傍上位k
    if user_id not in sim_df.index:
        return []
    sim_vec = sim_df.loc[user_id].drop(user_id, errors="ignore")
    neighbors = sim_vec.sort_values(ascending=False).head(k).index

    # スコア = 近傍ユーザーの評価の類似度加重平均
    scores = {}
    for it in mat.columns:
        # 既にそのユーザーが評価済みのアイテムは除外
        if user_id in mat.index and it in mat.columns and pd.notna(mat.loc[user_id, it]):
            continue
        num, den = 0.0, 0.0
        for nb in neighbors:
            r = mat.loc[nb, it] if (nb in mat.index and it in mat.columns) else np.nan
            s = sim_df.loc[user_id, nb]
            if not np.isnan(r):
                num += s * r; den += abs(s)
        if den > 0:
            scores[it] = num / den
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

target = st.selectbox("評価対象ユーザー", sorted(ratings["userId"].unique()))
k = st.slider("近傍ユーザーK", 1, 10, 5)

recs = predict_for(target, k=k)
st.write("**推薦Top5（itemId, 推定スコア）**")
df_rec = pd.DataFrame(recs, columns=["itemId", "score"])
df_rec.index = np.arange(1, len(df_rec) + 1)
df_rec.index.name = "rank"
st.dataframe(df_rec)

st.caption(
    "・選択したユーザーと似ているユーザー上位K名から、未評価アイテムの推定スコアを計算し、スコア上位5件を表示しています。"
)

# =====================
# 5. 推薦の精度を評価
# =====================
st.subheader("5. Precision/Recall@K による評価（test側）")

st.markdown(
    "- train で学習したレコメンドを、**test データ上で評価**します。\n"
    "- 対象ユーザーの test に含まれるアイテムを「正解集合」とみなし、\n"
    "  Precision/Recall@K　を計算します。"
)

K_eval = st.slider("K（Precision/Recall評価用）", 1, 10, 5)
true_items = set(test.loc[test.userId == target, "itemId"])

if len(true_items) > 0:
    p_at_k = precision_at_k(true_items, recs, K_eval)
    r_at_k = recall_at_k(true_items, recs, K_eval)
else:
    p_at_k = 0.0
    r_at_k = 0.0

c1, c2 = st.columns(2)
c1.metric(
    f"Precision@{K_eval}",
    f"{p_at_k:.2f}",
    help="推薦上位K件のうち、正解アイテムが占める割合（K分のヒット数）。",
)
c2.metric(
    f"Recall@{K_eval}",
    f"{r_at_k:.2f}",
    help="正解アイテムのうち、推薦上位K件でどれだけ当てられたか（正解集合に対するヒット率）。",
)
st.caption(
    "trainでUser-CFモデルを作り、test側のアイテムを正解集合としたときの Precision/Recall@K を表示しています。"
)
