#  Edu ML / EBPM / CF Demo
_教育データ × 機械学習 × EBPM × 推薦 × 運用サイクルの一気通貫デモ_

本リポジトリは、教育データを題材にした **機械学習・EBPM（因果推論）・推薦・A/B テスト・運用ダッシュボード** を  
ひとつの Streamlit アプリでまとめて再現したデモです。

- **予測・ダッシュボード**：ML モデル設計〜評価〜運用の一連の流れ
- **EBPM**：教育データ × 因果推論
- **推薦モデル**：推薦・ログ設計・改善サイクル

を意識して構成しています。

予測 → 提案 → クリックログ → ダッシュボード までを 1 つのアプリで完結
「精度の高いモデル」だけでなく、「改善サイクルを回せる状態」を示すことが目的です

## このデモで伝えたいこと
- モデリング：Liner / LightGBM / SHAP で予測と解釈を両立
- EBPM：t検定 / PSM / IPW / 回帰調整 で教育施策の効果を多面的に推定
- 推薦：User-based CF + Precision/Recall@K でレコメンドの基礎と評価を再現
- 運用：A/B テスト / CTR ログ / ダッシュボードで改善サイクルを設計

**教育データから課題を見つけ、実装し、運用するところまで実装を再現したモデルです**


## 🏗 構成概要

```text
app/
  app.py                      # ホーム（全体説明）
  modules/
    config.py                 # 設定（GCS 接続 / パス管理）
    model_io.py               # モデル・特徴量・SHAP
    metrics.py                # RMSE / R2 / Precision@K など
    log_utils.py              # A/B テスト用ログ / イベント記録
    ab_texts.py               # A/B テストで使うコピー文言
  pages/
    01_予測.py                 # 予測 + SHAP + A/B コピー + CTR 計測
    02_EBPM.py                # t検定 / PSM / IPW / 回帰調整 による施策効果推定
    03_推薦.py                # User-CF + Precision/Recall@K
    99_運用ダッシュボード.py   # CTR・利用回数などの可視化

ユーザー入力
    ↓
Streamlit UI(Predicts)
    ↓
Model Layer（LightGBM / Linear Regression / SHAP）
    ↓
予測値・SHAP値
    ↓
A/B コピー表示（学習提案）
    ↓
クリックログ（CTR）保存
    ↓
運用ダッシュボードで可視化

```

## 01_predicts（予測）ページ：回帰 + SHAP + A/B テスト

### 概要
- 教育データ（例：Kaggle公開データ(StudentsPerformanceInExams）を入力すると 数学スコアの予測 を実行
- モデルは Liner / LightGBM を切り替え可能
- SHAP による 特徴量ごとの寄与の可視化（ウォーターフォールなど）
- RMSE / R² をカード表示し、モデル性能を数値で明示
- 予測後に A/B コピー（学習提案メッセージ） をランダムに割り当て
- クリック/非クリックをログ（CTR 計測）

### 狙い
- モデル精度だけでなく、
- 「どんな提案メッセージがユーザー行動を変えるか」 を検証する流れを再現
- ログは log_utils.py 経由で保存し、運用ダッシュボードから確認可能

## 02_EBPM（エビデンスベース）ページ：t検定 / PSM / IPW / 回帰調整 による施策効果推定

### 概要
- 教育施策の「効いている / 効いていない」を、できるだけ公平に評価するためのページです。
- 実データは扱えないため、BigQuaryからKaggle公開データ(StudentsPerformanceInExams）を読み取る
- あるいはデモCSVを読み込む構成にしています。

### 実装している手法
- t検定
- 介入群 vs 非介入群の平均差をシンプルに比較
- PSM（Propensity Score Matching）
- 傾向スコアで近い生徒同士をマッチングし、バイアスを軽減
- IPW（Inverse Probability Weighting）「その生徒が介入群になった（ならなかった）確率」の逆数で重み付け
- 回帰調整で効果を検証
- マッチング前と後の重なりを可視化
- 3つの手法で推定された 平均処置効果（ATE） の比較
- 観測データでの限界（交絡・モデルミス指定）もテキストで明示

### 狙い
- 「ランダム化実験が難しい現場（学校・自治体） でどう分析するか」を示す

## 03_推薦ページ：User-based CF + Precision/Recall@K

### 概要
- ユーザー × アイテム行列から 類似ユーザー（cosine 類似度） を計算
- 類似ユーザーの高評価アイテムを元に User-based 協調フィルタリング を実行
- 上位 K 件（例：5件）の推薦を表示
- Precision/Recall@K を計算して、推薦の当たり具合を評価（評価用データを想定）

### 課題と改善アイデア（README で明示）
- コールドスタート問題（新規ユーザー / 新規アイテム）
- スケーラビリティ（ユーザー数・アイテム数の増加時）

### 改善案の例
- レコメンド用特徴量（ジャンル・難易度など）を足した Hybrid CF
- 近似最近傍探索（FAISS など）による高速化
- バッチ更新 + オフライン評価のパイプライン構築
    
### 狙い
- 基礎的なレコメンドの実装
- 教育データでどう活用するか指針にしたい

## 99_運用ダッシュボード：KPI 可視化

### 概要
このページでは、アプリ内部（predictsページ）で記録したログを元に簡易ダッシュボードを表示します。
- A/B コピーごとの CTR（クリック率）
- ページ別利用回数
- 直近 7 日のアクセス傾向
- モデルバージョンやリージョンなど、運用に関わるメタ情報（例）

### 狙い
- 「モデルを作って終わり」ではなく、
= 運用時にどの指標を追いかけるべきか を意識していることを示す

### Next ステップとしては
- BigQuery + Looker Studio での本格ダッシュボード化
- アラート／監視（CTR 急落・エラー増加など）の自動化


## 技術スタック
- 言語 / ライブラリ
- Python
- Streamlit
- scikit-learn / LightGBM
- pandas / numpy
- SHAP, statsmodels など

### インフラ想定
- GCP Cloud Run（コンテナデプロイ）
- BigQuery / Cloud Storage 連携

### ローカル開発（例）

※ 実際のバージョンやコマンドは環境に合わせて調整してください。

# 仮想環境作成
```bash
python -m venv .venv
source .venv/bin/activate  # Windows の場合: .venv\Scripts\activate
```

# 依存パッケージインストール
```bash
pip install -r requirements.txt
```

# Streamlit 起動
```bash
streamlit run app/app.py
```

# Cloud Run デプロイ（例）
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/edu-ml-demo
gcloud run deploy edu-ml-demo \
  --image gcr.io/PROJECT_ID/edu-ml-demo \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated
```

## 注意事項
- 学習済みモデルの非公開: 学習済みモデルファイル（LinearRegression.pkl, LightGBM.pkl, および feature_list.pkl）は、機密保持に従い、GitHub上では公開していません。
- データの前処理: BigQueryなどの環境で問題が発生しないよう、コード内では列名のスペースをアンダースコア (_) に変換する処理を行っています。
- PROJECT_IDなどはご自身の環境に応じて入力をしてください。
