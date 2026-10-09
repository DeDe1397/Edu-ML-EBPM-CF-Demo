import os
import pandas as pd
import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import joblib

from config import MODEL_PATHS, FEATURE_PATH, LOCAL_ARTEFACT_DIR

# --- 設定値 ---
# Kaggle「Students Performance in Exams」のCSVをここに配置してください
# https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "data/StudentsPerformance.csv")

# --- 1. ローカルCSVからデータを読み込む ---
df = pd.read_csv(CSV_PATH)

# 列名のスペースをアンダースコアに変換（race/ethnicityはスラッシュを維持）
df = df.rename(columns={
    "parental level of education": "parental_level_of_education",
    "test preparation course": "test_preparation_course",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score",
})

# --- 2. 特徴量と目的変数を定義 ---
X = df.drop('math_score', axis=1)
y = df['math_score']

# --- 3. データ分割 ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --- 4. 前処理（One-Hot エンコーディング） ---
categorical_cols = X_train.select_dtypes(include=['object']).columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
final_columns = X_train_encoded.columns
models = {}

# --- 5. モデル学習と評価 ---
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train)
lr_pred = lr_model.predict(X_test_encoded)
models["LinearRegression"] = lr_model

lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train_encoded, y_train)
lgb_pred = lgb_model.predict(X_test_encoded)
models["LightGBM"] = lgb_model

# --- 6. モデルと列情報とスコアをローカルに保存 ---
for name, model in models.items():
    local_path = os.path.join(LOCAL_ARTEFACT_DIR, MODEL_PATHS[name])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    joblib.dump(model, local_path)

feature_local_path = os.path.join(LOCAL_ARTEFACT_DIR, FEATURE_PATH)
os.makedirs(os.path.dirname(feature_local_path), exist_ok=True)
joblib.dump(list(final_columns), feature_local_path)

rmse_lr  = np.sqrt(mean_squared_error(y_test, lr_pred))
r2_lr    = r2_score(y_test, lr_pred)
rmse_lgb = np.sqrt(mean_squared_error(y_test, lgb_pred))
r2_lgb   = r2_score(y_test, lgb_pred)

metrics = {
    "LinearRegression": {"rmse": float(rmse_lr), "r2": float(r2_lr)},
    "LightGBM":        {"rmse": float(rmse_lgb), "r2": float(r2_lgb)}
}

metrics_local_path = os.path.join(LOCAL_ARTEFACT_DIR, "models/math_predictor/v1/metrics.json")
os.makedirs(os.path.dirname(metrics_local_path), exist_ok=True)
with open(metrics_local_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False)

print(f"モデル・特徴量リスト・metricsを {LOCAL_ARTEFACT_DIR} に保存しました")
