import pandas as pd
import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import joblib, io
from google.cloud import bigquery, storage
from sklearn.metrics import mean_squared_error, r2_score

# --- 設定値（あなたの環境に合わせて変更） ---
PROJECT_ID = "studentsperformance-472507"
BQ_TABLE_PATH = f"{PROJECT_ID}.StudentsPerformance0919.StudentsPerformanceTable"
GCS_BUCKET_NAME = "studentsperformance1007" 

# --- 1. BigQueryからデータを読み込む ---
client = bigquery.Client(project=PROJECT_ID)
sql = f"SELECT * FROM `{BQ_TABLE_PATH}`"
df = client.query(sql).to_dataframe()

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

final_columns
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

# --- 6. モデルと列情報とスコアを保存 ---
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

for name, model in models.items():
    joblib.dump(model, f"{name}.pkl")
    blob = bucket.blob(f"models/math_predictor/v1/{name}.pkl")
    blob.upload_from_filename(f"{name}.pkl")

joblib.dump(list(final_columns), "feature_list.pkl")
blob_list = bucket.blob(f"models/math_predictor/v1/feature_list.pkl")
blob_list.upload_from_filename('feature_list.pkl')

rmse_lr  = np.sqrt(mean_squared_error(y_test, lr_pred))
r2_lr    = r2_score(y_test, lr_pred)
rmse_lgb = np.sqrt(mean_squared_error(y_test, lgb_pred))
r2_lgb   = r2_score(y_test, lgb_pred)

metrics = {
    "LinearRegression": {"rmse": float(rmse_lr), "r2": float(r2_lr)},
    "LightGBM":        {"rmse": float(rmse_lgb), "r2": float(r2_lgb)}
}

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False)
bucket.blob("models/math_predictor/v1/metrics.json").upload_from_filename("metrics.json")
