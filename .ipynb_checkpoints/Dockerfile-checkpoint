# syntax=docker/dockerfile:1
FROM python:3.11-slim

# LightGBM の OpenMP ランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# 依存インストール（ビルドを安定させるためno-cache-dir）
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . /app

# Streamlit のCloud Run向け設定
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
