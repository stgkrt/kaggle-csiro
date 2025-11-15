#!/bin/bash

# Streamlit可視化アプリケーションを起動するスクリプト

echo "📊 CMI センサーデータ可視化ダッシュボードを起動中..."

# 必要なパッケージがインストールされているか確認
python3 -c "import streamlit, plotly, seaborn" 2>/dev/null || {
    echo "必要なパッケージをインストール中..."
    pip install streamlit plotly seaborn
}

# Streamlitアプリケーションを起動
cd /kaggle
echo "ブラウザで http://localhost:8501 を開いてください"
python3 -m streamlit run viz/visualize_dataset.py --server.port 8501 --server.address 0.0.0.0
