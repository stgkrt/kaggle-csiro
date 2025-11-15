#!/bin/bash
# Streamlit アプリケーション実行スクリプト

# 必要なパッケージをインストール
echo "Installing required packages..."
pip install streamlit plotly scikit-learn scipy seaborn

# Streamlit アプリケーションを起動
echo "Starting Streamlit app..."
cd /kaggle/viz
streamlit run compare_representive_waves.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
