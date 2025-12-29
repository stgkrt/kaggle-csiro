#!/bin/bash

# OOF Analyzer を起動するスクリプト

cd /kaggle

echo "Starting OOF Analyzer..."
streamlit run viz/oof_analyzer.py --server.port 8502 --server.address 0.0.0.0
