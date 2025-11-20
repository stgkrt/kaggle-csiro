# Biomass Data Viewer 🌿

牧草画像とバイオマス成分のメトリクスを視覚化するStreamlitアプリケーション

## 機能

### 1. 📸 画像詳細表示
- 訓練データ・テストデータの画像を閲覧
- バイオマス成分（5種類）のメトリクス表示
  - Dry_Green_g（緑植物）
  - Dry_Dead_g（枯死物）
  - Dry_Clover_g（クローバー）
  - GDM_g（緑乾物）
  - Dry_Total_g（総乾物）
- 環境情報の表示（サンプリング日、州、NDVI、平均高さ、種）
- バイオマス成分の棒グラフ表示
- フィルター機能（州、種別）

### 2. 📊 データセット統計
- データセット全体の統計情報
- 成分別の統計（平均、標準偏差、最小値、最大値、中央値）
- 州別サンプル数分布
- バイオマス成分間の相関分析
- NDVI vs バイオマスの散布図

### 3. 🔄 画像比較
- 複数画像の同時表示（最大5枚）
- バイオマス成分の比較チャート
- 詳細比較テーブル

## 必要なパッケージ

```bash
pip install streamlit pandas pillow plotly
```

## 使用方法

### 方法1: 起動スクリプトを使用

```bash
bash /kaggle/script/run_biomass_viewer.sh
```

### 方法2: 直接実行

```bash
cd /kaggle
streamlit run viz/biomass_viewer.py
```

## アクセス

ブラウザで以下のURLにアクセス:
```
http://localhost:8501
```

## データ構造

アプリケーションは `/kaggle/input/` ディレクトリ内のデータを参照します:

```
/kaggle/input/
├── train.csv          # 訓練データメタ情報
├── test.csv           # テストデータメタ情報
├── sample_submission.csv
├── train/             # 訓練画像
│   └── *.jpg
└── test/              # テスト画像
    └── *.jpg
```

## train.csvの構造

| カラム | 説明 |
|--------|------|
| sample_id | サンプルID |
| image_path | 画像パス |
| Sampling_Date | サンプリング日 |
| State | 州 |
| Species | 種（アンダースコア区切り） |
| Pre_GSHH_NDVI | NDVI値 |
| Height_Ave_cm | 平均高さ（cm） |
| target_name | ターゲット名 |
| target | ターゲット値（g） |

## スクリーンショット

アプリケーションには以下の3つのページがあります:

1. **画像詳細表示**: 個別の画像とそのメトリクスを詳細に表示
2. **データセット統計**: データセット全体の統計情報と相関分析
3. **画像比較**: 複数画像の比較分析

## カスタマイズ

`/kaggle/viz/biomass_viewer.py` を編集することで、表示内容や機能をカスタマイズできます。

## トラブルシューティング

### エラー: "画像が見つかりません"
- `/kaggle/input/` ディレクトリにデータが配置されているか確認してください

### エラー: "ModuleNotFoundError"
- 必要なパッケージがインストールされているか確認してください
- `pip install streamlit pandas pillow plotly` を実行してください

### ポートが使用中の場合
以下のようにポート番号を変更できます:
```bash
streamlit run viz/biomass_viewer.py --server.port 8502
```
