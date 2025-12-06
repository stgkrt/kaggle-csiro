# Test Suite for Kaggle CSIRO Biomass Project

このディレクトリには、`src/model`、`src/data`、`src/metrics`フォルダ内のファイルに対するテストコードが含まれています。

## テストファイル一覧

### Model Tests
- **test_losses.py**: 損失関数のテスト
  - WeightedMSELoss
  - MSELoss
  - SmoothL1Loss
  - WeightedSmoothL1Loss
  - HeightLoss
  - CloverLoss
  - LossModule

- **test_model_architectures.py**: モデルアーキテクチャのテスト
  - get_model_architecture関数
  - ModelArchitecturesクラス
  - 各種モデル（simple_model, height_model, clover_model等）
  - 補助ヘッド付きモデル

### Data Tests
- **test_augmentations.py**: データ拡張のテスト
  - get_train_transforms
  - get_valid_transforms
  - 各種Augmentation（HorizontalFlip, Shadow, Resize等）

- **test_simple_dataset.py**: データセットのテスト
  - SimpleDatasetクラス
  - __getitem__メソッド
  - 学習/検証/テストフェーズでの動作

- **test_data_module.py**: DataModuleのテスト
  - DataModuleクラス
  - train_dataloader/val_dataloaderメソッド
  - ハイパーパラメータの保存

### Metrics Tests
- **test_competition_metrics.py**: 評価指標のテスト
  - weighted_r2_score (numpy版)
  - weighted_r2_score_torch (PyTorch版)
  - CompetitionMetricsクラス
  - 各種R²スコア

## テストの実行方法

### すべてのテストを実行
```bash
pytest test/
```

### 特定のファイルのテストを実行
```bash
pytest test/test_losses.py
pytest test/test_augmentations.py
pytest test/test_competition_metrics.py
```

### 詳細な出力を表示
```bash
pytest test/ -v
```

### カバレッジレポートを生成
```bash
pytest test/ --cov=src --cov-report=html
```

### 特定のテストクラスを実行
```bash
pytest test/test_losses.py::TestWeightedMSELoss
```

### 特定のテストメソッドを実行
```bash
pytest test/test_losses.py::TestWeightedMSELoss::test_forward
```

## 簡易実行（デバッグ用）

各テストファイルは単体でも実行可能です：

```bash
cd /kaggle
python test/test_losses.py
python test/test_augmentations.py
python test/test_competition_metrics.py
```

## テストの構造

各テストファイルは以下の構造で組織化されています：

1. **基本機能テスト**: クラスや関数の基本的な動作を確認
2. **エッジケーステスト**: 境界値や特殊なケースでの動作を確認
3. **統合テスト**: 複数のコンポーネントを組み合わせた動作を確認

## テストカバレッジ

テストは以下の観点でカバーしています：

- ✅ 正常系の動作確認
- ✅ 異常系の動作確認（エラーハンドリング）
- ✅ エッジケース（境界値、空データ等）
- ✅ 型チェック（入出力の型が正しいか）
- ✅ 形状チェック（テンソルの形状が期待通りか）
- ✅ 数値精度（計算結果が正しいか）

## 依存関係

テストを実行するには以下のパッケージが必要です：

```bash
pip install pytest pytest-cov
```

必要な主な依存パッケージ：
- pytest
- pytest-cov (カバレッジ測定用)
- torch
- numpy
- pandas
- albumentations
- PIL (Pillow)
- pytorch-lightning

## 注意事項

- 一部のテストは実際のデータやGPUが必要な場合があります
- GPU関連のテストは、CUDA未使用環境では自動的にスキップされます
- ファイルシステムを使用するテストは一時ディレクトリを使用します
- モデルのpretrained weightsをダウンロードするテストは`pretrained=False`で実行されます

## トラブルシューティング

### ImportError が発生する場合
```bash
export PYTHONPATH=/kaggle:$PYTHONPATH
```

### CUDA out of memory エラーが発生する場合
バッチサイズを小さくしてテストを実行してください。

### テストが遅い場合
`-n auto`オプションで並列実行できます（pytest-xdist が必要）：
```bash
pip install pytest-xdist
pytest test/ -n auto
```

## 継続的な改善

テストは継続的に追加・改善されます。新しい機能を追加する際は、対応するテストも追加してください。

## 貢献

テストに関するバグや改善提案がある場合は、issueを作成してください。
