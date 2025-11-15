# DTW計算機能の使用方法

## 概要

`process_representative_waves.py`は、`/kaggle/working/representative_waves`フォルダに保存されたnumpy形式の代表波形ファイルと入力波形との間のDynamic Time Warping (DTW)距離を計算する機能を提供します。

## 主な機能

### 1. RepresentativeWaveDTWCalculator クラス

代表波形の読み込みとDTW計算を行うメインクラス。

```python
from data.process_representative_waves import RepresentativeWaveDTWCalculator

# インスタンス作成
calculator = RepresentativeWaveDTWCalculator()

# 利用可能な行動タイプを取得
behaviors = calculator.get_available_behaviors()
print(behaviors)
# ['Eyelash___pull_hair', 'Neck___scratch', 'Neck___pinch_skin', ...]

# 特定行動の利用可能な波形タイプを取得
wave_types = calculator.get_available_wave_types('Neck___pinch_skin')
print(wave_types)
# ['acc_x_gravity_free_wave', 'acc_y_gravity_free_wave', ...]
```

### 2. DTW計算

#### スライディングウィンドウ方式（推奨）

```python
import numpy as np

# 入力波形の準備
input_waves = {
    'acc_x_gravity_free_wave': np.random.randn(1000),
    'acc_y_gravity_free_wave': np.random.randn(1000),
    'acc_z_gravity_free_wave': np.random.randn(1000),
    'acc_mag_gravity_free_wave': np.random.randn(1000)
}

# 特定行動に対するDTW計算
dtw_distances = calculator.calculate_dtw_for_behavior(
    input_waves,
    'Neck___pinch_skin',
    method='sliding_window',
    window_size=50
)

# 結果確認
for wave_type, distances in dtw_distances.items():
    print(f"{wave_type}: shape={distances.shape}, mean={distances.mean():.4f}")
```

#### 全行動に対するDTW計算

```python
# 全行動に対してDTW計算
all_dtw_results = calculator.calculate_dtw_for_all_behaviors(
    input_waves,
    method='sliding_window',
    window_size=50
)

# 最も類似した行動を見つける
behavior_scores = {}
for behavior_name, wave_distances in all_dtw_results.items():
    avg_score = np.mean([distances.mean() for distances in wave_distances.values()])
    behavior_scores[behavior_name] = avg_score

best_match = min(behavior_scores.items(), key=lambda x: x[1])
print(f"最も類似した行動: {best_match[0]} (スコア: {best_match[1]:.4f})")
```

### 3. 便利関数

```python
from data.process_representative_waves import calculate_dtw_distances

# 簡単にDTW計算を実行
dtw_results = calculate_dtw_distances(
    input_waves,
    method='sliding_window',
    window_size=50,
    target_behaviors=['Neck___pinch_skin', 'Forehead___scratch']  # 特定行動のみ
)
```

## パラメータ説明

### DTW計算方式

- **sliding_window** (推奨): 各時点でスライディングウィンドウを使用してDTW距離を計算
  - 計算が高速
  - メモリ効率が良い
  - 実用的な結果

- **pointwise**: ワーピングパスに基づいて各点の正確なDTW距離を計算
  - 計算時間が長い
  - より精密な結果

### パラメータ

- `window_size`: スライディングウィンドウのサイズ（デフォルト: 50）
  - 小さい値: より細かい変化を捉える、計算時間短い
  - 大きい値: よりグローバルなパターンを捉える、計算時間長い

## 出力形式

DTW計算の結果は入力波形と同じshapeのnumpy配列で返されます：

```python
input_wave.shape   # (1000,)
dtw_distances.shape # (1000,) - 入力と同じ長さ

# 各時点でのDTW距離
# 値が小さいほど代表波形との類似度が高い
```

## 使用例

詳細な使用例は以下のファイルを参照：
- `simple_dtw_test.py`: 基本的な使用方法
- `dtw_example.py`: 完全な例とビジュアライゼーション

## ディレクトリ構造

```
/kaggle/working/representative_waves/
├── Above_ear___pull_hair/
│   ├── acc_x_gravity_free_wave.npy
│   ├── acc_y_gravity_free_wave.npy
│   ├── acc_z_gravity_free_wave.npy
│   └── acc_mag_gravity_free_wave.npy
├── Neck___pinch_skin/
│   └── ...
└── ...
```

各行動フォルダには4つの波形タイプのnpyファイルが含まれています。
