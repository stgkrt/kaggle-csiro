#!/bin/bash

# exp_name="public_base_001_homotrans_0_2"
exp_name="debug"
df_path="/kaggle/working/processed/processed_with_homotrans_df.csv"
imu_dim=40
tof_dim=25

# Add timestamp suffix if directory exists
if [ -d /kaggle/working/$exp_name ]; then
    suffix=$(date "+_%Y%m%d_%H%M%S")
    exp_name=$exp_name$suffix
fi

echo "Experiment Name: $exp_name"

# Run training with specific epoch count
python src/train_argparse.py \
    --fold=0 \
    --exp_name="$exp_name" \
    --df_path="$df_path" \
    --imu_dim="$imu_dim" \
    --tof_dim="$tof_dim"
