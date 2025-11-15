#!/bin/bash

# simple cnn best 005
# each_branch_cnn_model best exp_008_5_009_simple_cnn_model_imu_tof_thm
# --> each_branch_cnn_model best exp_009_5_002_eachbranch_cnn_model

# exp_name="exp_008_5_013_simple_cnn_model_imu_tof_thm"
exp_name="exp_009_5_002_eachbranch_cnn_model"
# exp_name="debug"
model_name="each_branch_cnn_model"  # or "public_model", "public_imu_model", "simple_cnn_model"
dataset_name="basic"  # or "public"
notes="Experiment with public base. EMA training. IMU Thm ToF. each branch model.ToF Thm scaling."
tags="public_base EMA homotrans later_signals hparam_tune each_branch_model imu_tof_thm tof_thm_scaling imu_scaling"
epochs=100
# df_path="/kaggle/working/processed/processed_with_homotrans_df.csv"
# imu_dim=11
df_path="/kaggle/working/processed_rotations_2/processed_with_rots_df.csv"
imu_dim=20
thm_dim=5
tof_dim=15
batch_size=8
model_emb_dim=32  # Example custom embedding dimension
model_layer_num=7  # Example custom layer number
lr=2e-4


# Add timestamp suffix if directory exists
if [ -d /kaggle/working/$exp_name ]; then
    suffix=$(date "+_%Y%m%d_%H%M%S")
    exp_name=$exp_name$suffix
fi

echo "Experiment Name: $exp_name"

# Run training with specific epoch count
for fold in 0 1 2 3 4
do
    echo "Run exp $exp_name fold $fold"
    python src/train_argparse.py \
        --model_name=$model_name \
        --batch_size=$batch_size \
        --lr=$lr \
        --notes="$notes" \
        --tags="$tags" \
        --fold=$fold \
        --dataset_name="$dataset_name" \
        --epochs=$epochs \
        --exp_name="$exp_name" \
        --df_path="$df_path" \
        --imu_dim="$imu_dim" \
        --thm_dim="$thm_dim" \
        --tof_dim="$tof_dim"
done
