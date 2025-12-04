#!/bin/bash

# exp_name="exp_004_000"
exp_name="debug"

# model_name="simple_model"
# dataset_name="simple"  # or "public"
# loss_name="smooth_l1"

# model_name="height_gshh_model"
# dataset_name="height_gshh"
# loss_name="height_gshh_loss"

model_name="height_model"
dataset_name="height"
loss_name="height_loss"

notes="aux height and gshh prediction"
tags="simple"
# tagsにmodel_name, dataset_name, loss_nameを追加
tags="$tags $model_name $dataset_name $loss_name"
epochs=30
batch_size=8
lr=2e-3
ema_decay=0.993
img_size=512
aux_weight=0.01

# Add timestamp suffix if directory exists
if [ -d /kaggle/working/$exp_name ]; then
    suffix=$(date "+_%Y%m%d_%H%M%S")
    exp_name=$exp_name$suffix
fi

echo "Experiment Name: $exp_name"

# Run training with specific epoch count
for fold in 0 1 2 3 4
# for fold in 0
do
    echo "Run exp $exp_name fold $fold"
    python src/train.py \
        --exp_name="$exp_name" \
        --model.model_name=$model_name \
        --dataset.dataset_name="$dataset_name" \
        --dataset.batch_size=$batch_size \
        --notes="$notes" \
        --tags="$tags" \
        --fold=$fold \
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --loss.aux_weight=$aux_weight
done
