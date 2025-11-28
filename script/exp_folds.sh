#!/bin/bash

exp_name="exp_003_000"
# exp_name="debug"
model_name="simple_clover_diff" 
dataset_name="simple"  # or "public"
loss_name="smooth_l1"
# loss_name="weighted_mse"
notes="first training with ReLU"
tags="simple"
# tagsにmodel_name, dataset_name, loss_nameを追加
tags="$tags $model_name $dataset_name $loss_name"
epochs=20
batch_size=8
# lr=7.5e-4
lr=5e-4
ema_decay=0.998
img_size=512


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
        --trainer.lr=$lr
done

exp_name="exp_003_001"
# exp_name="debug"
model_name="simple_clover_diff" 
dataset_name="simple"  # or "public"
# loss_name="smooth_l1"
loss_name="weighted_mse"
notes="first training with ReLU"
tags="simple"
# tagsにmodel_name, dataset_name, loss_nameを追加
tags="$tags $model_name $dataset_name $loss_name"
epochs=20
batch_size=8
# lr=7.5e-4
lr=5e-4
ema_decay=0.998
img_size=512


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
        --trainer.lr=$lr
done

