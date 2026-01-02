#!/bin/bash

exp_name="exp_020_001"
# exp_name="debug"
# only_fold_0=true

# model_name="clover_sum_height"
# model_name="clover_height_pyramid_sum"
# model_name="clover_height_seg_simple"
# model_name="clover_sum_height_film"
# model_name="clover_sum_height_nvdi"
# model_name="clover_height_seg"
# model_name="clover_height_pyramid_sum"
# model_name="tiled_clover_sum_height_film"
# model_name="clover_sum_height_nvdi"
model_name="clover_sum_height_nvdi"
# loss_name="clover_height_loss"
# loss_name="weighted_smooth_l1_loss"
# loss_name="log_clover_height_seg_loss"
# loss_name="clover_height_seg_loss"
# loss_name="clover_height_seg_loss"
# loss_name="log_clover_height_loss"
# loss_name="log_clover_height_nvdi_loss"
loss_name="log_clover_height_nvdi_loss"
# loss_name="log_clover_height_nvdi_loss"
# dataset_name="clover_height"
# dataset_name="clover_height_nvdi"
dataset_name="clover_height_nvdi"
# dataset_name="clover_height_seg"
# target_weight='[0.3, 0.3, 0.3, 0.05, 0.05]' # deadのみ学習
backbone_name="tf_efficientnet_b0"
# backbone_name="tf_efficientnet_b2_ns"
# backbone_name="resnet18.a3_in1k"
# backbone_name="vit_small_patch14_reg4_dinov2.lvd142m"

notes="pyramid_model, fixed_metrics"
tags="rickfolds fixed_metrics mixup_cutmix"
tags="$tags $model_name $dataset_name $loss_name"

split_dir="/kaggle/working/splits_rick_folds"
# split_dir="/kaggle/working/splits_group_date_stg_species"

# epochs=30
epochs=100
batch_size=8
# lr=5e-3
lr=1e-3
# lr=1e-5
# lr=1e-4
ema_decay=0.998
# img_size=512
img_height=512
img_width=512
aux_clover_weight=0.1
aux_height_weight=0.5
aux_nvdi_weight=1.0
aux_seg_weight=0.1
# aux_seg_weight=0.01
segmentation_depth=4
emb_dim=1024
# head_connection_type="direct"
head_connection_type="class_head"
mixup_prob=0.5
cutmix_prob=0.5

Add timestamp suffix if directory exists
if [ -d /kaggle/working/$exp_name ]; then
    suffix=$(date "+_%Y%m%d_%H%M%S")
    exp_name=$exp_name$suffix
fi

echo "Experiment Name: $exp_name"

# Run training with specific epoch count
# for fold in 0 1 2 3 4
if [ "$only_fold_0" = true ] ; then
    folds_to_run="0"
else
    folds_to_run="0 1 2 3 4"
fi

for fold in $folds_to_run
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
        --augmentation.resize_img_height=$img_height \
        --augmentation.resize_img_width=$img_width \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --loss.aux_seg_weight=$aux_seg_weight \
        --loss.aux_nvdi_weight=$aux_nvdi_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --model.segmentation_depth=$segmentation_depth \
        --dataset.mixup_prob=$mixup_prob \
        --dataset.cutmix_prob=$cutmix_prob \
        --split.split_dir="$split_dir"
done
