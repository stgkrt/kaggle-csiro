#!/bin/bash


exp_name="exp_009_001"
# exp_name="debug"
# only_fold_0=true

model_name="simple_model"
loss_name="smooth_l1_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="simple"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
backbone_name="efficientnet_b5"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
# lr=5e-4
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done



exp_name="exp_009_002"
# exp_name="debug"
# only_fold_0=true

model_name="clover_model"
loss_name="clover_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="clover"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
backbone_name="efficientnet_b5"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
# lr=5e-4
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done



exp_name="exp_009_003"
# exp_name="debug"
# only_fold_0=true

model_name="clover_model"
loss_name="clover_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="clover"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
backbone_name="efficientnet_b5"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
# lr=5e-4
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="class_head"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done




exp_name="exp_009_004"
# exp_name="debug"
# only_fold_0=true

model_name="simple_model"
loss_name="smooth_l1_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="simple"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
# 384のswin
backbone_name="swin_base_patch4_window12_384.ms_in1k"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
lr=1e-6
ema_decay=0.993
img_size=384
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done



exp_name="exp_009_005"
# exp_name="debug"
# only_fold_0=true

model_name="clover_model"
loss_name="clover_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="clover"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
backbone_name="swin_base_patch4_window12_384.ms_in1k"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
# lr=1e-3
lr=1e-6
ema_decay=0.993
img_size=384
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done



exp_name="exp_009_006"
# exp_name="debug"
# only_fold_0=true

model_name="clover_model"
loss_name="clover_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="clover"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"
backbone_name="efficientnet_b5"

split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
# lr=1e-3
lr=1e-6
ema_decay=0.993
img_size=384
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="class_head"

# Add timestamp suffix if directory exists
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
        --augmentation.resize_img_height=$img_size \
        --augmentation.resize_img_width=$img_size \
        --loss.loss_name="$loss_name" \
        --trainer.max_epochs=$epochs \
        --trainer.ema_decay=$ema_decay \
        --trainer.lr=$lr \
        --model.emb_dim=$emb_dim \
        --loss.aux_clover_weight=$aux_clover_weight \
        --loss.aux_height_weight=$aux_height_weight \
        --model.head_connection_type=$head_connection_type \
        --model.backbone_name=$backbone_name \
        --split.split_dir="$split_dir"
done
