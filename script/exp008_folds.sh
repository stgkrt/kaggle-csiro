#!/bin/bash


exp_name="exp_008_007"
# exp_name="debug"
# only_fold_0=true

model_name="clover_sum_height"
loss_name="clover_loss"
# loss_name="weighted_smooth_l1_loss"
dataset_name="clover_height"
# target_weight='[0.0, 1.0, 0.0, 0.0, 0.0]' # deadのみ学習

notes="only train dead output"
tags="rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


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
        --split.split_dir="$split_dir"
done



exp_name="exp_008_008"
# exp_name="debug"
# only_fold_0=true

dataset_name="clover_height"
model_name="clover_sum_height"
loss_name="weighted_clover_height_loss"
# loss_name="weighted_smooth_l1_loss"

notes="calculate dead from other outputs"
tags="simple rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


target_weight='[0.1, 0.1, 0.1, 0.2, 0.5]'
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
head_connection_type="direct"  # "class_head"

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
        --loss.target_weights="$target_weight" \
        --model.head_connection_type=$head_connection_type \
        --split.split_dir="$split_dir"
done






exp_name="exp_008_009"

# exp_name="debug"
# only_fold_0=true

model_name="clover_sum_height"
# loss_name="weighted_clover_loss"
# loss_name="weighted_smooth_l1_loss"
loss_name="weighted_clover_height_loss"

# mse_weightsのlistを書き換える (JSON形式)
target_weight='[0.5, 0.1, 0.5, 0.1, 0.1]'

notes="calculate dead from other outputs"
tags="simple rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"  # "class_head" or "no_head"

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
        --loss.target_weights="$target_weight" \
        --split.split_dir="$split_dir"
done





exp_name="exp_008_010"
# exp_name="debug"
# only_fold_0=true

model_name="clover_sum_height"
loss_name="weighted_clover_height_loss"
# loss_name="weighted_smooth_l1_loss"

# mse_weightsのlistを書き換える (JSON形式)
target_weight='[0.5, 0.5, 0.5, 0.1, 0.1]'

notes="calculate dead from other outputs"
tags="simple rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.5
emb_dim=1024
head_connection_type="direct"  # "class_head" or "no_head"

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
        --loss.target_weights="$target_weight" \
        --split.split_dir="$split_dir"
done


exp_name="exp_008_011"
# exp_name="debug"
# only_fold_0=true

model_name="clover_sum_height"
# loss_name="weighted_clover_height_loss"
loss_name="clover_height_loss"
# loss_name="weighted_smooth_l1_loss"

# mse_weightsのlistを書き換える (JSON形式)
target_weight='[0.5, 0.5, 0.5, 0.1, 0.1]'

notes="calculate dead from other outputs"
tags="simple rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=0.1
emb_dim=1024
head_connection_type="direct"  # "class_head"

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
        --loss.target_weights="$target_weight" \
        --split.split_dir="$split_dir"
done



exp_name="exp_008_012"
# exp_name="debug"
# only_fold_0=true

model_name="clover_sum_height"
# loss_name="weighted_clover_height_loss"
loss_name="clover_height_loss"
# loss_name="weighted_smooth_l1_loss"

# mse_weightsのlistを書き換える (JSON形式)
target_weight='[0.5, 0.5, 0.5, 0.1, 0.1]'

notes="calculate dead from other outputs"
tags="simple rickfolds"
tags="$tags $model_name $dataset_name $loss_name"


split_dir="/kaggle/working/splits_rick_folds"

epochs=30
batch_size=8
lr=1e-3
ema_decay=0.993
img_size=512
aux_clover_weight=20.0
aux_height_weight=1.0
emb_dim=1024
head_connection_type="direct"  # "class_head"

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
        --loss.target_weights="$target_weight" \
        --split.split_dir="$split_dir"
done
