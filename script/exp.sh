# Description: Run the experiment
# exp_name="public_base_000_1_9"
# # exp_name="debug"
# # /kaggle/working/にexp_nameのディレクトリがあるときsuffixをつける
# if [ -d /kaggle/working/$exp_name ]; then
#   suffix=$(date "+_%Y%m%d_%H%M%S")
#   exp_name=$exp_name$suffix
# fi

# # 作成したexp_nameを表示
# echo "exp_name: $exp_name"

# # train with argparse and dataclass (instead of hydra)
# python src/train_argparse.py --exp_name="$exp_name"


# exp_name="exp_007_clover_sum_033"
exp_name="debug"
only_fold_0=true

model_name="clover_sum"
loss_name="weighted_clover_loss"
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
aux_weight=20.0
emb_dim=1024
head_connection_type="class_head"  # "class_head" or "no_head"

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
        --loss.aux_weight=$aux_weight \
        --loss.target_weights="$target_weight" \
        --model.head_connection_type=$head_connection_type \
        --split.split_dir="$split_dir"
done
