#!/bin/bash

# exp_name="exp_000"
exp_name="debug"
model_name="simple_model" 
dataset_name="simple"  # or "public"
notes="first training"
tags="simple"
# tagsにmodel_name, dataset_nameをスペースを開けて追加
tags="$tags $model_name $dataset_name"
epochs=10
batch_size=64


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
        --trainer.max_epochs=$epochs 
done
