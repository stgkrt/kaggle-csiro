#!/bin/bash

# python src/inference.py \
#   --exp_dir /kaggle/working/exp_002 \
#   --folds 0 1 2 3 4 \
#   --weight_type best \
#   --test_csv /kaggle/input/csiro-biomass/test.csv \
#   --data_root /kaggle/input/csiro-biomass/ \
#   --output_dir /kaggle/working \
#   --batch_size 64 \
#   --num_workers 0 \
#   --device cuda

# python src/inference.py \
#   --exp_dir /kaggle/working/exp_002_002 \
#   --folds 0 1 2 3 4 \
#   --weight_type best \
#   --test_csv /kaggle/input/csiro-biomass/train.csv \
#   --data_root /kaggle/input/csiro-biomass/ \
#   --output_dir /kaggle/working \
#   --batch_size 64 \
#   --num_workers 0 \
#   --device cuda


python src/inference.py \
  --exp_dir /kaggle/working/exp_000_000 \
  --folds 0 \
  --weight_type best \
  --test_csv /kaggle/input/csiro-biomass/train.csv \
  --data_root /kaggle/input/csiro-biomass/ \
  --output_dir /kaggle/working \
  --batch_size 64 \
  --num_workers 0 \
  --device cuda
