# upload_mode="create"
upload_mode="upload"
# exp_name="encoders"
# exp_name="exp_002_publicbasehomotrans_0_016_mixup_ema"
# exp_name="exp_002_publicbasehomotrans_0_014_mixup_ema_rerun"
# exp_name="exp_002_publicbasehomotrans_0_022_mixup_ema"
# exp_name="exp_009_5_002_eachbranch_cnn_model"
# exp_name="exp_012_6_001_cnn_transformer_model"
# exp_name="exp_013_5_002_eachbranch_cnn_model"
# exp_name="exp_006_imuonly_1_014_simple_cnn_model"
# exp_name="exp_013_5_004_eachbranch_cnn_model"
# exp_name="exp_013_5_005_eachbranch_cnn_model"
# exp_name="exp_013_5_006_eachbranch_cnn_model"
# exp_name="exp_013_6_007_eachbranch_trans_model"
# exp_name="exp_017_8_009_orient_behavior_aux"
# exp_name="exp_018_8_002_splitseq"
# exp_name="exp_019_8_006_splitseq_bugfixed"
# exp_name="exp_022_8_004_splitold_layer_dim_num"
# exp_name="exp_023_9_004_splitold_half"
# exp_name="exp_023_9_005_split_half"
# exp_name="exp_032_9_009_splitold_validfilter_thmscaler"
# exp_name="exp_033_9_004_splitold_valid_swap"
# exp_name="exp_034_9_001_splitpublic_valid_swap"
# exp_name="exp_035_9_004_splitpublic_valid_swap_metascale_scaleaug"  # Experiment name
# exp_name="exp_037_10_001_splitpublic_valid_swap_Nanaug01_cnn"  # Experiment name
# exp_name="exp_039_10_002_splitpublic_valid_swap_timestrech_cnn"  # Experiment name
# exp_name="exp_039_11_003_splitpublic_valid_swap_timestrech_trans"
# exp_name="exp_039_10_003_splitpublic_valid_swap_timestrech_lstmhead"
# exp_name="exp_041_10_004_splitpublic_valid_swap_drop03timestrech_nan-1or0_cnn"
# exp_name="exp_044_9_010_splitpublic_cnn"
# exp_name="exp_044_9_011_splitpublic_tran"
# exp_name="exp_044_9_012_splitpublic_lstm"
# exp_name="exp_044_9_013_splitpublic_cnnbce"

# exp_name="exp_046_9_001_splitpublic_cnn"
# exp_name="exp_046_9_002_splitpublic_tran"
# exp_name="exp_046_9_003_splitpublic_lstm" # lstm best
# exp_name="exp_046_9_004_splitpublic_cnnbce"
# exp_name="exp_047_9_002_splitpublic_tran" # tran best

# exp_name="exp_048_9_001_splitpublic_cnn"
# exp_name="exp_048_9_002_splitpublic_tran"
# exp_name="exp_048_9_003_splitpublic_lstm" # lstm best

exp_name="expL_001_9_001_cnn"

if [ "$upload_mode" = "create" ]; then
    echo "Creating a new dataset"
    python /kaggle/kaggle_apis/upload_models.py --exp-name "$exp_name" \
                                                --upload-or-create "create"
elif [ "$upload_mode" = "upload" ]; then
    echo "Uploading the existing dataset"
    python /kaggle/kaggle_apis/upload_models.py --exp-name "$exp_name"  \
                                                 --upload-or-create "upload"
else
    echo "Invalid upload mode"
fi
