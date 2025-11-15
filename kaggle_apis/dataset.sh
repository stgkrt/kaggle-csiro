upload_mode="create"
# upload_mode="upload"
# dataset_path="/kaggle/working/processed_crop_ajust_sgsize0.2_axsize0.15_sgl1tol5w-0.085_sgl1tol5h0.0_sgl5s1w-0.085_sgl5s1h0.15_axsize0.0_png"
# dataset_path="/kaggle/working/processed_crop256_ajust_sgsize0.2_axsize0.15_sgl1tol5w-0.085_sgl1tol5h0.0_sgl5s1w-0.085_sgl5s1h0.15_axsize0.0_png"
# dataset_path="/kaggle/working/processed_clahe_physical_square_filter55_crop256_ajust_stsize50_axsize25_sgl1tol5w0_sgl1tol5h0_sgl5s1w0_sgl5s1h0_png"
# dataset_path="/kaggle/working/processed_clahe2_physical_square_filter55_crop256_ajust_stsize50_axsize25_sgl1tol5w0_sgl1tol5h0_sgl5s1w0_sgl5s1h0_png"
dataset_path="/kaggle/working/processed_clahe_lowpass_physical_square_filter55_crop256_ajust_stsize50_axsize25_sgl1tol5w0_sgl1tol5h0_sgl5s1w0_sgl5s1h0_png"
dataset_no=lowpass

if [ "$upload_mode" = "create" ]; then
    echo "Creating a new dataset"
    python /kaggle/kaggle_apis/upload_dataset.py --src-dir ${dataset_path} --dataset-suffix ${dataset_no} --upload-or-create "create"
elif [ "$upload_mode" = "upload" ]; then
    echo "Uploading the existing dataset"
    python /kaggle/kaggle_apis/upload_dataset.py --src-dir ${dataset_path} --dataset-suffix ${dataset_no} --upload-or-create "upload"
else
    echo "Invalid upload mode"
fi
