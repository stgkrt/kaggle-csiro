upload_mode="create"
dataset_path="/kaggle/working/"

if [ "$upload_mode" = "create" ]; then
    echo "Creating a new dataset"
    python /kaggle/kaggle_apis/upload_dataset.py --src-dir ${dataset_path} --dataset-suffix ${dataset_no} --upload-or-create "create"
elif [ "$upload_mode" = "upload" ]; then
    echo "Uploading the existing dataset"
    python /kaggle/kaggle_apis/upload_dataset.py --src-dir ${dataset_path} --dataset-suffix ${dataset_no} --upload-or-create "upload"
else
    echo "Invalid upload mode"
fi
