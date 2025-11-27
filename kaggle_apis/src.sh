upload_mode="create"
# upload_mode="upload"

if [ "$upload_mode" = "create" ]; then
    echo "Creating a new dataset"
    python /kaggle/kaggle_apis/upload_src.py --upload-or-create "create"
elif [ "$upload_mode" = "upload" ]; then
    echo "Uploading the existing dataset"
    python /kaggle/kaggle_apis/upload_src.py --upload-or-create "upload"
else
    echo "Invalid upload mode"
fi
