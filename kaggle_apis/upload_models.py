import argparse
import json
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, default="/kaggle/working")
    parser.add_argument("--kaggle-username", type=str, default="stgkrtua")
    parser.add_argument("--competition", type=str, default="cmi2025")
    parser.add_argument("--exp-name", type=str, default="debug")
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--upload-or-create", type=str, default="upload")
    args = parser.parse_args()
    target_files = [
        "final_weights.pth",
        "best_weights.pth",
        "final_weights_ema.pth",
        "config.yaml",
        "oof.csv",
        "inverse_gesture_dict.pkl",
        "feature_columns.yaml",
        "feature_scaler.yaml",
    ]
    upload_dir = "./upload_models"
    zip_dir = "./models"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)
    # src_dirがなければエラー
    exp_dir = os.path.join(args.src_dir, args.exp_name)
    print(f"Checking if {exp_dir} exists. exp_name: {args.exp_name}")
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"{exp_dir} is not found.")
    if args.exp_name == "encoders":
        data_dir = os.path.join(args.src_dir, args.exp_name)
        dist_dir = os.path.join(zip_dir, args.exp_name)
        if os.path.exists(dist_dir):
            print(f"{dist_dir} already exists. Skipping.")
        else:
            print(f"Copying from {data_dir} to {dist_dir}")
            os.makedirs(dist_dir, exist_ok=True)
            for target_file in target_files:
                src_path = os.path.join(data_dir, target_file)
                dst_path = os.path.join(dist_dir, target_file)
                # src_pathからdst_pathにファイルをコピー
                print(f"Copying {src_path} to {dst_path}")
                if not os.path.exists(src_path):
                    print(f"{src_path} does not exist. Skipping.")
                    continue
                os.system(f"cp {src_path} {dst_path}")

    # dst_dirを作成
    for fold in args.folds:
        exp_fold_dir = os.path.join(args.src_dir, args.exp_name, f"fold_{fold}")
        fold_dst_dir = os.path.join(zip_dir, args.exp_name, f"fold_{fold}")
        print(f"Processing {exp_fold_dir} to {fold_dst_dir}")
        if os.path.exists(fold_dst_dir):
            print(f"{fold_dst_dir} already exists. Skipping.")
            continue

        print(f"Processing fold {fold} from {exp_fold_dir} to {fold_dst_dir}")
        os.makedirs(fold_dst_dir, exist_ok=True)
        for target_file in target_files:
            src_path = os.path.join(exp_fold_dir, target_file)
            dst_path = os.path.join(fold_dst_dir, target_file)
            # src_pathからdst_pathにファイルをコピー
            print(f"Copying {src_path} to {dst_path}")
            os.system(f"cp {src_path} {dst_path}")

    # dst_pathの中身をzipにしてupload_dirに保存
    upload_zip_path = os.path.join(upload_dir, "models.zip")
    os.system(f"zip -r {upload_zip_path} {zip_dir}")
    print(f"Zipped {zip_dir} to {upload_zip_path}.zip")

    # dataset用のmetadataファイルを作成
    dataset_meta = {
        "title": f"{args.competition}-models",
        "id": f"{args.kaggle_username}/{args.competition}-models",
        "licenses": [{"name": "CC0-1.0"}],
    }
    # metadataファイルをupload_dirに保存
    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_meta, f)
    print(f"Metadata saved to {upload_dir}/dataset-metadata.json")
    print(f"Uploading: {upload_dir}")

    # os.system(f"rm -r {dst_path}")
    # kaggle APIを使ってファイルをアップロード
    if args.upload_or_create == "upload":
        subprocess.run(
            f"kaggle datasets version -p {upload_dir} -m 'upload' --dir-mode zip",
            shell=True,
        )
    elif args.upload_or_create == "create":
        subprocess.run(
            f"kaggle datasets create -p {upload_dir} --dir-mode zip", shell=True
        )
    else:
        raise ValueError("upload_or_create must be 'upload' or 'create'")
    # 一時ファイルを削除
    os.system(f"rm -r {upload_zip_path}")
