import argparse
import json
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, default="./src")
    parser.add_argument("--upload-dir", type=str, default="./upload_src")
    parser.add_argument("--kaggle-username", type=str, default="stgkrtua")
    parser.add_argument("--competition", type=str, default="csiro-biomass")
    parser.add_argument("--upload-or-create", type=str, default="upload")
    args = parser.parse_args()
    # src_dirがなければエラー
    if not os.path.exists(args.src_dir):
        print(f"{args.src_dir} is not found.")
        raise FileNotFoundError("src is not found.")

    # dst_pathの中身をzipにしてupload_dirに保存
    os.makedirs(args.upload_dir, exist_ok=True)
    zip_path = f"{args.upload_dir}/src.zip"
    os.system(f"zip -r {zip_path} {args.src_dir}")
    dataset_name = os.path.basename(args.src_dir)
    # dataset用のmetadataファイルを作成
    dataset_id = f"{args.kaggle_username}/{args.competition}-dataset"
    dataset_meta = {
        "title": f"{args.competition}-dataset-{args.dataset_suffix}",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    # metadataファイルをupload_dirに保存
    with open(os.path.join(args.upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_meta, f)

    # kaggle APIを使ってファイルをアップロード
    if args.upload_or_create == "upload":
        subprocess.run(
            f"kaggle datasets version -p {args.upload_dir} -m 'upload' -d {dataset_id}",
            shell=True,
        )
    elif args.upload_or_create == "create":
        subprocess.run(
            f"kaggle datasets create -p {args.upload_dir} --dir-mode zip", shell=True
        )
    else:
        raise ValueError("upload_or_create must be 'upload' or 'create'")
    # 一時ファイルを削除
    os.system(f"rm -r {args.upload_dir}")
