# Description: Run the experiment
exp_name="public_base_000_1_9"
# exp_name="debug"
# /kaggle/working/にexp_nameのディレクトリがあるときsuffixをつける
if [ -d /kaggle/working/$exp_name ]; then
  suffix=$(date "+_%Y%m%d_%H%M%S")
  exp_name=$exp_name$suffix
fi

# 作成したexp_nameを表示
echo "exp_name: $exp_name"

# train with argparse and dataclass (instead of hydra)
python src/train_argparse.py --exp_name="$exp_name"
