"""
Streamlit OOF Analysis App for Model Experiments
実験結果のOOF予測を分析し、targetごとのスコアや分布を可視化
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ページ設定
st.set_page_config(
    page_title="OOF分析ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """評価指標を計算"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (0除算を避ける)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def load_oof_data(exp_folder: Path) -> Tuple[pd.DataFrame, List[str]]:
    """実験フォルダからOOFデータを読み込む"""
    all_oof = []
    fold_dirs = sorted(
        [d for d in exp_folder.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    )

    for fold_dir in fold_dirs:
        oof_file = fold_dir / "oof.csv"
        if oof_file.exists():
            df = pd.read_csv(oof_file)
            df["fold"] = int(fold_dir.name.split("_")[1])
            all_oof.append(df)

    if not all_oof:
        return None, []

    oof_df = pd.concat(all_oof, ignore_index=True)
    folds = sorted(oof_df["fold"].unique())

    return oof_df, folds


def load_train_data() -> pd.DataFrame:
    """訓練データを読み込む"""
    train_df = pd.read_csv("/kaggle/input/csiro-biomass/train.csv")
    return train_df


def merge_oof_with_train(oof_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """OOFデータと訓練データをマージ"""
    # sample_idを使ってマージ
    merged_df = oof_df.merge(
        train_df[
            [
                "sample_id",
                "image_path",
                "State",
                "Species",
                "Pre_GSHH_NDVI",
                "Height_Ave_cm",
                "target_name",
            ]
        ],
        on="sample_id",
        how="left",
    )

    # target_nameを抽出（sample_idに含まれている場合）
    if "target_name" not in merged_df.columns or merged_df["target_name"].isna().all():
        merged_df["target_name"] = merged_df["sample_id"].str.extract(
            r"__(Dry_\w+_g)$"
        )[0]

    return merged_df


def load_image(
    image_path: str, base_dir: Path = Path("/kaggle/input/csiro-biomass")
) -> Image.Image:
    """画像を読み込む"""
    try:
        full_path = base_dir / image_path
        if full_path.exists():
            return Image.open(full_path)
        else:
            # パスの形式が異なる場合の対応
            alt_path = base_dir / image_path.replace("train/", "")
            if alt_path.exists():
                return Image.open(alt_path)
    except Exception as e:
        st.error(f"画像読み込みエラー: {e}")
    return None


def plot_target_metrics(merged_df: pd.DataFrame):
    """各targetごとの評価指標を可視化"""
    st.subheader("📈 Target別の評価指標")

    target_names = sorted(merged_df["target_name"].dropna().unique())

    metrics_list = []
    for target_name in target_names:
        target_data = merged_df[merged_df["target_name"] == target_name]
        metrics = calculate_metrics(
            target_data["target"].values, target_data["pred"].values
        )
        metrics["target_name"] = target_name
        metrics["n_samples"] = len(target_data)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # メトリクスを表示
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            metrics_df.style.format(
                {
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "R2": "{:.4f}",
                    "MAPE": "{:.2f}%",
                    "n_samples": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        # 棒グラフで可視化
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("MAE", "RMSE", "R2", "MAPE (%)"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        fig.add_trace(
            go.Bar(x=metrics_df["target_name"], y=metrics_df["MAE"], name="MAE"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=metrics_df["target_name"], y=metrics_df["RMSE"], name="RMSE"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(x=metrics_df["target_name"], y=metrics_df["R2"], name="R2"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=metrics_df["target_name"], y=metrics_df["MAPE"], name="MAPE"),
            row=2,
            col=2,
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def plot_prediction_scatter(merged_df: pd.DataFrame, selected_targets: List[str]):
    """予測値 vs 実測値の散布図"""
    st.subheader("🎯 予測値 vs 実測値")

    n_targets = len(selected_targets)
    n_cols = min(2, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=selected_targets,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for idx, target_name in enumerate(selected_targets):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        target_data = merged_df[merged_df["target_name"] == target_name]

        # 散布図
        fig.add_trace(
            go.Scatter(
                x=target_data["target"],
                y=target_data["pred"],
                mode="markers",
                marker=dict(size=4, opacity=0.5),
                name=target_name,
                text=target_data["sample_id"],
                hovertemplate="<b>%{text}</b><br>Target: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # 理想線（y=x）
        max_val = max(target_data["target"].max(), target_data["pred"].max())
        min_val = min(target_data["target"].min(), target_data["pred"].min())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="y=x",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="True Value", row=row, col=col)
        fig.update_yaxes(title_text="Predicted Value", row=row, col=col)

    fig.update_layout(height=400 * n_rows, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_residual_distribution(merged_df: pd.DataFrame, selected_targets: List[str]):
    """残差の分布を可視化"""
    st.subheader("📊 残差分布")

    merged_df["residual"] = merged_df["pred"] - merged_df["target"]

    # ヒストグラム
    fig = go.Figure()

    for target_name in selected_targets:
        target_data = merged_df[merged_df["target_name"] == target_name]
        fig.add_trace(
            go.Histogram(
                x=target_data["residual"], name=target_name, opacity=0.7, nbinsx=50
            )
        )

    fig.update_layout(
        title="残差のヒストグラム",
        xaxis_title="Residual (Pred - True)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 箱ひげ図
    fig2 = go.Figure()

    for target_name in selected_targets:
        target_data = merged_df[merged_df["target_name"] == target_name]
        fig2.add_trace(
            go.Box(y=target_data["residual"], name=target_name, boxmean="sd")
        )

    fig2.update_layout(
        title="残差の箱ひげ図", yaxis_title="Residual (Pred - True)", height=400
    )

    st.plotly_chart(fig2, use_container_width=True)


def plot_error_by_features(merged_df: pd.DataFrame, selected_targets: List[str]):
    """特徴量別の誤差分析"""
    st.subheader("🔍 特徴量別の誤差分析")

    merged_df["abs_error"] = np.abs(merged_df["pred"] - merged_df["target"])

    col1, col2 = st.columns(2)

    with col1:
        # State別の誤差
        if "State" in merged_df.columns:
            fig = px.box(
                merged_df[merged_df["target_name"].isin(selected_targets)],
                x="State",
                y="abs_error",
                color="target_name",
                title="State別の絶対誤差",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Species別の誤差
        if "Species" in merged_df.columns:
            fig = px.box(
                merged_df[merged_df["target_name"].isin(selected_targets)],
                x="Species",
                y="abs_error",
                color="target_name",
                title="Species別の絶対誤差",
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Height vs Error
    if "Height_Ave_cm" in merged_df.columns:
        fig = px.scatter(
            merged_df[merged_df["target_name"].isin(selected_targets)],
            x="Height_Ave_cm",
            y="abs_error",
            color="target_name",
            title="草丈と絶対誤差の関係",
            opacity=0.5,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # NDVI vs Error
    if "Pre_GSHH_NDVI" in merged_df.columns:
        fig = px.scatter(
            merged_df[merged_df["target_name"].isin(selected_targets)],
            x="Pre_GSHH_NDVI",
            y="abs_error",
            color="target_name",
            title="NDVIと絶対誤差の関係",
            opacity=0.5,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def plot_fold_comparison(merged_df: pd.DataFrame, selected_targets: List[str]):
    """Fold間のスコア比較"""
    st.subheader("📂 Fold別のスコア比較")

    fold_metrics = []
    for fold in sorted(merged_df["fold"].unique()):
        fold_data = merged_df[merged_df["fold"] == fold]
        for target_name in selected_targets:
            target_data = fold_data[fold_data["target_name"] == target_name]
            if len(target_data) > 0:
                metrics = calculate_metrics(
                    target_data["target"].values, target_data["pred"].values
                )
                metrics["fold"] = fold
                metrics["target_name"] = target_name
                fold_metrics.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metrics)

    # MAEの折れ線グラフ
    fig = px.line(
        fold_metrics_df,
        x="fold",
        y="MAE",
        color="target_name",
        markers=True,
        title="Fold別のMAE",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ヒートマップ
    pivot_mae = fold_metrics_df.pivot(index="target_name", columns="fold", values="MAE")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_mae.values,
            x=pivot_mae.columns,
            y=pivot_mae.index,
            colorscale="RdYlGn_r",
            text=np.round(pivot_mae.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title="Fold別MAEヒートマップ",
        xaxis_title="Fold",
        yaxis_title="Target",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_worst_predictions(
    merged_df: pd.DataFrame, selected_targets: List[str], n_top: int = 10
):
    """予測誤差が大きいサンプルを表示"""
    st.subheader("⚠️ 予測誤差の大きいサンプル")

    merged_df["abs_error"] = np.abs(merged_df["pred"] - merged_df["target"])

    for target_name in selected_targets:
        st.markdown(f"**{target_name}**")
        target_data = merged_df[merged_df["target_name"] == target_name]
        worst = target_data.nlargest(n_top, "abs_error")

        display_cols = ["sample_id", "target", "pred", "abs_error", "fold"]
        if "State" in worst.columns:
            display_cols.append("State")
        if "Species" in worst.columns:
            display_cols.append("Species")
        if "Height_Ave_cm" in worst.columns:
            display_cols.append("Height_Ave_cm")

        st.dataframe(
            worst[display_cols].style.format(
                {
                    "target": "{:.2f}",
                    "pred": "{:.2f}",
                    "abs_error": "{:.2f}",
                    "Height_Ave_cm": "{:.2f}"
                    if "Height_Ave_cm" in worst.columns
                    else None,
                }
            ),
            use_container_width=True,
        )


def plot_predictions_with_images(
    merged_df: pd.DataFrame,
    selected_targets: List[str],
    n_top: int = 10,
    show_worst: bool = True,
):
    """予測誤差が大きい/小さいサンプルを画像付きで表示"""

    if show_worst:
        st.subheader("⚠️ 予測誤差の大きいサンプル(画像付き)")
        sort_ascending = False
    else:
        st.subheader("✅ 予測誤差の小さいサンプル(画像付き)")
        sort_ascending = True

    merged_df["abs_error"] = np.abs(merged_df["pred"] - merged_df["target"])

    for target_name in selected_targets:
        st.markdown(f"### {target_name}")
        target_data = merged_df[merged_df["target_name"] == target_name].copy()

        # 誤差でソート
        if show_worst:
            samples = target_data.nlargest(n_top, "abs_error")
        else:
            samples = target_data.nsmallest(n_top, "abs_error")

        # グリッドで表示
        n_cols = 3
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx % n_cols == 0:
                cols = st.columns(n_cols)

            col_idx = idx % n_cols
            with cols[col_idx]:
                # 画像表示
                if "image_path" in row and pd.notna(row["image_path"]):
                    img = load_image(row["image_path"])
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("画像が見つかりません")
                else:
                    st.warning("画像パスがありません")

                # 情報表示
                st.markdown(f"**Sample ID:** `{row['sample_id']}`")
                st.markdown(f"**Target:** {row['target']:.2f}")
                st.markdown(f"**Prediction:** {row['pred']:.2f}")
                st.markdown(f"**Error:** {row['abs_error']:.2f}")
                st.markdown(f"**Fold:** {row['fold']}")

                if "State" in row and pd.notna(row["State"]):
                    st.markdown(f"**State:** {row['State']}")
                if "Species" in row and pd.notna(row["Species"]):
                    st.markdown(f"**Species:** {row['Species']}")
                if "Height_Ave_cm" in row and pd.notna(row["Height_Ave_cm"]):
                    st.markdown(f"**Height:** {row['Height_Ave_cm']:.2f} cm")
                if "Pre_GSHH_NDVI" in row and pd.notna(row["Pre_GSHH_NDVI"]):
                    st.markdown(f"**NDVI:** {row['Pre_GSHH_NDVI']:.2f}")

                st.divider()

        st.markdown("---")


def main():
    st.title("🔬 OOF分析ツール")
    st.markdown(
        "実験結果のOut-of-Fold予測を分析し、targetごとのスコアや分布を可視化します"
    )

    # サイドバー
    st.sidebar.header("設定")

    # 実験フォルダの選択
    working_dir = Path("/kaggle/working")
    exp_folders = sorted(
        [
            d
            for d in working_dir.iterdir()
            if d.is_dir() and d.name.startswith("exp_") and not d.name.endswith("debug")
        ]
    )

    if not exp_folders:
        st.error("実験フォルダが見つかりません")
        return

    exp_names = [d.name for d in exp_folders]
    selected_exp = st.sidebar.selectbox(
        "実験を選択",
        exp_names,
        index=len(exp_names) - 1,  # 最新の実験をデフォルト選択
    )

    exp_folder = working_dir / selected_exp

    # データ読み込み
    with st.spinner("データを読み込んでいます..."):
        oof_df, folds = load_oof_data(exp_folder)

        if oof_df is None:
            st.error(f"{selected_exp} にOOFファイルが見つかりません")
            return

        train_df = load_train_data()
        merged_df = merge_oof_with_train(oof_df, train_df)

    st.success(
        f"✅ {selected_exp} を読み込みました ({len(merged_df)} samples, {len(folds)} folds)"
    )

    # 全体のスコアを表示
    st.header("📊 全体のスコア")
    col1, col2, col3, col4 = st.columns(4)

    overall_metrics = calculate_metrics(
        merged_df["target"].values, merged_df["pred"].values
    )

    with col1:
        st.metric("MAE", f"{overall_metrics['MAE']:.4f}")
    with col2:
        st.metric("RMSE", f"{overall_metrics['RMSE']:.4f}")
    with col3:
        st.metric("R²", f"{overall_metrics['R2']:.4f}")
    with col4:
        st.metric("MAPE", f"{overall_metrics['MAPE']:.2f}%")

    # Target選択
    st.sidebar.header("Target選択")
    target_names = sorted(merged_df["target_name"].dropna().unique())
    selected_targets = st.sidebar.multiselect(
        "分析するTargetを選択", target_names, default=target_names
    )

    if not selected_targets:
        st.warning("少なくとも1つのTargetを選択してください")
        return

    # タブで分析を整理
    tabs = st.tabs(
        [
            "Target別メトリクス",
            "予測 vs 実測",
            "残差分析",
            "特徴量別誤差",
            "Fold比較",
            "ワーストケース",
            "誤差大（画像）",
            "誤差小（画像）",
        ]
    )

    with tabs[0]:
        plot_target_metrics(merged_df)

    with tabs[1]:
        plot_prediction_scatter(merged_df, selected_targets)

    with tabs[2]:
        plot_residual_distribution(merged_df, selected_targets)

    with tabs[3]:
        plot_error_by_features(merged_df, selected_targets)

    with tabs[4]:
        plot_fold_comparison(merged_df, selected_targets)

    with tabs[5]:
        n_top = st.slider("表示するサンプル数", 5, 50, 10)
        plot_worst_predictions(merged_df, selected_targets, n_top)

    with tabs[6]:
        n_top_worst = st.slider(
            "表示するサンプル数（誤差大）", 3, 20, 6, key="worst_images"
        )
        plot_predictions_with_images(
            merged_df, selected_targets, n_top_worst, show_worst=True
        )

    with tabs[7]:
        n_top_best = st.slider(
            "表示するサンプル数（誤差小）", 3, 20, 6, key="best_images"
        )
        plot_predictions_with_images(
            merged_df, selected_targets, n_top_best, show_worst=False
        )

    # データダウンロード
    st.sidebar.header("データエクスポート")
    if st.sidebar.button("OOFデータをダウンロード"):
        csv = merged_df.to_csv(index=False)
        st.sidebar.download_button(
            label="CSVダウンロード",
            data=csv,
            file_name=f"{selected_exp}_oof_merged.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
