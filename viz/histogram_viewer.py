"""
State と Month ごとの画像ヒストグラムをカラーごとに表示する Streamlit アプリケーション
"""

from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ページ設定
st.set_page_config(
    page_title="画像ヒストグラム分析", layout="wide", initial_sidebar_state="expanded"
)

# データパス
DATA_DIR = Path("/kaggle/input")
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"


@st.cache_data
def load_data():
    """CSVデータを読み込み、月情報を追加"""
    df = pd.read_csv(TRAIN_CSV)

    # 日付をパース
    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"], format="%Y/%m/%d")
    df["Year"] = df["Sampling_Date"].dt.year
    df["Month"] = df["Sampling_Date"].dt.month
    df["Month_Name"] = df["Sampling_Date"].dt.strftime("%Y-%m")

    # 画像IDを抽出（重複を除去）
    df["image_id"] = df["image_path"].str.replace("train/", "").str.replace(".jpg", "")

    # 画像単位でユニークなデータを取得
    image_df = df.drop_duplicates(subset=["image_id"])[
        ["image_id", "image_path", "State", "Year", "Month", "Month_Name", "Species"]
    ].reset_index(drop=True)

    return df, image_df


def compute_histogram(image_path, bins=256):
    """画像のRGBヒストグラムを計算"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # BGRからRGBに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 各チャンネルのヒストグラムを計算
        hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

        return {"r": hist_r.flatten(), "g": hist_g.flatten(), "b": hist_b.flatten()}
    except Exception as e:
        st.warning(f"画像読み込みエラー: {image_path} - {e}")
        return None


def plot_histogram(histograms, title="RGB Histogram"):
    """RGBヒストグラムを描画"""
    if not histograms:
        st.warning("表示するヒストグラムがありません")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    colors = ["red", "green", "blue"]
    channels = ["r", "g", "b"]
    channel_names = ["Red", "Green", "Blue"]

    for idx, (ax, color, channel, name) in enumerate(
        zip(axes, colors, channels, channel_names)
    ):
        # 全画像の平均ヒストグラムを計算
        all_hists = np.array([h[channel] for h in histograms])
        mean_hist = np.mean(all_hists, axis=0)
        std_hist = np.std(all_hists, axis=0)

        x = np.arange(256)

        # 平均を描画
        ax.plot(x, mean_hist, color=color, linewidth=2, label="Mean")

        # 標準偏差を塗りつぶし
        ax.fill_between(
            x,
            np.maximum(mean_hist - std_hist, 0),
            mean_hist + std_hist,
            color=color,
            alpha=0.2,
            label="±1 std",
        )

        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{name} Channel", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_combined_histogram(histograms, title="Combined RGB Histogram"):
    """3チャンネルを重ねて表示したヒストグラム"""
    if not histograms:
        st.warning("表示するヒストグラムがありません")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["red", "green", "blue"]
    channels = ["r", "g", "b"]
    channel_names = ["Red", "Green", "Blue"]

    for color, channel, name in zip(colors, channels, channel_names):
        all_hists = np.array([h[channel] for h in histograms])
        mean_hist = np.mean(all_hists, axis=0)

        x = np.arange(256)
        ax.plot(x, mean_hist, color=color, linewidth=2, label=name, alpha=0.7)

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    return fig


def plot_comparison_histogram(histogram_groups, labels, title="Comparison Histogram"):
    """複数グループのヒストグラムを比較表示"""
    if not histogram_groups:
        st.warning("表示するヒストグラムがありません")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors_palette = ["red", "blue", "green", "orange", "purple", "brown"]
    channels = ["r", "g", "b"]
    channel_names = ["Red", "Green", "Blue"]

    for idx, (ax, channel, name) in enumerate(zip(axes, channels, channel_names)):
        for group_idx, (histograms, label) in enumerate(zip(histogram_groups, labels)):
            if histograms:
                all_hists = np.array([h[channel] for h in histograms])
                mean_hist = np.mean(all_hists, axis=0)

                x = np.arange(256)
                color = colors_palette[group_idx % len(colors_palette)]
                ax.plot(x, mean_hist, color=color, linewidth=2, label=label, alpha=0.8)

        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{name} Channel", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def compute_histograms_for_group(image_df, image_dir, max_samples=100):
    """グループの画像からヒストグラムを計算"""
    sample_df = image_df.sample(n=min(len(image_df), max_samples), random_state=42)

    histograms = []
    for _, row in sample_df.iterrows():
        img_path = image_dir / f"{row['image_id']}.jpg"
        if img_path.exists():
            hist = compute_histogram(img_path)
            if hist is not None:
                histograms.append(hist)

    return histograms


def show_sample_images(image_paths, max_images=6):
    """サンプル画像を表示"""
    st.subheader("サンプル画像")

    cols = st.columns(min(3, len(image_paths)))

    for idx, img_path in enumerate(image_paths[:max_images]):
        col_idx = idx % 3
        with cols[col_idx]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
            except Exception as e:
                st.error(f"画像読み込みエラー: {e}")


def main():
    st.title("🎨 画像ヒストグラム分析ツール")
    st.markdown("**State** と **Month** ごとに画像のRGBヒストグラムを分析")

    # データ読み込み
    with st.spinner("データ読み込み中..."):
        df, image_df = load_data()

    st.success(f"✅ {len(image_df)} 枚の画像データを読み込みました")

    # サイドバー: モード選択
    st.sidebar.header("⚙️ 表示モード")
    view_mode = st.sidebar.radio(
        "モードを選択", options=["単一表示", "State比較", "Month比較"], index=0
    )

    # サイドバー: フィルター設定
    st.sidebar.header("📊 フィルター設定")

    if view_mode == "単一表示":
        # State選択
        states = sorted(image_df["State"].unique())
        selected_state = st.sidebar.selectbox(
            "State を選択", options=["All"] + states, index=0
        )

        # Month選択
        months = sorted(image_df["Month_Name"].unique())
        selected_month = st.sidebar.selectbox(
            "Month を選択", options=["All"] + months, index=0
        )
    elif view_mode == "State比較":
        # State複数選択
        states = sorted(image_df["State"].unique())
        selected_states = st.sidebar.multiselect(
            "比較するState を選択",
            options=states,
            default=states[: min(3, len(states))],
        )

        # Month選択
        months = sorted(image_df["Month_Name"].unique())
        selected_month = st.sidebar.selectbox(
            "Month を選択", options=["All"] + months, index=0
        )
    else:  # Month比較
        # State選択
        states = sorted(image_df["State"].unique())
        selected_state = st.sidebar.selectbox(
            "State を選択", options=["All"] + states, index=0
        )

        # Month複数選択
        months = sorted(image_df["Month_Name"].unique())
        selected_months = st.sidebar.multiselect(
            "比較するMonth を選択",
            options=months,
            default=months[: min(3, len(months))],
        )

    # Species選択（オプション）
    species_list = sorted(image_df["Species"].unique())
    selected_species = st.sidebar.multiselect(
        "Species でフィルター（オプション）", options=species_list, default=[]
    )

    # サンプル数制限
    max_samples = st.sidebar.slider(
        "ヒストグラム計算に使用する最大画像数",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

    # サンプル画像表示数
    show_samples = st.sidebar.checkbox("サンプル画像を表示", value=False)
    if show_samples:
        num_sample_images = st.sidebar.slider(
            "表示するサンプル画像数", min_value=3, max_value=12, value=6, step=3
        )

    # 比較モード処理
    if view_mode == "単一表示":
        # データフィルタリング
        filtered_df = image_df.copy()

        if selected_state != "All":
            filtered_df = filtered_df[filtered_df["State"] == selected_state]

        if selected_month != "All":
            filtered_df = filtered_df[filtered_df["Month_Name"] == selected_month]

        if selected_species:
            filtered_df = filtered_df[filtered_df["Species"].isin(selected_species)]

        # 結果表示
        st.sidebar.markdown("---")
        st.sidebar.metric("フィルター適用後の画像数", len(filtered_df))

        if len(filtered_df) == 0:
            st.warning("⚠️ 選択条件に一致する画像がありません")
            return

        # メインエリア: 統計情報
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("State", selected_state)
        with col2:
            st.metric("Month", selected_month)
        with col3:
            st.metric("画像数", len(filtered_df))
        with col4:
            st.metric("使用画像数", min(len(filtered_df), max_samples))

        # サンプリング
        sample_df = filtered_df.sample(
            n=min(len(filtered_df), max_samples), random_state=42
        )

        # ヒストグラム計算
        st.subheader("📈 ヒストグラム計算中...")
        progress_bar = st.progress(0)

        histograms = []
        image_paths = []

        for i, (_, row) in enumerate(sample_df.iterrows()):
            img_path = TRAIN_IMG_DIR / f"{row['image_id']}.jpg"

            if img_path.exists():
                hist = compute_histogram(img_path)
                if hist is not None:
                    histograms.append(hist)
                    image_paths.append(img_path)

            progress_bar.progress((i + 1) / len(sample_df))

        progress_bar.empty()

        if not histograms:
            st.error("❌ ヒストグラムを計算できませんでした")
            return

        st.success(f"✅ {len(histograms)} 枚の画像のヒストグラムを計算しました")

        # ヒストグラム表示
        st.header("📊 RGB ヒストグラム")

        # タブで切り替え
        tab1, tab2 = st.tabs(["個別チャンネル", "重ね合わせ"])

        with tab1:
            title = f"RGB Histogram - State: {selected_state}, Month: {selected_month}"
            fig1 = plot_histogram(histograms, title=title)
            st.pyplot(fig1)
            plt.close(fig1)

        with tab2:
            title = f"Combined RGB Histogram - State: {selected_state}, Month: {selected_month}"
            fig2 = plot_combined_histogram(histograms, title=title)
            st.pyplot(fig2)
            plt.close(fig2)

        # サンプル画像表示
        if show_samples and image_paths:
            st.markdown("---")
            show_sample_images(image_paths, max_images=num_sample_images)

        # データテーブル表示
        with st.expander("📋 フィルター適用後のデータ詳細"):
            st.dataframe(filtered_df, use_container_width=True)

        # ダウンロード用の統計情報
        with st.expander("📥 統計情報をダウンロード"):
            # チャンネルごとの平均ヒストグラムをデータフレームに変換
            stats_data = {
                "pixel_value": list(range(256)),
                "red_mean": np.mean([h["r"] for h in histograms], axis=0),
                "green_mean": np.mean([h["g"] for h in histograms], axis=0),
                "blue_mean": np.mean([h["b"] for h in histograms], axis=0),
                "red_std": np.std([h["r"] for h in histograms], axis=0),
                "green_std": np.std([h["g"] for h in histograms], axis=0),
                "blue_std": np.std([h["b"] for h in histograms], axis=0),
            }
            stats_df = pd.DataFrame(stats_data)

            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 CSV としてダウンロード",
                data=csv,
                file_name=f"histogram_stats_{selected_state}_{selected_month}.csv",
                mime="text/csv",
            )

    elif view_mode == "State比較":
        # State比較モード
        if not selected_states:
            st.warning("⚠️ 比較するStateを選択してください")
            return

        st.header(
            f"📊 State比較 (Month: {selected_month if selected_month != 'All' else 'All'})"
        )

        # 各Stateのヒストグラムを計算
        histogram_groups = []
        labels = []

        progress_text = st.empty()
        progress_bar = st.progress(0)

        for idx, state in enumerate(selected_states):
            progress_text.text(f"計算中: {state}...")

            filtered_df = image_df[image_df["State"] == state]

            if selected_month != "All":
                filtered_df = filtered_df[filtered_df["Month_Name"] == selected_month]

            if selected_species:
                filtered_df = filtered_df[filtered_df["Species"].isin(selected_species)]

            if len(filtered_df) > 0:
                histograms = compute_histograms_for_group(
                    filtered_df, TRAIN_IMG_DIR, max_samples
                )
                if histograms:
                    histogram_groups.append(histograms)
                    labels.append(f"{state} (n={len(histograms)})")

            progress_bar.progress((idx + 1) / len(selected_states))

        progress_text.empty()
        progress_bar.empty()

        if not histogram_groups:
            st.error("❌ ヒストグラムを計算できませんでした")
            return

        st.success(f"✅ {len(histogram_groups)} グループのヒストグラムを計算しました")

        # 比較ヒストグラム表示
        title = f"State Comparison - Month: {selected_month}"
        fig = plot_comparison_histogram(histogram_groups, labels, title=title)
        st.pyplot(fig)
        plt.close(fig)

        # 統計情報
        with st.expander("📊 グループ統計"):
            for label in labels:
                st.write(f"**{label}**")

    else:  # Month比較
        # Month比較モード
        if not selected_months:
            st.warning("⚠️ 比較するMonthを選択してください")
            return

        st.header(
            f"📊 Month比較 (State: {selected_state if selected_state != 'All' else 'All'})"
        )

        # 各Monthのヒストグラムを計算
        histogram_groups = []
        labels = []

        progress_text = st.empty()
        progress_bar = st.progress(0)

        for idx, month in enumerate(selected_months):
            progress_text.text(f"計算中: {month}...")

            filtered_df = image_df[image_df["Month_Name"] == month]

            if selected_state != "All":
                filtered_df = filtered_df[filtered_df["State"] == selected_state]

            if selected_species:
                filtered_df = filtered_df[filtered_df["Species"].isin(selected_species)]

            if len(filtered_df) > 0:
                histograms = compute_histograms_for_group(
                    filtered_df, TRAIN_IMG_DIR, max_samples
                )
                if histograms:
                    histogram_groups.append(histograms)
                    labels.append(f"{month} (n={len(histograms)})")

            progress_bar.progress((idx + 1) / len(selected_months))

        progress_text.empty()
        progress_bar.empty()

        if not histogram_groups:
            st.error("❌ ヒストグラムを計算できませんでした")
            return

        st.success(f"✅ {len(histogram_groups)} グループのヒストグラムを計算しました")

        # 比較ヒストグラム表示
        title = f"Month Comparison - State: {selected_state}"
        fig = plot_comparison_histogram(histogram_groups, labels, title=title)
        st.pyplot(fig)
        plt.close(fig)

        # 統計情報
        with st.expander("📊 グループ統計"):
            for label in labels:
                st.write(f"**{label}**")


if __name__ == "__main__":
    main()
