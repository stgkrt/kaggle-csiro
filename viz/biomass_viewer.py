"""
Biomass Data Viewer - Streamlit Application
牧草画像とバイオマス成分のメトリクスを表示するアプリケーション
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ページ設定
st.set_page_config(page_title="Biomass Data Viewer", page_icon="🌿", layout="wide")

# データパス
DATA_DIR = Path("/kaggle/input/csiro-biomass/")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

# CSSスタイル
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #4caf50;
    }
    .stMetric label {
        color: #2e7d32 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1b5e20 !important;
        font-size: 1.5rem !important;
    }
    div[data-testid="stAlert"] {
        background-color: #fff3e0;
        color: #e65100;
        border: 2px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="stAlert"] p {
        color: #e65100 !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_train_data():
    """訓練データの読み込み"""
    df = pd.read_csv(TRAIN_CSV)
    return df


@st.cache_data
def load_test_data():
    """テストデータの読み込み"""
    try:
        df = pd.read_csv(TEST_CSV)
        return df
    except Exception:
        return None


def get_unique_images(df):
    """ユニークな画像パスのリストを取得"""
    return sorted(df["image_path"].unique())


def get_image_data(df, image_path):
    """特定の画像の全データを取得"""
    return df[df["image_path"] == image_path]


def display_image_with_info(image_path, base_dir=DATA_DIR):
    """画像とその情報を表示"""
    full_path = base_dir / image_path

    if full_path.exists():
        img = Image.open(full_path)
        st.image(img, use_container_width=True)

        # 画像のメタ情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("画像サイズ", f"{img.size[0]} x {img.size[1]}")
        with col2:
            st.metric("フォーマット", img.format)
        with col3:
            st.metric("モード", img.mode)
    else:
        st.error(f"画像が見つかりません: {full_path}")


def display_biomass_metrics(data):
    """バイオマスメトリクスの表示"""
    st.subheader("🌱 バイオマス成分 (Biomass Components)")

    # メトリクスを横並びで表示
    cols = st.columns(5)

    target_labels = {
        "Dry_Green_g": "緑植物 (Green)",
        "Dry_Dead_g": "枯死物 (Dead)",
        "Dry_Clover_g": "クローバー (Clover)",
        "GDM_g": "緑乾物 (GDM)",
        "Dry_Total_g": "総乾物 (Total)",
    }

    for col, target_name in zip(cols, target_labels.keys(), strict=False):
        target_data = data[data["target_name"] == target_name]
        if not target_data.empty:
            value = target_data["target"].values[0]
            with col:
                st.metric(
                    target_labels[target_name], f"{value:.2f} g", help=f"{target_name}"
                )


def display_environmental_info(data):
    """環境情報の表示"""
    st.subheader("📍 環境・測定情報 (Environmental Data)")

    # 最初の行から情報を取得（すべての行で同じ）
    row = data.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("サンプリング日", row["Sampling_Date"])

    with col2:
        st.metric("州 (State)", row["State"])

    with col3:
        st.metric("NDVI", f"{row['Pre_GSHH_NDVI']:.3f}")

    with col4:
        st.metric("平均高さ", f"{row['Height_Ave_cm']:.2f} cm")

    # 種情報
    st.info(f"🌾 **種 (Species):** {row['Species'].replace('_', ', ')}")


def create_biomass_chart(data):
    """バイオマス成分のチャートを作成"""
    # データの整形
    chart_data = data[["target_name", "target"]].copy()

    # グラフ作成
    fig = px.bar(
        chart_data,
        x="target_name",
        y="target",
        title="バイオマス成分の内訳",
        labels={"target_name": "成分", "target": "重量 (g)"},
        color="target",
        color_continuous_scale="Greens",
    )

    fig.update_layout(
        xaxis_title="バイオマス成分",
        yaxis_title="重量 (g)",
        showlegend=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def create_comparison_chart(df, selected_images):
    """複数画像の比較チャート"""
    if len(selected_images) < 2:
        st.warning("比較には2つ以上の画像を選択してください")
        return

    # データの準備
    comparison_data = []
    for img_path in selected_images:
        img_data = df[df["image_path"] == img_path]
        for _, row in img_data.iterrows():
            comparison_data.append(
                {
                    "image": img_path.split("/")[-1].replace(".jpg", ""),
                    "target_name": row["target_name"],
                    "target": row["target"],
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    # グラフ作成
    fig = px.bar(
        comparison_df,
        x="target_name",
        y="target",
        color="image",
        barmode="group",
        title="画像間のバイオマス成分比較",
        labels={"target_name": "成分", "target": "重量 (g)", "image": "画像ID"},
    )

    fig.update_layout(xaxis_title="バイオマス成分", yaxis_title="重量 (g)", height=500)

    st.plotly_chart(fig, use_container_width=True)


def display_statistics(df):
    """統計情報の表示"""
    st.subheader("📊 データセット統計 (Dataset Statistics)")

    col1, col2, col3 = st.columns(3)

    with col1:
        unique_images = df["image_path"].nunique()
        st.metric("総画像数", unique_images)

    with col2:
        total_samples = len(df)
        st.metric("総サンプル数", total_samples)

    with col3:
        unique_states = df["State"].nunique()
        st.metric("州の数", unique_states)

    # ターゲットごとの統計
    st.subheader("成分別統計")

    target_stats = df.groupby("target_name")["target"].agg(
        ["mean", "std", "min", "max", "median"]
    )
    target_stats.columns = ["平均", "標準偏差", "最小値", "最大値", "中央値"]
    target_stats = target_stats.round(2)

    st.dataframe(target_stats, use_container_width=True)

    # 州別分布
    st.subheader("州別サンプル数")
    state_counts = (
        df.groupby("State")["image_path"].nunique().sort_values(ascending=False)
    )

    fig = px.bar(
        x=state_counts.index,
        y=state_counts.values,
        labels={"x": "州", "y": "画像数"},
        title="州別の画像数分布",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🌿 牧草バイオマス データビューア")
    st.markdown("---")

    # サイドバー
    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio(
        "ページを選択", ["画像詳細表示", "データセット統計", "画像比較", "画像一覧"]
    )

    # データ読み込み
    df_train = load_train_data()
    df_test = load_test_data()

    if page == "画像詳細表示":
        st.header("📸 画像詳細表示")

        # データセット選択
        dataset_option = st.sidebar.selectbox(
            "データセット",
            ["訓練データ (Train)", "テストデータ (Test)"]
            if df_test is not None
            else ["訓練データ (Train)"],
        )

        df = df_train if "訓練" in dataset_option else df_test

        # 画像選択
        unique_images = get_unique_images(df)

        # フィルタリングオプション
        with st.sidebar.expander("🔍 フィルター"):
            if "State" in df.columns:
                states = ["全て"] + sorted(df["State"].unique().tolist())
                selected_state = st.selectbox("州でフィルター", states)

                if selected_state != "全て":
                    filtered_images = df[df["State"] == selected_state][
                        "image_path"
                    ].unique()
                    unique_images = [
                        img for img in unique_images if img in filtered_images
                    ]

            if "Species" in df.columns:
                species_list = ["全て"] + sorted(df["Species"].unique().tolist())
                selected_species = st.selectbox("種でフィルター", species_list)

                if selected_species != "全て":
                    filtered_images = df[df["Species"] == selected_species][
                        "image_path"
                    ].unique()
                    unique_images = [
                        img for img in unique_images if img in filtered_images
                    ]

        st.sidebar.info(f"表示可能な画像: {len(unique_images)}枚")

        # 画像選択
        selected_image = st.selectbox(
            "画像を選択", unique_images, format_func=lambda x: x.split("/")[-1]
        )

        if selected_image:
            # 画像データの取得
            image_data = get_image_data(df, selected_image)

            # レイアウト
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.subheader("🖼️ 画像")
                display_image_with_info(selected_image)

            with col_info:
                if "target" in df.columns:  # 訓練データの場合
                    display_biomass_metrics(image_data)
                    st.markdown("---")
                    display_environmental_info(image_data)

            # チャート表示（訓練データのみ）
            if "target" in df.columns:
                st.markdown("---")
                create_biomass_chart(image_data)

            # 生データ表示
            with st.expander("📋 生データを表示"):
                st.dataframe(image_data, use_container_width=True)

    elif page == "データセット統計":
        st.header("📊 データセット統計")
        display_statistics(df_train)

        # 相関分析
        st.subheader("🔗 相関分析")

        # ピボットテーブルの作成
        pivot_data = df_train.pivot_table(
            index="image_path", columns="target_name", values="target"
        ).dropna()

        if not pivot_data.empty:
            corr_matrix = pivot_data.corr()

            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="バイオマス成分間の相関",
            )
            st.plotly_chart(fig, use_container_width=True)

        # NDVI vs Biomass
        st.subheader("📈 NDVI vs バイオマス")

        target_for_plot = st.selectbox("表示する成分", df_train["target_name"].unique())

        plot_data = df_train[df_train["target_name"] == target_for_plot]

        fig = px.scatter(
            plot_data,
            x="Pre_GSHH_NDVI",
            y="target",
            color="State",
            size="Height_Ave_cm",
            title=f"NDVI vs {target_for_plot}",
            labels={"Pre_GSHH_NDVI": "NDVI", "target": "重量 (g)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "画像比較":
        st.header("🔄 画像比較")

        unique_images = get_unique_images(df_train)

        # 複数選択
        selected_images = st.multiselect(
            "比較する画像を選択（複数選択可）",
            unique_images,
            max_selections=5,
            format_func=lambda x: x.split("/")[-1],
        )

        if selected_images:
            # 画像をグリッド表示
            st.subheader("選択された画像")
            cols = st.columns(min(len(selected_images), 3))

            for idx, img_path in enumerate(selected_images):
                with cols[idx % 3]:
                    st.markdown(f"**{img_path.split('/')[-1]}**")
                    full_path = DATA_DIR / img_path
                    if full_path.exists():
                        img = Image.open(full_path)
                        st.image(img, use_container_width=True)

            # 比較チャート
            st.markdown("---")
            create_comparison_chart(df_train, selected_images)

            # 比較テーブル
            st.subheader("📋 詳細比較")
            comparison_table = []
            for img_path in selected_images:
                img_data = df_train[df_train["image_path"] == img_path].iloc[0]
                row_data = {
                    "画像": img_path.split("/")[-1],
                    "サンプリング日": img_data["Sampling_Date"],
                    "州": img_data["State"],
                    "NDVI": f"{img_data['Pre_GSHH_NDVI']:.3f}",
                    "高さ(cm)": f"{img_data['Height_Ave_cm']:.2f}",
                }

                # 各成分の値を追加
                for target_name in df_train["target_name"].unique():
                    target_row = df_train[
                        (df_train["image_path"] == img_path)
                        & (df_train["target_name"] == target_name)
                    ]
                    if not target_row.empty:
                        row_data[target_name] = f"{target_row['target'].values[0]:.2f}"

                comparison_table.append(row_data)

            comparison_df = pd.DataFrame(comparison_table)
            st.dataframe(comparison_df, use_container_width=True)

    elif page == "画像一覧":
        st.header("📋 画像一覧表示")

        # データセット選択
        dataset_option = st.sidebar.selectbox(
            "データセット",
            ["訓練データ (Train)", "テストデータ (Test)"]
            if df_test is not None
            else ["訓練データ (Train)"],
        )

        df = df_train if "訓練" in dataset_option else df_test

        # フィルタリングオプション
        with st.sidebar.expander("🔍 フィルター", expanded=True):
            # ページネーション設定
            images_per_page = st.slider("1ページあたりの画像数", 6, 100, 30, step=6)

            if "State" in df.columns:
                states = ["全て"] + sorted(df["State"].unique().tolist())
                selected_state = st.selectbox("州でフィルター", states)
            else:
                selected_state = "全て"

            if "Species" in df.columns:
                species_list = ["全て"] + sorted(df["Species"].unique().tolist())
                selected_species = st.selectbox("種でフィルター", species_list)
            else:
                selected_species = "全て"

            # バイオマス範囲フィルター（訓練データのみ）
            if "target" in df.columns:
                st.markdown("**バイオマス範囲フィルター**")
                filter_by_biomass = st.checkbox("総乾物量でフィルター")
                if filter_by_biomass:
                    total_biomass_df = df[df["target_name"] == "Dry_Total_g"]
                    min_val = float(total_biomass_df["target"].min())
                    max_val = float(total_biomass_df["target"].max())
                    biomass_range = st.slider(
                        "総乾物量 (g)",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        step=10.0,
                    )

        # フィルタリング処理
        unique_images = get_unique_images(df)

        if selected_state != "全て":
            filtered_images = df[df["State"] == selected_state]["image_path"].unique()
            unique_images = [img for img in unique_images if img in filtered_images]

        if selected_species != "全て":
            filtered_images = df[df["Species"] == selected_species][
                "image_path"
            ].unique()
            unique_images = [img for img in unique_images if img in filtered_images]

        if "target" in df.columns and filter_by_biomass:
            total_biomass_df = df[df["target_name"] == "Dry_Total_g"]
            filtered_df = total_biomass_df[
                (total_biomass_df["target"] >= biomass_range[0])
                & (total_biomass_df["target"] <= biomass_range[1])
            ]
            filtered_images = filtered_df["image_path"].unique()
            unique_images = [img for img in unique_images if img in filtered_images]

        st.sidebar.info(f"表示可能な画像: {len(unique_images)}枚")

        # ページネーション
        total_images = len(unique_images)
        total_pages = (total_images - 1) // images_per_page + 1

        if total_pages > 0:
            page_number = st.number_input(
                f"ページ ({total_pages}ページ中)",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )

            start_idx = (page_number - 1) * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)

            st.info(
                f"📊 表示中: {start_idx + 1} - {end_idx} / {total_images} 枚 (ページ {page_number}/{total_pages})"
            )

            # 画像を3列のグリッドで表示
            page_images = unique_images[start_idx:end_idx]

            for row_start in range(0, len(page_images), 3):
                cols = st.columns(3)
                for col_idx, img_path in enumerate(
                    page_images[row_start : row_start + 3]
                ):
                    with cols[col_idx]:
                        # 画像表示
                        full_path = DATA_DIR / img_path
                        if full_path.exists():
                            img = Image.open(full_path)
                            st.image(img, use_container_width=True)

                            # 画像ID
                            img_id = img_path.split("/")[-1].replace(".jpg", "")
                            st.markdown(
                                f"""
                                <div style='background-color: #1a237e; padding: 8px; 
                                border-radius: 5px; text-align: center; margin-bottom: 10px;'>
                                <strong style='color: #ffffff; font-size: 1.1em;'>
                                ID: {img_id}</strong>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # 属性情報表示（訓練データのみ）
                            if "target" in df.columns:
                                image_data = get_image_data(df, img_path)
                                if not image_data.empty:
                                    row = image_data.iloc[0]

                                    # コンパクトな属性表示
                                    species_display = row["Species"][:20]
                                    if len(row["Species"]) > 20:
                                        species_display += "..."

                                    st.markdown(
                                        f"""
                                        <div style='background-color: #e3f2fd; 
                                        padding: 12px; border-radius: 8px; 
                                        border: 2px solid #1976d2; 
                                        font-size: 0.85em; margin-bottom: 10px;'>
                                        <div style='color: #0d47a1;'>
                                        <strong style='color: #1565c0;'>📅 日付:</strong> 
                                        {row["Sampling_Date"]}<br>
                                        <strong style='color: #1565c0;'>📍 州:</strong> 
                                        {row["State"]}<br>
                                        <strong style='color: #1565c0;'>🌾 種:</strong> 
                                        {species_display}<br>
                                        <strong style='color: #1565c0;'>NDVI:</strong> 
                                        {row["Pre_GSHH_NDVI"]:.3f} | 
                                        <strong style='color: #1565c0;'>高さ:</strong> 
                                        {row["Height_Ave_cm"]:.1f}cm
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # バイオマス成分表示
                                    biomass_data = {}
                                    for _, r in image_data.iterrows():
                                        biomass_data[r["target_name"]] = r["target"]

                                    # バイオマスのHTML表示
                                    green_val = biomass_data.get("Dry_Green_g", 0)
                                    dead_val = biomass_data.get("Dry_Dead_g", 0)
                                    clover_val = biomass_data.get("Dry_Clover_g", 0)
                                    gdm_val = biomass_data.get("GDM_g", 0)
                                    total_val = biomass_data.get("Dry_Total_g", 0)

                                    biomass_html = f"""
                                    <div style="background-color: #f3e5f5; padding: 12px; border-radius: 8px; border: 2px solid #7b1fa2; font-size: 0.85em;">
                                        <div style="color: #4a148c; font-weight: bold; margin-bottom: 6px;">🌱 バイオマス成分 (g)</div>
                                        <div style="color: #6a1b9a;">🌱 緑: <strong>{green_val:.1f}</strong>g | 🍂 枯: <strong>{dead_val:.1f}</strong>g</div>
                                        <div style="color: #6a1b9a;">☘️ クローバー: <strong>{clover_val:.1f}</strong>g</div>
                                        <div style="color: #6a1b9a;">🌿 GDM: <strong>{gdm_val:.1f}</strong>g | 📊 総: <strong>{total_val:.1f}</strong>g</div>
                                    </div>
                                    """
                                    st.markdown(biomass_html, unsafe_allow_html=True)
                            else:
                                # テストデータの場合
                                st.markdown(
                                    """
                                    <div style='background-color: #fafafa; 
                                    padding: 10px; border-radius: 5px; 
                                    border: 1px solid #bdbdbd; text-align: center;'>
                                    <span style='color: #757575;'>
                                    テストデータ - ラベルなし</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        st.markdown("---")
        else:
            st.warning("フィルター条件に一致する画像がありません。")


if __name__ == "__main__":
    main()
