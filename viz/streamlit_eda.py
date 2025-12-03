"""
Streamlit EDA App for Pasture Biomass Prediction
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(
    page_title="牧草バイオマス予測 EDA",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# データ読み込み
@st.cache_data
def load_data():
    """データの読み込みとキャッシュ"""
    train_df = pd.read_csv("/kaggle/input/csiro-biomass/train.csv")
    test_df = pd.read_csv("/kaggle/input/csiro-biomass/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/csiro-biomass/sample_submission.csv")

    # 日付を datetime に変換
    train_df["Sampling_Date"] = pd.to_datetime(train_df["Sampling_Date"])

    # 月、年、季節を追加
    train_df["Year"] = train_df["Sampling_Date"].dt.year
    train_df["Month"] = train_df["Sampling_Date"].dt.month
    train_df["Season"] = train_df["Month"].apply(
        lambda x: "Spring"
        if x in [9, 10, 11]
        else "Summer"
        if x in [12, 1, 2]
        else "Autumn"
        if x in [3, 4, 5]
        else "Winter"
    )

    return train_df, test_df, sample_submission


# データ読み込み
train_df, test_df, sample_submission = load_data()

# サイドバー
st.sidebar.title("🌱 Navigation")
page = st.sidebar.radio(
    "ページを選択",
    [
        "📊 Overview",
        "📈 Target Analysis",
        "🗺️ Geographical Analysis",
        "📅 Temporal Analysis",
        "🌿 Species Analysis",
        "📏 Feature Analysis",
        "🖼️ Image Viewer",
        "🔗 Correlation Analysis",
    ],
)

# タイトル
st.title("🌱 牧草バイオマス予測 - 探索的データ分析")
st.markdown("---")

# ================== Overview ==================
if page == "📊 Overview":
    st.header("📊 データセット概要")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("訓練サンプル数", len(train_df))
        st.metric("ユニーク画像数", train_df["image_path"].nunique())
    with col2:
        st.metric("テストサンプル数", len(test_df))
        st.metric("ターゲット種類数", train_df["target_name"].nunique())
    with col3:
        st.metric("州の数", train_df["State"].nunique())
        st.metric("種の種類数", train_df["Species"].nunique())

    st.markdown("---")

    # データセットの説明
    st.subheader("📝 コンペティション概要")
    st.markdown("""
    このコンペティションでは、牧草の画像から以下の5つのバイオマス成分を予測します:
    
    1. **Dry_Green_g**: 乾燥緑色植生（クローバーを除く）
    2. **Dry_Dead_g**: 乾燥死物質
    3. **Dry_Clover_g**: 乾燥クローバーバイオマス
    4. **GDM_g**: 緑色乾物
    5. **Dry_Total_g**: 総乾燥バイオマス
    """)

    st.markdown("---")

    # 訓練データのサンプル表示
    st.subheader("🔍 訓練データサンプル")
    st.dataframe(train_df.head(20), use_container_width=True)

    # データ型と欠損値
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 カラム情報")
        info_df = pd.DataFrame(
            {
                "Column": train_df.columns,
                "Type": train_df.dtypes.astype(str),
                "Non-Null Count": train_df.count(),
                "Null Count": train_df.isnull().sum(),
            }
        )
        st.dataframe(info_df, use_container_width=True)

    with col2:
        st.subheader("📊 基本統計量")
        st.dataframe(train_df.describe(), use_container_width=True)

# ================== Target Analysis ==================
elif page == "📈 Target Analysis":
    st.header("📈 ターゲット変数分析")

    # ターゲットごとの統計
    st.subheader("🎯 ターゲット別統計")
    target_stats = (
        train_df.groupby("target_name")["target"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .round(2)
    )
    st.dataframe(target_stats, use_container_width=True)

    # ターゲットの分布
    st.markdown("---")
    st.subheader("📊 ターゲット値の分布")

    target_names = train_df["target_name"].unique()

    # ヒストグラム
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=list(target_names),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for idx, target in enumerate(target_names):
        row = idx // 3 + 1
        col = idx % 3 + 1

        data = train_df[train_df["target_name"] == target]["target"]
        fig.add_trace(go.Histogram(x=data, name=target, nbinsx=50), row=row, col=col)

    fig.update_layout(
        height=600, showlegend=False, title_text="ターゲット値のヒストグラム"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Box plot
    st.markdown("---")
    st.subheader("📦 ターゲット値の箱ひげ図")
    fig = px.box(
        train_df,
        x="target_name",
        y="target",
        color="target_name",
        title="ターゲット別の分布",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Log scale
    st.markdown("---")
    st.subheader("📊 対数スケールでの分布")
    train_df_nonzero = train_df[train_df["target"] > 0].copy()
    train_df_nonzero["log_target"] = np.log1p(train_df_nonzero["target"])

    fig = px.histogram(
        train_df_nonzero,
        x="log_target",
        color="target_name",
        facet_col="target_name",
        facet_col_wrap=3,
        title="log(target+1)の分布",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ゼロ値の割合
    st.markdown("---")
    st.subheader("🔢 ゼロ値の割合")
    zero_ratio = (
        train_df.groupby("target_name")
        .apply(lambda x: (x["target"] == 0).sum() / len(x) * 100)
        .round(2)
    )

    fig = px.bar(
        x=zero_ratio.index,
        y=zero_ratio.values,
        labels={"x": "Target Name", "y": "Zero Ratio (%)"},
        title="ターゲット別ゼロ値の割合",
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== Geographical Analysis ==================
elif page == "🗺️ Geographical Analysis":
    st.header("🗺️ 地理的分析")

    # 画像ごとに集約（ターゲットが5行に分かれているので1行にまとめる）
    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()

    # 州別のサンプル数
    st.subheader("📍 州別サンプル数")
    col1, col2 = st.columns(2)

    with col1:
        state_counts = image_df["State"].value_counts()
        st.dataframe(state_counts.to_frame("Count"), use_container_width=True)

    with col2:
        fig = px.pie(
            values=state_counts.values,
            names=state_counts.index,
            title="州別サンプル分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 州別ターゲット分布
    st.markdown("---")
    st.subheader("📊 州別ターゲット値の分布")

    fig = px.box(
        train_df,
        x="State",
        y="target",
        color="target_name",
        title="州別・ターゲット別の値の分布",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 州別の平均値
    st.markdown("---")
    st.subheader("📈 州別・ターゲット別平均値")

    state_target_mean = train_df.pivot_table(
        values="target", index="State", columns="target_name", aggfunc="mean"
    ).round(2)

    fig = px.imshow(
        state_target_mean.T,
        labels=dict(x="State", y="Target Name", color="Mean Value"),
        title="州別・ターゲット別平均値ヒートマップ",
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(state_target_mean, use_container_width=True)

# ================== Temporal Analysis ==================
elif page == "📅 Temporal Analysis":
    st.header("📅 時系列分析")

    # 年別サンプル数
    st.subheader("📆 年別サンプル数")
    col1, col2 = st.columns(2)

    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()

    with col1:
        year_counts = image_df["Year"].value_counts().sort_index()
        st.dataframe(year_counts.to_frame("Count"), use_container_width=True)

    with col2:
        fig = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={"x": "Year", "y": "Count"},
            title="年別サンプル分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 月別サンプル数
    st.markdown("---")
    st.subheader("📅 月別サンプル数")
    col1, col2 = st.columns(2)

    with col1:
        month_counts = image_df["Month"].value_counts().sort_index()
        st.dataframe(month_counts.to_frame("Count"), use_container_width=True)

    with col2:
        fig = px.bar(
            x=month_counts.index,
            y=month_counts.values,
            labels={"x": "Month", "y": "Count"},
            title="月別サンプル分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 季節別分析
    st.markdown("---")
    st.subheader("🌸 季節別分析")

    season_counts = image_df["Season"].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(season_counts.to_frame("Count"), use_container_width=True)

    with col2:
        fig = px.pie(
            values=season_counts.values,
            names=season_counts.index,
            title="季節別サンプル分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 時系列でのターゲット値推移
    st.markdown("---")
    st.subheader("📈 時系列でのターゲット値推移")

    # 月別平均
    monthly_target = (
        train_df.groupby(["Month", "target_name"])["target"].mean().reset_index()
    )
    fig = px.line(
        monthly_target,
        x="Month",
        y="target",
        color="target_name",
        title="月別ターゲット平均値の推移",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 季節別ターゲット分布
    st.markdown("---")
    st.subheader("🍂 季節別ターゲット分布")
    fig = px.box(
        train_df,
        x="Season",
        y="target",
        color="target_name",
        title="季節別・ターゲット別の値の分布",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ================== Species Analysis ==================
elif page == "🌿 Species Analysis":
    st.header("🌿 種の分析")

    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()

    # 種別サンプル数
    st.subheader("🌱 種別サンプル数（Top 20）")
    species_counts = image_df["Species"].value_counts().head(20)

    fig = px.bar(
        x=species_counts.values,
        y=species_counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Species"},
        title="種別サンプル数 Top 20",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 種別ターゲット分布（Top 10種）
    st.markdown("---")
    st.subheader("📊 主要種別ターゲット値の分布")

    top_species = species_counts.head(10).index.tolist()
    train_df_top = train_df[train_df["Species"].isin(top_species)]

    fig = px.box(
        train_df_top,
        x="Species",
        y="target",
        color="target_name",
        title="主要種別・ターゲット別の値の分布",
    )
    fig.update_layout(height=600)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # 種別の平均値比較
    st.markdown("---")
    st.subheader("📈 主要種別・ターゲット別平均値")

    species_target_mean = train_df_top.pivot_table(
        values="target", index="Species", columns="target_name", aggfunc="mean"
    ).round(2)

    st.dataframe(species_target_mean, use_container_width=True)

    # ヒートマップ
    fig = px.imshow(
        species_target_mean.T,
        labels=dict(x="Species", y="Target Name", color="Mean Value"),
        title="種別・ターゲット別平均値ヒートマップ",
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 種の多様性
    st.markdown("---")
    st.subheader("🔢 種の多様性")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ユニーク種数", image_df["Species"].nunique())
    with col2:
        st.metric("最多種サンプル数", species_counts.iloc[0])
    with col3:
        st.metric("最少種サンプル数", species_counts.iloc[-1])

# ================== Feature Analysis ==================
elif page == "📏 Feature Analysis":
    st.header("📏 特徴量分析")

    # NDVI分析
    st.subheader("🌿 NDVI (Pre_GSHH_NDVI) 分析")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**基本統計量**")
        st.dataframe(train_df["Pre_GSHH_NDVI"].describe(), use_container_width=True)

    with col2:
        fig = px.histogram(
            train_df.drop_duplicates(subset=["image_path"]),
            x="Pre_GSHH_NDVI",
            nbins=50,
            title="NDVI分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # NDVIとターゲットの関係
    st.markdown("---")
    st.subheader("📊 NDVIとターゲット値の関係")

    fig = px.scatter(
        train_df,
        x="Pre_GSHH_NDVI",
        y="target",
        color="target_name",
        facet_col="target_name",
        facet_col_wrap=3,
        title="NDVIとターゲット値の散布図",
        opacity=0.5,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 高さ分析
    st.markdown("---")
    st.subheader("📏 牧草の高さ (Height_Ave_cm) 分析")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**基本統計量**")
        st.dataframe(train_df["Height_Ave_cm"].describe(), use_container_width=True)

    with col2:
        fig = px.histogram(
            train_df.drop_duplicates(subset=["image_path"]),
            x="Height_Ave_cm",
            nbins=50,
            title="高さ分布",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 高さとターゲットの関係
    st.markdown("---")
    st.subheader("📊 高さとターゲット値の関係")

    fig = px.scatter(
        train_df,
        x="Height_Ave_cm",
        y="target",
        color="target_name",
        facet_col="target_name",
        facet_col_wrap=3,
        title="高さとターゲット値の散布図",
        opacity=0.5,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # NDVIと高さの関係
    st.markdown("---")
    st.subheader("🔗 NDVIと高さの関係")

    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()
    fig = px.scatter(
        image_df,
        x="Pre_GSHH_NDVI",
        y="Height_Ave_cm",
        color="State",
        title="NDVIと高さの散布図",
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== Image Viewer ==================
elif page == "🖼️ Image Viewer":
    st.header("🖼️ 画像ビューア")

    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()

    # フィルター
    st.sidebar.subheader("🔍 フィルター")

    selected_state = st.sidebar.selectbox(
        "州を選択", ["All"] + sorted(image_df["State"].unique().tolist())
    )
    selected_species = st.sidebar.selectbox(
        "種を選択", ["All"] + sorted(image_df["Species"].unique().tolist())[:20]
    )

    # フィルタリング
    filtered_df = image_df.copy()
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]
    if selected_species != "All":
        filtered_df = filtered_df[filtered_df["Species"] == selected_species]

    st.write(f"フィルター後のサンプル数: {len(filtered_df)}")

    # ランダムサンプルを表示
    if len(filtered_df) > 0:
        n_images = st.slider("表示画像数", 1, min(20, len(filtered_df)), 6)

        sample_images = filtered_df.sample(n=min(n_images, len(filtered_df)))

        cols_per_row = 3
        for i in range(0, len(sample_images), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, (idx, row) in enumerate(
                list(sample_images.iloc[i : i + cols_per_row].iterrows())
            ):
                with cols[j]:
                    image_path = Path("/kaggle/input") / row["image_path"]

                    if image_path.exists():
                        try:
                            img = Image.open(image_path)
                            st.image(img, use_container_width=True)

                            # 画像情報を表示
                            st.markdown(f"""
                            **ID**: {row["image_path"].split("/")[-1].replace(".jpg", "")}  
                            **State**: {row["State"]}  
                            **Species**: {row["Species"]}  
                            **Date**: {row["Sampling_Date"].strftime("%Y-%m-%d")}  
                            **NDVI**: {row["Pre_GSHH_NDVI"]:.2f}  
                            **Height**: {row["Height_Ave_cm"]:.2f} cm
                            """)

                            # ターゲット値を表示
                            targets = train_df[
                                train_df["image_path"] == row["image_path"]
                            ][["target_name", "target"]]
                            st.dataframe(
                                targets.set_index("target_name"),
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"画像を読み込めませんでした: {e}")
                    else:
                        st.warning(f"画像が見つかりません: {image_path}")
    else:
        st.warning("フィルター条件に一致するサンプルがありません。")

# ================== Correlation Analysis ==================
elif page == "🔗 Correlation Analysis":
    st.header("🔗 相関分析")

    # ターゲット間の相関
    st.subheader("🎯 ターゲット間の相関")

    # Pivot to wide format
    target_wide = train_df.pivot_table(
        values="target", index="image_path", columns="target_name"
    )

    corr_matrix = target_wide.corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="ターゲット間の相関係数",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(corr_matrix, use_container_width=True)

    # 特徴量とターゲットの相関
    st.markdown("---")
    st.subheader("📊 特徴量とターゲット間の相関")

    # 各特徴量とターゲットの相関を計算
    feature_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm"]

    corr_results = []
    for target in train_df["target_name"].unique():
        target_df = train_df[train_df["target_name"] == target]
        for feature in feature_cols:
            corr = target_df[feature].corr(target_df["target"])
            corr_results.append(
                {"Target": target, "Feature": feature, "Correlation": corr}
            )

    corr_df = pd.DataFrame(corr_results)

    # Pivot for heatmap
    corr_pivot = corr_df.pivot(index="Target", columns="Feature", values="Correlation")

    fig = px.imshow(
        corr_pivot,
        labels=dict(color="Correlation"),
        title="特徴量とターゲット間の相関係数",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(corr_pivot, use_container_width=True)

    # ペアプロット的な分析
    st.markdown("---")
    st.subheader("📈 特徴量間の関係")

    image_df = train_df.drop_duplicates(subset=["image_path"]).copy()

    # NDVI vs Height colored by state
    fig = px.scatter(
        image_df,
        x="Pre_GSHH_NDVI",
        y="Height_Ave_cm",
        color="State",
        size="Height_Ave_cm",
        title="NDVI vs 高さ（州別）",
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ターゲット値の関係性
    st.markdown("---")
    st.subheader("🎯 ターゲット値間の散布図")

    # Dry_Total vs Dry_Green
    scatter_df = target_wide.reset_index()

    fig = px.scatter(
        scatter_df,
        x="Dry_Green_g",
        y="Dry_Total_g",
        title="Dry_Green vs Dry_Total",
        opacity=0.5,
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

    # GDM vs Dry_Green
    fig = px.scatter(
        scatter_df,
        x="Dry_Green_g",
        y="GDM_g",
        title="Dry_Green vs GDM",
        opacity=0.5,
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

# フッター
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>🌱 Pasture Biomass Prediction EDA | Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
