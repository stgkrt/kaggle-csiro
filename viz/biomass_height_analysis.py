from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ページの設定
st.set_page_config(
    page_title="CSIRO Biomass - Height Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 CSIRO Biomass - Height_Ave_cm Analysis")


# データの読み込み
@st.cache_data
def load_data():
    train_csv_path = Path("/kaggle/input/csiro-biomass/train.csv")
    df = pd.read_csv(train_csv_path)
    return df


df = load_data()

# サイドバーでのフィルタリング
st.sidebar.header("Filters")

# 基本統計情報
st.sidebar.subheader("Dataset Info")
st.sidebar.metric("Total Samples", len(df))
st.sidebar.metric("Unique Images", df["image_path"].nunique())
st.sidebar.metric(
    "Height Range (cm)",
    f"{df['Height_Ave_cm'].min():.1f} - {df['Height_Ave_cm'].max():.1f}",
)

# フィルタ条件
state_filter = st.sidebar.multiselect(
    "Select State(s):",
    sorted(df["State"].unique()),
    default=sorted(df["State"].unique()),
)

species_filter = st.sidebar.multiselect(
    "Select Species:",
    sorted(df["Species"].unique()),
    default=sorted(df["Species"].unique()),
)

target_filter = st.sidebar.multiselect(
    "Select Target:",
    sorted(df["target_name"].unique()),
    default=sorted(df["target_name"].unique()),
)

# フィルター適用
filtered_df = df[
    (df["State"].isin(state_filter))
    & (df["Species"].isin(species_filter))
    & (df["target_name"].isin(target_filter))
].copy()

# メインコンテンツ
st.sidebar.info(f"Filtered samples: {len(filtered_df)}")

# タブの作成
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "📈 Distribution",
        "🔗 Correlation",
        "🎯 Target Analysis",
        "📍 Geographic Analysis",
    ]
)

# Tab 1: Overview
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Height (cm)", f"{filtered_df['Height_Ave_cm'].mean():.2f}")
    with col2:
        st.metric("Median Height (cm)", f"{filtered_df['Height_Ave_cm'].median():.2f}")
    with col3:
        st.metric("Std Dev (cm)", f"{filtered_df['Height_Ave_cm'].std():.2f}")
    with col4:
        st.metric("Max Height (cm)", f"{filtered_df['Height_Ave_cm'].max():.2f}")

    st.subheader("Height Statistics by Target")
    height_stats = (
        filtered_df.groupby("target_name")["Height_Ave_cm"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .round(2)
    )
    st.dataframe(height_stats, use_container_width=True)

# Tab 2: Distribution
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        # ヒストグラム
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            filtered_df["Height_Ave_cm"],
            bins=50,
            color="#2ecc71",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel("Height (cm)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Height_Ave_cm", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        # ボックスプロット by Target
        fig, ax = plt.subplots(figsize=(10, 6))
        filtered_df.boxplot(column="Height_Ave_cm", by="target_name", ax=ax)
        plt.suptitle("")
        ax.set_title("Height Distribution by Target", fontsize=14, fontweight="bold")
        ax.set_xlabel("Target Name", fontsize=12)
        ax.set_ylabel("Height (cm)", fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Tab 3: Correlation
with tab3:
    st.subheader("Height vs Target Values")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            filtered_df["Height_Ave_cm"],
            filtered_df["target"],
            c=filtered_df["target_name"].astype("category").cat.codes,
            cmap="viridis",
            alpha=0.6,
            s=50,
        )
        ax.set_xlabel("Height (cm)", fontsize=12)
        ax.set_ylabel("Target Value (g)", fontsize=12)
        ax.set_title("Height vs Biomass Target", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Target Type")
        st.pyplot(fig)

    with col2:
        # 相関係数の計算
        correlation = filtered_df["Height_Ave_cm"].corr(filtered_df["target"])

        fig, ax = plt.subplots(figsize=(10, 6))
        for target_name in filtered_df["target_name"].unique():
            target_data = filtered_df[filtered_df["target_name"] == target_name]
            corr = target_data["Height_Ave_cm"].corr(target_data["target"])
            ax.scatter(
                target_data["Height_Ave_cm"],
                target_data["target"],
                label=f"{target_name} (r={corr:.3f})",
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("Height (cm)", fontsize=12)
        ax.set_ylabel("Target Value (g)", fontsize=12)
        ax.set_title("Height vs Target by Category", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.metric("Overall Correlation (Height vs Target)", f"{correlation:.4f}")

# Tab 4: Target Analysis
with tab4:
    st.subheader("Height Statistics by Target Type")

    target_heights = (
        filtered_df.groupby("target_name")
        .agg(
            {
                "Height_Ave_cm": ["mean", "median", "std", "min", "max"],
                "target": ["mean", "median", "std"],
            }
        )
        .round(2)
    )
    st.dataframe(target_heights, use_container_width=True)

    # ヒートマップ
    st.subheader("Height by Target - Distribution Heatmap")

    pivot_data = filtered_df.pivot_table(
        values="Height_Ave_cm",
        index="target_name",
        columns=pd.cut(filtered_df["Height_Ave_cm"], bins=10),
        aggfunc="count",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot_data, annot=True, fmt="g", cmap="YlGn", ax=ax, cbar_kws={"label": "Count"}
    )
    ax.set_title(
        "Height Distribution Heatmap by Target", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Height Range (cm)", fontsize=12)
    ax.set_ylabel("Target Type", fontsize=12)
    st.pyplot(fig)

# Tab 5: Geographic Analysis
with tab5:
    st.subheader("Height Analysis by State and Species")

    col1, col2 = st.columns(2)

    with col1:
        # State別分析
        fig, ax = plt.subplots(figsize=(10, 6))
        state_data = filtered_df.groupby("State")["Height_Ave_cm"].apply(list)
        ax.boxplot(state_data.values, labels=state_data.index)
        ax.set_ylabel("Height (cm)", fontsize=12)
        ax.set_title("Height Distribution by State", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        # Species別分析
        fig, ax = plt.subplots(figsize=(10, 6))
        species_data = filtered_df.groupby("Species")["Height_Ave_cm"].apply(list)
        ax.boxplot(
            species_data.values,
            labels=[s[:15] + "..." if len(s) > 15 else s for s in species_data.index],
        )
        ax.set_ylabel("Height (cm)", fontsize=12)
        ax.set_title("Height Distribution by Species", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # State x Target クロス分析
    st.subheader("Average Height by State and Target")
    state_target = filtered_df.pivot_table(
        values="Height_Ave_cm", index="State", columns="target_name", aggfunc="mean"
    ).round(2)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        state_target,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Height (cm)"},
    )
    ax.set_title(
        "Average Height (cm) by State and Target", fontsize=14, fontweight="bold"
    )
    st.pyplot(fig)

# Footer
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📊 Dataset: {len(filtered_df)} samples")
with col2:
    st.info(f"📅 Date Range: CSIRO Biomass Dataset")
with col3:
    st.info(
        f"🎯 Height Range: {filtered_df['Height_Ave_cm'].min():.1f}-{filtered_df['Height_Ave_cm'].max():.1f} cm"
    )
