import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_categorical_feature(
    df_original,
    df_filtered,
    feature,
    classification_type,
    selected_class,
    filters,
    classification_types,
    selected_classes,
):
    """
    Plots the distribution of a categorical feature for the filtered data
    and compares it to the overall distribution, with percentage labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Overall distribution
    overall_counts = df_original[feature].value_counts()
    overall_percentages = (overall_counts / len(df_original)) * 100
    overall_df_plot = overall_percentages.reset_index()
    overall_df_plot.columns = [feature, "Percentage"]

    sns.barplot(
        data=overall_df_plot,
        x="Percentage",
        y=feature,
        ax=axes[0],
        palette="viridis",
        order=overall_counts.index,
    )
    axes[0].set_title(f"Distribuição Geral de {feature} (%)", fontsize=14)
    axes[0].set_xlabel("Porcentagem", fontsize=12)
    axes[0].set_ylabel(feature, fontsize=12)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].tick_params(axis="x", labelsize=10)
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)

    for index, row in overall_df_plot.iterrows():
        axes[0].text(
            row["Percentage"] + 0.5,
            index,
            f"{row['Percentage']:.1f}%",
            color="black",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Filtered distribution
    if not df_filtered.empty and feature in df_filtered.columns:
        filtered_counts = df_filtered[feature].value_counts()
        filtered_percentages = (filtered_counts / len(df_filtered)) * 100
        filtered_df_plot = filtered_percentages.reset_index()
        filtered_df_plot.columns = [feature, "Percentage"]

        sns.barplot(
            data=filtered_df_plot,
            x="Percentage",
            y=feature,
            ax=axes[1],
            palette="magma",
            order=overall_counts.index,
        )
        axes[1].set_title(
            f"Distribuição de {feature} para a Categoria '{selected_class}' (%)",
            fontsize=14,
        )
        axes[1].set_xlabel("Porcentagem", fontsize=12)
        axes[1].set_ylabel(feature, fontsize=12)
        axes[1].tick_params(axis="y", labelsize=10)
        axes[1].tick_params(axis="x", labelsize=10)
        axes[1].grid(axis="x", linestyle="--", alpha=0.7)

        for index, row in filtered_df_plot.iterrows():
            axes[1].text(
                row["Percentage"] + 0.5,
                index,
                f"{row['Percentage']:.1f}%",
                color="black",
                ha="left",
                va="center",
                fontsize=9,
            )
    else:
        axes[1].set_title(
            f"Sem dados para {feature} na categoria '{selected_class}'", fontsize=14
        )
        axes[1].text(
            0.5,
            0.5,
            "Dados insuficientes para plotar",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[1].transAxes,
            fontsize=12,
            color="gray",
        )

    plt.tight_layout()
    st.pyplot(fig)

    plt.close(fig)


def plot_numerical_feature(
    df_original, df_filtered, feature, classification_type, selected_class, filters
):
    """
    Plots the distribution (histogram and KDE) of a numerical feature
    for the filtered data and compares it to the overall distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Overall distribution
    sns.histplot(
        df_original[feature].dropna(), kde=True, ax=axes[0], color="skyblue", bins=30
    )
    axes[0].set_title(f"Distribuição Geral de {feature}", fontsize=14)
    axes[0].set_xlabel(feature, fontsize=12)
    axes[0].set_ylabel("Densidade / Contagem", fontsize=12)
    axes[0].tick_params(axis="x", labelsize=10)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Filtered distribution
    if not df_filtered.empty and feature in df_filtered.columns:
        sns.histplot(
            df_filtered[feature].dropna(), kde=True, ax=axes[1], color="salmon", bins=30
        )
        axes[1].set_title(
            f"Distribuição de {feature} para a Categoria '{selected_class}'",
            fontsize=14,
        )
        axes[1].set_xlabel(feature, fontsize=12)
        axes[1].set_ylabel("Densidade / Contagem", fontsize=12)
        axes[1].tick_params(axis="x", labelsize=10)
        axes[1].tick_params(axis="y", labelsize=10)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1].set_title(
            f"Sem dados para {feature} na categoria '{selected_class}'", fontsize=14
        )
        axes[1].text(
            0.5,
            0.5,
            "Dados insuficientes para plotar",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[1].transAxes,
            fontsize=12,
            color="gray",
        )

    plt.tight_layout()
    st.pyplot(fig)

    plt.close(fig)


def plot_comparative_categorical_distribution(
    df,
    feature,
    c_type,
    selected_classes,
    all_classification_types,
    all_selected_classes,
):
    """
    Plots a comparative distribution of a categorical feature across selected
    classification categories with percentage labels.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    df_filtered_compare = df[df[c_type].astype(str).isin(selected_classes)]

    if (
        df_filtered_compare.empty
        or feature not in df_filtered_compare.columns
        or c_type not in df_filtered_compare.columns
    ):
        st.warning(
            f"Dados insuficientes ou coluna '{feature}'/'{c_type}' ausente para comparação da feature '{feature}'."
        )
        plt.close(fig)
        return

    if (
        df_filtered_compare[feature].isnull().all()
        or df_filtered_compare[c_type].isnull().all()
    ):
        st.warning(
            f"Valores ausentes em '{feature}' ou '{c_type}' impedem o agrupamento para comparação."
        )
        plt.close(fig)
        return

    prop_df = (
        df_filtered_compare.groupby([feature, c_type], observed=True)
        .size()
        .groupby(level=0, observed=True)
        .transform(lambda x: x / x.sum())
        .rename("Proporção")
        .reset_index()
    )

    x_categories = prop_df[feature].unique()
    hue_categories = prop_df[c_type].unique()
    n_hue = len(hue_categories)
    x_positions = np.arange(len(x_categories))
    colors = sns.color_palette("husl", n_hue)

    for i, hue_cat in enumerate(hue_categories):
        hue_data = prop_df[prop_df[c_type] == hue_cat].sort_values(by=feature)
        aligned_proportions = []
        for cat_val in x_categories:
            val_series = hue_data[hue_data[feature] == cat_val]["Proporção"]
            aligned_proportions.append(
                val_series.iloc[0] if not val_series.empty else 0
            )

        ax.fill_between(
            x_positions,
            aligned_proportions,
            color=colors[i],
            alpha=0.4,
            label=f"{hue_cat}",
            zorder=1,
        )
        ax.plot(
            x_positions,
            aligned_proportions,
            marker="o",
            color=colors[i],
            linewidth=2.5,
            zorder=2,
        )
        for j, prop in enumerate(aligned_proportions):
            if prop > 0:  # Only add label if proportion is greater than zero
                ax.text(
                    x_positions[j],
                    prop + 0.01,  # Slightly above the point
                    f"{prop:.1%}",
                    color=colors[i],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

    title_details = (
        f"Distribuição Comparativa: {feature}\n"
        f"Tipo(s) de Classificação: {', '.join(all_classification_types)}\n"
        f"Categorias de Nota: {', '.join(selected_classes)}"
    )
    ax.set_title(title_details, fontsize=14)
    ax.set_ylabel("Proporção", fontsize=12)
    ax.set_xlabel(feature, fontsize=12)
    plt.xticks(x_positions, x_categories, rotation=45, ha="right")
    ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
