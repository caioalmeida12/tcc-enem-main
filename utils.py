import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- (Suas funções load_preprocessed_data, identify_feature_types, init_session_state aqui) ---
# Copiei apenas as definições para contexto, mas o foco é na nova função de plotagem.

# Define COLUNAS_NOTAS if it's not already defined globally in your utils.py
# If it's imported from another file, ensure that file exists and is correct.
COLUNAS_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]


def load_preprocessed_data(path):
    """
    Loads preprocessed data from a CSV file, explicitly converting
    known numerical columns to numeric types.
    """
    try:
        # Read all columns as strings initially to handle potential mixed types
        # or non-numeric entries gracefully.
        df = pd.read_csv(path, sep=";", dtype=str)

        # Define columns that should be numeric
        numeric_cols_to_convert = [
            "NU_NOTA_CN",
            "NU_NOTA_CH",
            "NU_NOTA_LC",
            "NU_NOTA_MT",
            "NU_NOTA_COMP1",
            "NU_NOTA_COMP2",
            "NU_NOTA_COMP3",
            "NU_NOTA_COMP4",
            "NU_NOTA_COMP5",
            "NU_NOTA_REDACAO",
            "NOTA_GERAL_COM_REDACAO",
            "NOTA_GERAL_SEM_REDACAO",
            "Q006",
            "Q022",
            "Q024",
            "Q008",  # Added the new Q columns here
        ]

        for col in numeric_cols_to_convert:
            if col in df.columns:
                # Use errors='coerce' to turn unparseable values into NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo não encontrado no caminho: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar ou pré-processar os dados: {e}")
        st.stop()


def identify_feature_types(df):
    """
    Identifies categorical and numerical features in the DataFrame.
    Converts object columns to category if they have few unique values.
    """
    cat_features = []
    num_features = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_features.append(col)
        elif df[col].nunique() < 50 and df[col].dtype == "object":
            # Convert to category if it's an object type and has few unique values
            df[col] = df[col].astype("category")
            cat_features.append(col)
        elif (
            df[col].dtype == "object"
        ):  # If it's an object and not converted to category (too many unique values)
            cat_features.append(
                col
            )  # Treat as categorical (e.g., text IDs, though not ideal for direct plotting)

    # Filter out columns that are classifications or grades as they are handled separately
    # You might want to adjust this logic based on how strictly you define 'features' for plotting
    exclude_from_features = [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
        "NOTA_GERAL_COM_REDACAO",
        "NOTA_GERAL_SEM_REDACAO",
    ] + COLUNAS_NOTAS

    cat_features = [f for f in cat_features if f not in exclude_from_features]
    num_features = [f for f in num_features if f not in exclude_from_features]

    return df, sorted(cat_features), sorted(num_features)


def init_session_state(cat_features, num_features):
    """Initializes session state variables for feature selections."""
    if "selected_categorical_features_selected" not in st.session_state:
        st.session_state.selected_categorical_features_selected = []
    if "selected_numerical_features_selected" not in st.session_state:
        st.session_state.selected_numerical_features_selected = []
    if "gallery_plots" not in st.session_state:
        st.session_state.gallery_plots = []


# --- Função de plotagem de gráfico de barras com porcentagens (já modificada anteriormente) ---
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
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))  # Increased width for labels

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
        order=overall_counts.index,  # Maintain the order by counts
    )
    axes[0].set_title(f"Distribuição Geral de {feature} (%)", fontsize=14)
    axes[0].set_xlabel("Porcentagem", fontsize=12)
    axes[0].set_ylabel(feature, fontsize=12)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].tick_params(axis="x", labelsize=10)
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)

    # Add percentage labels to overall plot
    for index, row in overall_df_plot.iterrows():
        axes[0].text(
            row["Percentage"] + 0.5,  # Slightly to the right of the bar
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
            order=overall_counts.index,  # Use the same order as overall for consistency
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

        # Add percentage labels to filtered plot
        for index, row in filtered_df_plot.iterrows():
            axes[1].text(
                row["Percentage"] + 0.5,  # Slightly to the right of the bar
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
    add_to_gallery(
        fig,
        f"Categórico: {feature} (Categoria: {selected_class})",
        filters,
        plot_type="categorical",
        feature_name=feature,
        classification_type=classification_type,
        selected_class=selected_class,
        classification_types_all=classification_types,
        selected_classes_all=selected_classes,
    )
    plt.close(fig)


# --- Função de plotagem de gráfico numérico (sem alterações para labels de porcentagem) ---
def plot_numerical_feature(
    df_original, df_filtered, feature, classification_type, selected_class, filters
):
    """
    Plots the distribution (histogram and KDE) of a numerical feature
    for the filtered data and compares it to the overall distribution.

    Note: Adding percentage labels to every bin of a histogram or points
    on a KDE curve is not standard practice as it can make the plot
    very cluttered and is less interpretable than the overall shape.
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
    add_to_gallery(
        fig,
        f"Numérico: {feature} (Categoria: {selected_class})",
        filters,
        plot_type="numerical",
        feature_name=feature,
        classification_type=classification_type,
        selected_class=selected_class,
    )
    plt.close(fig)


def add_to_gallery(
    fig,
    title,
    filters,
    plot_type,
    feature_name,
    classification_type,
    selected_class,
    classification_types_all=None,
    selected_classes_all=None,
):
    """Adds a plot to the gallery in session state."""
    # This function is meant to store plots for later display in a gallery section.
    # The current implementation only stores the plot, but you might expand it
    # to render the gallery on demand.
    plot_info = {
        "title": title,
        "plot": fig,
        "filters": filters,
        "plot_type": plot_type,
        "feature_name": feature_name,
        "classification_type": classification_type,
        "selected_class": selected_class,
        "classification_types_all": classification_types_all,
        "selected_classes_all": selected_classes_all,
    }
    # st.session_state.gallery_plots.append(plot_info) # Uncomment if you have a gallery display feature


# --- NOVA FUNÇÃO PARA PLOTAR DISTRIBUIÇÃO COMPARATIVA COM PORCENTAGENS ---
def plot_comparative_distribution_with_percentages(
    df, x_feature, grouping_feature, title_prefix="", hue_order=None
):
    """
    Plots a comparative distribution of a feature across different groups
    with percentage labels on each data point.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_feature (str): The column name for the x-axis (e.g., 'TP_COR_RACA').
        grouping_feature (str): The column name used for grouping (e.g., 'CLASSIFICACAO_NOTA_GERAL_COM_REDACAO').
        title_prefix (str): A prefix for the plot title.
        hue_order (list, optional): The order for the 'hue' categories. Defaults to None.
    """
    if df.empty:
        st.warning("DataFrame vazio para plotar a distribuição comparativa.")
        return

    # Calculate proportions
    # Group by the grouping_feature and x_feature, then calculate proportions
    # Make sure to dropna or handle NaNs appropriately for the calculation
    df_plot = df.groupby([grouping_feature, x_feature]).size().unstack(fill_value=0)
    df_proportions = (
        df_plot.apply(lambda x: x / x.sum(), axis=1)
        .stack()
        .reset_index(name="Proportion")
    )

    # Convert 'Proportion' to percentage for display
    df_proportions["Percentage"] = df_proportions["Proportion"] * 100

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the lines and fill areas
    sns.lineplot(
        data=df_proportions,
        x=x_feature,
        y="Proportion",
        hue=grouping_feature,
        marker="o",
        ax=ax,
        linewidth=2.5,
        alpha=0.7,
        hue_order=hue_order,  # Use hue_order if provided for consistent coloring/legend
    )

    # Fill the area under the lines (similar to the image provided)
    # This requires iterating through the groups
    palette = sns.color_palette(n_colors=len(df_proportions[grouping_feature].unique()))
    if hue_order:
        # Map hue_order to palette colors
        color_map = {group: palette[i] for i, group in enumerate(hue_order)}
    else:
        # Default mapping if no specific order is given
        color_map = {
            group: palette[i]
            for i, group in enumerate(df_proportions[grouping_feature].unique())
        }

    for i, group in enumerate(df_proportions[grouping_feature].unique()):
        group_data = df_proportions[
            df_proportions[grouping_feature] == group
        ].sort_values(x_feature)
        ax.fill_between(
            group_data[x_feature],
            group_data["Proportion"],
            alpha=0.2,  # Adjust transparency of the fill
            color=color_map.get(
                group, palette[i % len(palette)]
            ),  # Get color from map or default
        )

    # Add percentage labels to each point
    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        color = line.get_color()

        # Get the corresponding group (hue) for this line
        # This part assumes a direct mapping between line index and group
        # This might need refinement based on how seaborn internally orders lines for 'hue'
        # A more robust way would be to join with the original df_proportions

        group_value = df_proportions.loc[
            (df_proportions[x_feature].isin(x_data))
            & (df_proportions["Proportion"].isin(y_data)),
            grouping_feature,
        ].iloc[
            0
        ]  # Get the first group value that matches the line's data

        # Filter df_proportions to get only the data points for the current line's group
        current_line_proportions = df_proportions[
            df_proportions[grouping_feature] == group_value
        ]

        # Ensure order for matching x_data, y_data with percentages
        current_line_proportions = current_line_proportions.sort_values(x_feature)

        for j, (x, y) in enumerate(zip(x_data, y_data)):
            # Find the corresponding percentage
            percentage = current_line_proportions.loc[
                current_line_proportions[x_feature] == x, "Percentage"
            ].iloc[0]

            ax.text(
                x,
                y + 0.015,  # Slightly above the point
                f"{percentage:.1f}%",
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
            )

    ax.set_title(
        f"{title_prefix}Distribuição Comparativa: {x_feature} por {grouping_feature}",
        fontsize=16,
    )
    ax.set_xlabel(x_feature, fontsize=12)
    ax.set_ylabel("Proporção", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(title=grouping_feature, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
