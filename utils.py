import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

COLUNAS_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]


def load_preprocessed_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, delimiter=";", encoding="utf-8")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()


def identify_feature_types(df: pd.DataFrame):
    cat, num = [], []
    excl = set(
        COLUNAS_NOTAS
        + [
            "NOTA_GERAL_COM_REDACAO",
            "NOTA_GERAL_SEM_REDACAO",
            "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
            "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
            "NU_INSCRICAO",
            "NU_ANO",
        ]
    )
    excl.update(
        [
            col
            for col in df.columns
            if "MUNICIPIO" in col
            or "UF" in col
            or "GABARITO" in col
            or "TX_RESPOSTAS" in col
        ]
    )
    for col in df.columns:
        if col in excl:
            continue
        if df[col].dtype == "object" or col.startswith(("Q", "TP_")):
            cat.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num.append(col)
    for col in cat:
        df[col] = df[col].astype("category")
    return df, sorted(cat), sorted(num)


def init_session_state(cat, num):
    st.session_state.setdefault(
        "classification_types_selected", ["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"]
    )
    st.session_state.setdefault("selected_classifications_selected", [])
    st.session_state.setdefault("selected_categorical_features_selected", cat[:5])
    st.session_state.setdefault("selected_numerical_features_selected", num[:5])


def restore_filters(filters):
    st.session_state.classification_types_selected = filters["classification_types"]
    st.session_state.selected_classifications_selected = filters[
        "selected_classifications"
    ]
    st.session_state.selected_categorical_features_selected = filters[
        "selected_categorical_features"
    ]
    st.session_state.selected_numerical_features_selected = filters[
        "selected_numerical_features"
    ]
    st.rerun()


def plot_categorical_feature(
    df,
    df_filtered,
    feature,
    c_type,
    cls,
    current_filters,
    classification_types=None,
    selected_classes=None,
):
    """
    Plots the distribution of a categorical feature for a specific note category.

    Args:
        df (pd.DataFrame): The full preprocessed DataFrame.
        df_filtered (pd.DataFrame): The DataFrame filtered by the current classification type and note category.
        feature (str): The categorical feature to plot.
        c_type (str): The classification type being analyzed (e.g., 'CLASSIFICACAO_NOTA_GERAL_COM_REDACAO').
        cls (str): The specific note category being analyzed (e.g., '400-499').
        current_filters (dict): A dictionary of all active filters from the sidebar.
        classification_types (list, optional): List of selected classification types from the sidebar.
        selected_classes (list, optional): List of selected note categories from the sidebar.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate value counts for the selected feature in the filtered DataFrame
    # Normalize to get proportions
    feature_counts = df_filtered[feature].value_counts(normalize=True).reset_index()
    feature_counts.columns = [feature, "Proporção"]

    # Plotting the bar chart
    sns.barplot(
        data=feature_counts,
        x=feature,
        y="Proporção",
        ax=ax,
        palette="viridis",  # Using a different palette for distinctness
    )

    # Adding percentage labels on top of the bars for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="edge", padding=3)

    # Construct the detailed title
    title_details = (
        f"Distribuição de '{feature}' para Categoria de Nota '{cls}'\n"
        f"Baseado em Classificação: {c_type.replace('_', ' ')}"
    )

    # Add general filter information to the title for full context
    if classification_types:
        title_details += f"\nTipo(s) de Classificação Selecionado(s): {', '.join(classification_types).replace('_', ' ')}"
    if selected_classes:
        # Only add if it's not just the current 'cls' (which is already in the first line)
        if len(selected_classes) > 1 or selected_classes[0] != cls:
            title_details += f"\nCategorias de Nota Selecionadas (Geral): {', '.join(selected_classes)}"

    ax.set_title(
        title_details, fontsize=12, pad=20
    )  # Increased padding for multi-line title
    ax.set_ylabel("Proporção", fontsize=10)
    ax.set_xlabel(feature, fontsize=10)
    plt.xticks(rotation=45, ha="right")  # Rotate and align x-axis labels

    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    st.pyplot(fig)
    plt.close(fig)


def plot_numerical_feature(df, df_filtered, feature, c_type, cls, filters):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    sns.histplot(
        df_filtered[feature], kde=True, ax=ax[0], color="skyblue", label="Selecionada"
    )
    sns.histplot(
        df[feature], kde=True, ax=ax[0], color="orange", alpha=0.5, label="Geral"
    )
    ax[0].legend()
    ax[0].set_title("Distribuição")

    sns.boxplot(x=df_filtered[feature], ax=ax[1], color="skyblue")
    ax[1].set_title("Boxplot")

    st.pyplot(fig)
    if st.button(
        f"Salvar Numérico: {feature} - {cls}", key=f"save_num_{c_type}_{cls}_{feature}"
    ):
        add_to_gallery(fig, c_type, cls, feature, "Numérico", filters)
    plt.close(fig)


def add_to_gallery(fig, c_type, selected_classification, feature, plot_type, filters):
    if "gallery_items" not in st.session_state:
        st.session_state.gallery_items = []

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    title = f"{plot_type} - {feature} - {selected_classification} ({c_type})"
    for item in st.session_state.gallery_items:
        if item["title"] == title:
            st.warning("Gráfico já está na galeria.")
            return

    st.session_state.gallery_items.append(
        {
            "title": title,
            "image_bytes": buf.getvalue(),
            "filters": filters,
            "c_type": c_type,
            "selected_classification": selected_classification,
            "feature_name": feature,
            "plot_type": plot_type,
        }
    )
    st.success(f"Gráfico '{title}' salvo com sucesso.")
