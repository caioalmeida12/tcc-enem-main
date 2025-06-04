import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import (
    load_preprocessed_data,
    identify_feature_types,
    plot_categorical_feature,
    plot_numerical_feature,
    add_to_gallery,
    restore_filters,
    init_session_state,
)

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise ENEM - Classificações de Notas")

# Caminho fixo para os dados
DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"
)


# Início da Aplicação
def main():
    st.title("🔎 Análise de Correlação de Features com Classificações de Notas do ENEM")
    st.markdown(
        """
        Explore as **características mais comuns** associadas a classificações de nota geral do ENEM (A, B, C, D, E).
        """
    )

    df = load_preprocessed_data(DATA_PATH)
    df, cat_features, num_features = identify_feature_types(df)

    init_session_state(cat_features, num_features)

    # Sidebar - filtros
    if "classification_types_selected" not in st.session_state:
        st.session_state.classification_types_selected = [
            "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"
        ]

    classification_types = st.sidebar.multiselect(
        "Tipo(s) de Classificação:",
        [
            "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
            "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
        ],
        default=st.session_state.classification_types_selected,
        key="classification_types_widget",
    )

    if not classification_types:
        st.warning("Selecione ao menos um tipo de classificação.")
        st.stop()

    # Coleta categorias de nota (A, B, ...)
    all_classes = sorted(
        set(
            [
                val
                for c in classification_types
                for val in df[c].dropna().astype(str).unique()
            ]
        )
    )

    display_classes = ["Todas"] + all_classes

    if "selected_classifications_selected" not in st.session_state:
        st.session_state.selected_classifications_selected = [all_classes[0]]

    selected_classes_input = st.sidebar.multiselect(
        "Categorias de Nota (A, B, ...):",
        display_classes,
        default=st.session_state.selected_classifications_selected,
        key="selected_classifications_widget",
    )

    selected_classes = (
        all_classes if "Todas" in selected_classes_input else selected_classes_input
    )

    if not selected_classes:
        st.warning("Selecione ao menos uma categoria de nota.")
        st.stop()

    if "selected_categorical_features_selected" not in st.session_state:
        st.session_state.selected_categorical_features_selected = []

    selected_cat_features = st.sidebar.multiselect(
        "Features Categóricas:",
        cat_features,
        default=st.session_state.selected_categorical_features_selected,
        key="selected_categorical_features_widget",
    )

    if "selected_numerical_features_selected" not in st.session_state:
        st.session_state.selected_numerical_features_selected = []

    selected_num_features = st.sidebar.multiselect(
        "Features Numéricas:",
        num_features,
        default=st.session_state.selected_numerical_features_selected,
        key="selected_numerical_features_widget",
    )

    current_filters = {
        "classification_types": classification_types,
        "selected_classifications": selected_classes,
        "selected_categorical_features": selected_cat_features,
        "selected_numerical_features": selected_num_features,
    }

    for c_type in classification_types:
        st.subheader(f"🔍 Classificação: **{c_type.replace('_', ' ')}**")

        if len(selected_classes) > 1 and selected_cat_features:
            st.subheader("📊 Comparação Categórica entre Categorias Selecionadas")
            for feature in selected_cat_features:
                fig, ax = plt.subplots(figsize=(12, 7))

                df_filtered = df[df[c_type].astype(str).isin(selected_classes)]

                prop_df = (
                    df_filtered.groupby([feature, c_type])
                    .size()
                    .groupby(level=0)
                    .transform(lambda x: x / x.sum())
                    .rename("Proporção")
                    .reset_index()
                )

                x_categories = prop_df[feature].unique()
                hue_categories = prop_df[c_type].unique()
                n_hue = len(hue_categories)
                x_positions = np.arange(len(x_categories))

                # --- MODIFICAÇÃO AQUI: USAR fill_between PARA GRÁFICO DE ÁREA ---
                colors = sns.color_palette("husl", n_hue)

                for i, hue_cat in enumerate(hue_categories):
                    hue_data = prop_df[prop_df[c_type] == hue_cat].sort_values(
                        by=feature
                    )

                    # Ensure alignment of hue_data with x_categories
                    aligned_proportions = [
                        (
                            hue_data[hue_data[feature] == cat]["Proporção"].iloc[0]
                            if not hue_data[hue_data[feature] == cat]["Proporção"].empty
                            else 0
                        )
                        for cat in x_categories
                    ]

                    ax.fill_between(
                        x_positions,
                        aligned_proportions,
                        color=colors[i],
                        alpha=0.4,  # Adiciona transparência
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
                # --- FIM DA MODIFICAÇÃO ---

                title_details = (
                    f"Distribuição Comparativa: {feature}\n"
                    f"Tipo(s) de Classificação: {', '.join(classification_types)}\n"
                    f"Categorias de Nota: {', '.join(selected_classes)}"
                )
                ax.set_title(title_details, fontsize=14)
                ax.set_ylabel("Proporção", fontsize=12)
                ax.set_xlabel(feature, fontsize=12)

                plt.xticks(
                    x_positions, x_categories, rotation=45, ha="right"
                )  # Define ticks explicitamente
                ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        for cls in selected_classes:
            st.markdown(f"### 🎯 Categoria: **'{cls}'**")
            df_filtered = df[df[c_type].astype(str) == cls]

            if df_filtered.empty:
                st.info("Sem dados para essa categoria.")
                continue

            if selected_cat_features:
                st.markdown("#### 📊 Features Categóricas")
                for feature in selected_cat_features:
                    plot_categorical_feature(
                        df,
                        df_filtered,
                        feature,
                        c_type,
                        cls,
                        current_filters,
                        classification_types=classification_types,
                        selected_classes=selected_classes,
                    )

            if selected_num_features:
                st.markdown("#### 📈 Features Numéricas")
                for feature in selected_num_features:
                    plot_numerical_feature(
                        df,
                        df_filtered,
                        feature,
                        c_type,
                        cls,
                        current_filters,
                        classification_types=classification_types,
                        selected_classes=selected_classes,
                    )

    st.info("Use a barra lateral para refinar a análise.")


if __name__ == "__main__":
    main()
