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
    init_session_state,
    COLUNAS_NOTAS,
)

# Page configuration
st.set_page_config(layout="wide", page_title="Análise ENEM - Classificações de Notas")

# Fixed path to data
DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"
)


# Application start
def main():
    st.title("🔎 Análise de Correlação de Features com Classificações de Notas do ENEM")
    st.markdown(
        """
        Explore as **características mais comuns** associadas a classificações de nota geral do ENEM (A, B, C, D, E).
        """
    )

    df = load_preprocessed_data(DATA_PATH)

    # --- DEBUG: Check dtypes after loading ---
    # Uncomment the following lines to inspect data types for the Q columns
    # st.write("Dtypes after loading data (main.py debug):")
    # st.write(df[['Q006', 'Q022', 'Q024', 'Q008']].dtypes)
    # ----------------------------------------

    df, cat_features, num_features = identify_feature_types(df)

    init_session_state(cat_features, num_features)

    # Sidebar - filters
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
        st.session_state.selected_classifications_selected = (
            [all_classes[0]] if all_classes else []
        )

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
        "Features Categóricas (para plots e agrupamento de estatísticas):",
        cat_features,
        default=st.session_state.selected_categorical_features_selected,
        key="selected_categorical_features_widget",
    )

    if "selected_numerical_features_selected" not in st.session_state:
        st.session_state.selected_numerical_features_selected = []

    selected_num_features = st.sidebar.multiselect(
        "Features Numéricas (para plots e análise estatística adicional):",
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
                    continue

                if (
                    df_filtered_compare[feature].isnull().all()
                    or df_filtered_compare[c_type].isnull().all()
                ):
                    st.warning(
                        f"Valores ausentes em '{feature}' ou '{c_type}' impedem o agrupamento para comparação."
                    )
                    plt.close(fig)
                    continue

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
                    hue_data = prop_df[prop_df[c_type] == hue_cat].sort_values(
                        by=feature
                    )
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

                title_details = (
                    f"Distribuição Comparativa: {feature}\n"
                    f"Tipo(s) de Classificação: {', '.join(classification_types)}\n"
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

        for cls in selected_classes:
            st.markdown(
                f"### 🎯 Categoria de Nota: **'{cls}'** (dentro de {c_type.replace('_', ' ')})"
            )

            if c_type not in df.columns:
                st.warning(
                    f"Coluna de classificação '{c_type}' não encontrada no DataFrame."
                )
                continue

            df_filtered = df[df[c_type].astype(str) == str(cls)]

            if df_filtered.empty:
                st.info(
                    f"Sem dados para a categoria de nota '{cls}' na classificação '{c_type}'."
                )
                continue

            if selected_cat_features:
                st.markdown("#### 📊 Plots: Features Categóricas")
                for feature in selected_cat_features:
                    if feature in df_filtered.columns:
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
                    else:
                        st.warning(
                            f"Feature categórica '{feature}' não encontrada nos dados filtrados para '{cls}'."
                        )

            if selected_num_features:
                st.markdown("#### 📈 Plots: Features Numéricas")
                for feature in selected_num_features:
                    if feature in df_filtered.columns:
                        plot_numerical_feature(
                            df,
                            df_filtered,
                            feature,
                            c_type,
                            cls,
                            current_filters,
                        )
                    else:
                        st.warning(
                            f"Feature numérica '{feature}' não encontrada nos dados filtrados para '{cls}'."
                        )

            # --- INÍCIO DA SEÇÃO DE ESTATÍSTICAS DETALHADAS ---
            st.markdown(f"---")
            st.markdown(
                f"### 🔬 Estatísticas Descritivas Detalhadas para Categoria de Nota: **'{cls}'**"
            )

            # Define numeric features for statistical analysis, including the new Q columns
            score_cols_for_stats = [
                col
                for col in (
                    COLUNAS_NOTAS
                    + ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]
                    + ["Q006", "Q022", "Q024", "Q008"]  # Added the new columns here
                )
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

            valid_selected_num_features = [
                col
                for col in selected_num_features
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

            numerical_features_to_analyze = sorted(
                list(set(score_cols_for_stats + valid_selected_num_features))
            )

            if not numerical_features_to_analyze:
                st.info(
                    "Nenhuma feature numérica (incluindo notas gerais e questões socioeconômicas) disponível ou selecionada para análise estatística detalhada."
                )
            else:
                st.markdown(
                    "#### 📜 Estatísticas Gerais para Features Numéricas na Categoria"
                )
                if not df_filtered.empty:
                    cols_in_df_filtered = [
                        col
                        for col in numerical_features_to_analyze
                        if col in df_filtered.columns
                    ]
                    if cols_in_df_filtered:
                        try:
                            stats_df = (
                                df_filtered[cols_in_df_filtered].describe().transpose()
                            )
                            cols_to_show = [
                                "count",
                                "mean",
                                "std",
                                "min",
                                "25%",
                                "50%",
                                "75%",
                                "max",
                            ]
                            stats_df = stats_df[
                                [col for col in cols_to_show if col in stats_df.columns]
                            ]
                            st.dataframe(stats_df)
                        except Exception as e:
                            st.error(
                                f"Erro ao calcular estatísticas gerais para '{cls}': {e}"
                            )
                    else:
                        st.warning(
                            f"Nenhuma das features numéricas selecionadas para estatísticas está presente nos dados filtrados para a categoria '{cls}'."
                        )
                else:
                    st.warning(
                        f"Sem dados em `df_filtered` para a categoria '{cls}' para calcular estatísticas gerais."
                    )

                st.markdown("#### 🧮 Estatísticas Agrupadas por Feature Categórica")
                if not cat_features:
                    st.info(
                        "Nenhuma feature categórica disponível no dataset para agrupamento."
                    )
                else:
                    grouping_cat_feature = st.selectbox(
                        f"Selecione uma feature categórica para agrupar dados da categoria '{cls}' (Classificação: {c_type.replace('_',' ')}):",
                        options=["Nenhuma"] + cat_features,
                        key=f"grouping_cat_{c_type}_{cls}_{'_'.join(selected_classes)}",
                    )

                    if grouping_cat_feature and grouping_cat_feature != "Nenhuma":
                        if grouping_cat_feature not in df_filtered.columns:
                            st.warning(
                                f"Feature de agrupamento '{grouping_cat_feature}' não encontrada nos dados filtrados para '{cls}'."
                            )
                        else:
                            for num_feat_to_stat in numerical_features_to_analyze:
                                if num_feat_to_stat in df_filtered.columns:
                                    st.markdown(
                                        f"##### Estatísticas para **'{num_feat_to_stat}'** agrupado por **'{grouping_cat_feature}'**"
                                    )
                                    try:
                                        if pd.api.types.is_numeric_dtype(
                                            df_filtered[num_feat_to_stat]
                                        ):
                                            if (
                                                df_filtered[grouping_cat_feature]
                                                .notna()
                                                .any()
                                            ):
                                                grouped_stats = (
                                                    df_filtered.groupby(
                                                        grouping_cat_feature,
                                                        observed=True,
                                                    )[num_feat_to_stat]
                                                    .agg(
                                                        [
                                                            "count",
                                                            "mean",
                                                            "median",
                                                            "min",
                                                            "max",
                                                            "std",
                                                        ]
                                                    )
                                                    .reset_index()
                                                )
                                                st.dataframe(grouped_stats)
                                            else:
                                                st.warning(
                                                    f"A coluna de agrupamento '{grouping_cat_feature}' contém apenas valores ausentes para a categoria '{cls}'."
                                                )
                                        else:
                                            st.warning(
                                                f"A coluna '{num_feat_to_stat}' não é numérica nos dados filtrados para '{cls}'. Não é possível calcular estatísticas agrupadas."
                                            )
                                    except Exception as e:
                                        st.error(
                                            f"Erro ao calcular estatísticas para '{num_feat_to_stat}' agrupado por '{grouping_cat_feature}': {e}"
                                        )
                                else:
                                    st.warning(
                                        f"Feature numérica '{num_feat_to_stat}' não encontrada nos dados filtrados para '{cls}'."
                                    )
            # --- END OF DETAILED STATISTICS SECTION ---

    st.info("Use a barra lateral para refinar a análise.")


if __name__ == "__main__":
    main()
