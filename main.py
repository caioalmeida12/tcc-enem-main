import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)

# Importações dos novos módulos
from analysis.data_loader import load_preprocessed_data, identify_feature_types
from analysis.feature_analyzer import (
    display_chi_squared_test,
    display_anova_test,
    display_general_numerical_stats,
    display_grouped_categorical_counts,
    display_grouped_numerical_stats,
)
from analysis.visualization import (
    plot_categorical_feature,
    plot_numerical_feature,
    plot_comparative_categorical_distribution,
)
from analysis.clustering_eval import (
    evaluate_davies_bouldin,
    evaluate_silhouette_score,
    evaluate_calinski_harabasz_score,
)
from utils import init_session_state, COLUNAS_NOTAS


# Page configuration
st.set_page_config(layout="wide", page_title="Análise ENEM - Classificações de Notas")

# Fixed path to data
DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"
)

# Lista de features categóricas pré-definidas para a ANOVA comparativa
ANOVA_PREDEFINED_CAT_FEATURES = [
    "TP_COR_RACA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_SEXO",
    "TP_LINGUA",
    "TP_ESCOLA",
]

# Definindo as features que provavelmente foram usadas para o Birch
FEATURES_FOR_CLUSTERING_EVAL = COLUNAS_NOTAS + ["NU_NOTA_REDACAO"]

# --- Configuração para amostragem do Silhouette Score ---
SILHOUETTE_SAMPLING_PERCENTAGE = 0.1
MIN_SAMPLES_FOR_SAMPLING_SILHOUETTE = 100_000


# Application start
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

    # --- Sidebar - Filters ---
    with st.sidebar:
        st.header("Filtros e Opções")

        if "classification_types_selected" not in st.session_state:
            st.session_state.classification_types_selected = [
                "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"
            ]

        classification_types = st.multiselect(
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

        selected_classes_input = st.multiselect(
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

        selected_cat_features = st.multiselect(
            "Features Categóricas (para plots e agrupamento de estatísticas):",
            cat_features,
            default=st.session_state.selected_categorical_features_selected,
            key="selected_categorical_features_widget",
        )

        if "selected_numerical_features_selected" not in st.session_state:
            st.session_state.selected_numerical_features_selected = []

        selected_num_features = st.multiselect(
            "Features Numéricas (para plots e análise estatística adicional):",
            num_features,
            default=st.session_state.selected_numerical_features_selected,
            key="selected_numerical_features_widget",
        )

        visualization_options = [
            "Comparação Categórica entre Categorias Selecionadas",
            "Plots: Features Categóricas",
            "Teste Qui-Quadrado",
            "Teste ANOVA (Seleção Manual)",
            "Análise ANOVA Comparativa (Pré-definida)",
            "Avaliação de Agrupamento (Davies-Bouldin)",
            "Avaliação de Agrupamento (Silhouette Score)",
            "Avaliação de Agrupamento (Calinski-Harabasz)",
            "Plots: Features Numéricas",
            "Estatísticas Gerais para Features Numéricas na Categoria",
            "Estatísticas Agrupadas por Feature Categórica (contagens)",
            "Estatísticas Agrupadas por Feature Categórica (médias/medianas)",
        ]

        if "selected_visualizations" not in st.session_state:
            st.session_state.selected_visualizations = [
                "Estatísticas Gerais para Features Numéricas na Categoria"
            ]

        selected_visualizations = st.multiselect(
            "Selecione as visualizações a exibir:",
            options=visualization_options,
            default=st.session_state.selected_visualizations,
            key="visualization_selector",
        )

    current_filters = {
        "classification_types": classification_types,
        "selected_classifications": selected_classes,
        "selected_categorical_features": selected_cat_features,
        "selected_numerical_features": selected_num_features,
    }

    # --- Avaliação de Agrupamento ---
    if "Avaliação de Agrupamento (Davies-Bouldin)" in selected_visualizations:
        evaluate_davies_bouldin(df, FEATURES_FOR_CLUSTERING_EVAL)

    if "Avaliação de Agrupamento (Silhouette Score)" in selected_visualizations:
        evaluate_silhouette_score(
            df,
            FEATURES_FOR_CLUSTERING_EVAL,
            SILHOUETTE_SAMPLING_PERCENTAGE,
            MIN_SAMPLES_FOR_SAMPLING_SILHOUETTE,
        )

    if "Avaliação de Agrupamento (Calinski-Harabasz)" in selected_visualizations:
        evaluate_calinski_harabasz_score(df, FEATURES_FOR_CLUSTERING_EVAL)

    # --- Seções de Análise por Tipo de Classificação e Categoria ---
    for c_type in classification_types:
        st.subheader(f"🔍 Classificação: **{c_type.replace('_', ' ')}**")

        if (
            "Comparação Categórica entre Categorias Selecionadas"
            in selected_visualizations
        ):
            if len(selected_classes) > 1 and selected_cat_features:
                st.subheader("📊 Comparação Categórica entre Categorias Selecionadas")
                for feature in selected_cat_features:
                    plot_comparative_categorical_distribution(
                        df,
                        feature,
                        c_type,
                        selected_classes,
                        classification_types,
                        selected_classes,
                    )
            elif (
                "Comparação Categórica entre Categorias Selecionadas"
                in selected_visualizations
            ):
                st.info(
                    "Selecione mais de uma categoria de nota e features categóricas para ver a comparação."
                )

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

            if "Plots: Features Categóricas" in selected_visualizations:
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
                else:
                    st.info(
                        "Selecione features categóricas para ver os plots categóricos."
                    )

            if "Teste Qui-Quadrado" in selected_visualizations:
                if selected_cat_features:
                    st.markdown("#### Teste Qui-Quadrado")
                    for feature in selected_cat_features:
                        display_chi_squared_test(
                            df,
                            feature,
                            c_type,
                            selected_classes_input,
                            selected_classes,
                        )
                else:
                    st.info(
                        "Selecione features categóricas para realizar o teste Qui-Quadrado."
                    )

            if "Teste ANOVA (Seleção Manual)" in selected_visualizations:
                st.markdown("#### 🧪 Teste ANOVA (Seleção Manual)")
                st.write(
                    "Compare as médias de uma feature numérica entre grupos definidos por uma feature categórica."
                )
                display_anova_test(
                    df_filtered, c_type, cls, cat_features, num_features, COLUNAS_NOTAS
                )

            if "Plots: Features Numéricas" in selected_visualizations:
                if selected_num_features:
                    st.markdown("#### 📈 Plots: Features Numéricas")
                    for feature in selected_num_features:
                        if feature in df_filtered.columns:
                            plot_numerical_feature(
                                df, df_filtered, feature, c_type, cls, current_filters
                            )
                        else:
                            st.warning(
                                f"Feature numérica '{feature}' não encontrada nos dados filtrados para '{cls}'."
                            )
                else:
                    st.info("Selecione features numéricas para ver os plots numéricos.")

            st.markdown(f"---")
            st.markdown(
                f"### 🔬 Estatísticas Descritivas Detalhadas para Categoria de Nota: **'{cls}'**"
            )

            score_cols_for_stats = [
                col
                for col in (
                    COLUNAS_NOTAS
                    + ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]
                    + ["Q006", "Q022", "Q024", "Q008"]
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

            if (
                "Estatísticas Gerais para Features Numéricas na Categoria"
                in selected_visualizations
            ):
                display_general_numerical_stats(
                    df_filtered, cls, numerical_features_to_analyze
                )

            if (
                "Estatísticas Agrupadas por Feature Categórica (contagens)"
                in selected_visualizations
                or "Estatísticas Agrupadas por Feature Categórica (médias/medianas)"
                in selected_visualizations
            ):

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
                        if (
                            "Estatísticas Agrupadas por Feature Categórica (contagens)"
                            in selected_visualizations
                        ):
                            display_grouped_categorical_counts(
                                df_filtered, cls, grouping_cat_feature
                            )

                        if (
                            "Estatísticas Agrupadas por Feature Categórica (médias/medianas)"
                            in selected_visualizations
                        ):
                            display_grouped_numerical_stats(
                                df_filtered,
                                cls,
                                grouping_cat_feature,
                                numerical_features_to_analyze,
                            )
                    else:
                        st.info(
                            "Selecione uma feature categórica para ver as estatísticas agrupadas."
                        )

    st.info(
        "Use a barra lateral para refinar a análise e selecionar as visualizações desejadas."
    )


if __name__ == "__main__":
    main()
