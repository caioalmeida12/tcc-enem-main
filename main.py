import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# Importações dos módulos de análise
from analysis.data_loader import load_preprocessed_data, identify_feature_types
from analysis.feature_analyzer import (
    display_chi_squared_test,
    display_anova_test,
    display_general_numerical_stats,
    display_grouped_categorical_counts,
    display_grouped_numerical_stats,
    display_score_histogram,
)
from analysis.visualization import (
    plot_categorical_feature,
    plot_numerical_feature,
    plot_comparative_categorical_distribution,
    # Novos plots unificados
    plot_unified_categorical_feature,
    plot_unified_numerical_feature,
)

# Importações do módulo de avaliação de clusters
from analysis.clustering_eval import (
    evaluate_davies_bouldin,
    evaluate_silhouette_score,
    evaluate_calinski_harabasz_score,
    display_elbow_curve,
    display_davies_bouldin_comparison,
    display_silhouette_comparison,
    display_calinski_harabasz_comparison,
)
from utils import init_session_state

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise ENEM - Classificações de Notas")

# --- CONFIGURAÇÕES E CONSTANTES ---

DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"
)

COLUNAS_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
]

FEATURES_FOR_CLUSTERING_EVAL = COLUNAS_NOTAS + ["NU_NOTA_REDACAO"]

SILHOUETTE_SAMPLING_PERCENTAGE = 0.1
MIN_SAMPLES_FOR_SAMPLING_SILHOUETTE = 100_000


# --- APLICAÇÃO PRINCIPAL ---
def main():
    st.title(
        "🔎 Análise de Correlação de Features com a Classificação de Notas do ENEM"
    )
    st.markdown(
        "Explore as **características socioeconômicas e de prova** associadas a cada grupo de desempenho no ENEM."
    )

    try:
        df = load_preprocessed_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Arquivo de dados não encontrado em: {DATA_PATH}. Verifique o caminho."
        )
        st.stop()

    if df.empty:
        st.error(
            "Não foi possível carregar os dados. O arquivo pode estar vazio ou corrompido."
        )
        st.stop()

    df, cat_features, num_features = identify_feature_types(df)
    init_session_state(cat_features, num_features)

    with st.sidebar:
        st.header("Filtros e Opções")

        possible_classification_cols = [
            col
            for col in ["CLASSIFICACAO", "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"]
            if col in df.columns
        ]

        if not possible_classification_cols:
            st.error(
                "Nenhuma coluna de classificação encontrada. Verifique o pré-processamento."
            )
            st.stop()

        classification_types = st.multiselect(
            "Tipo(s) de Classificação:",
            possible_classification_cols,
            default=st.session_state.get(
                "classification_types_selected", [possible_classification_cols[0]]
            ),
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

        selected_classes_input = st.multiselect(
            "Filtrar por Categoria de Nota:",
            display_classes,
            default=st.session_state.get(
                "selected_classifications_selected", ["Todas"]
            ),
            key="selected_classifications_widget",
        )

        selected_classes = (
            all_classes if "Todas" in selected_classes_input else selected_classes_input
        )

        if not selected_classes:
            st.warning("Selecione ao menos uma categoria de nota (ou 'Todas').")
            st.stop()

        selected_cat_features = st.multiselect(
            "Analisar Features Categóricas:",
            cat_features,
            default=st.session_state.get("selected_categorical_features_selected", []),
            key="selected_categorical_features_widget",
        )

        selected_num_features = st.multiselect(
            "Analisar Features Numéricas:",
            num_features,
            default=st.session_state.get("selected_numerical_features_selected", []),
            key="selected_numerical_features_widget",
        )

        visualization_options = [
            "Histograma de Notas por Classificação",
            "Curva de Cotovelo (Elbow Method)",
            "Comparativo: Índice Davies-Bouldin",
            "Comparativo: Silhouette Score",
            "Comparativo: Índice Calinski-Harabasz",
            "Comparação Categórica entre Categorias",
            "Avaliação de Agrupamento (Métricas)",
            "Teste Qui-Quadrado",
            "Teste ANOVA (Seleção Manual)",
            "Plots: Features Categóricas",
            "Plots: Features Numéricas",
            # Novas opções unificadas
            "Plots: Features Categóricas Unificadas",
            "Plots: Features Numéricas Unificadas",
            "Estatísticas Gerais (Features Numéricas)",
            "Estatísticas Agrupadas (Contagens)",
            "Estatísticas Agrupadas (Médias/Medianas)",
        ]

        selected_visualizations = st.multiselect(
            "Selecione as análises a exibir:",
            options=visualization_options,
            default=["Histograma de Notas por Classificação"],
            key="visualization_selector",
        )

    current_filters = {
        "classification_types": classification_types,
        "selected_classes": selected_classes,
        "selected_categorical_features": selected_cat_features,
        "selected_numerical_features": selected_num_features,
    }

    # --- Lógica para criar a lista de dataframes para os plots unificados ---
    list_of_filtered_dfs = []
    if classification_types and selected_classes:
        for cls in selected_classes:
            df_cls = df[df[classification_types[0]].astype(str) == str(cls)].copy()
            if not df_cls.empty:
                list_of_filtered_dfs.append(df_cls)

    # Seção de Análises Unificadas
    if (
        "Plots: Features Categóricas Unificadas" in selected_visualizations
        and selected_cat_features
    ):
        st.header("Análise Unificada de Features Categóricas")
        for feature in selected_cat_features:
            plot_unified_categorical_feature(
                list_of_filtered_dfs, feature, selected_classes
            )

    if (
        "Plots: Features Numéricas Unificadas" in selected_visualizations
        and selected_num_features
    ):
        st.header("Análise Unificada de Features Numéricas")
        for feature in selected_num_features:
            plot_unified_numerical_feature(
                list_of_filtered_dfs, feature, selected_classes
            )

    # Seção de Análises Comparativas Gerais
    if "Curva de Cotovelo (Elbow Method)" in selected_visualizations:
        display_elbow_curve()
        st.markdown("---")

    if "Comparativo: Índice Davies-Bouldin" in selected_visualizations:
        display_davies_bouldin_comparison()
        st.markdown("---")

    if "Comparativo: Silhouette Score" in selected_visualizations:
        display_silhouette_comparison()
        st.markdown("---")

    if "Comparativo: Índice Calinski-Harabasz" in selected_visualizations:
        display_calinski_harabasz_comparison()
        st.markdown("---")

    # Seção de Métricas Individuais
    if "Avaliação de Agrupamento (Métricas)" in selected_visualizations:
        st.header("Métricas de Avaliação do Agrupamento")
        for c_type in classification_types:
            with st.expander(f"Avaliação para: **{c_type}**"):
                evaluate_davies_bouldin(df, FEATURES_FOR_CLUSTERING_EVAL, c_type)
                evaluate_silhouette_score(
                    df,
                    FEATURES_FOR_CLUSTERING_EVAL,
                    c_type,
                    SILHOUETTE_SAMPLING_PERCENTAGE,
                    MIN_SAMPLES_FOR_SAMPLING_SILHOUETTE,
                )
                evaluate_calinski_harabasz_score(
                    df, FEATURES_FOR_CLUSTERING_EVAL, c_type
                )

    # Seção de Análise por Tipo de Classificação
    for c_type in classification_types:
        st.header(f"Análise para Classificação: **{c_type.replace('_', ' ').title()}**")

        if "Histograma de Notas por Classificação" in selected_visualizations:
            display_score_histogram(df, c_type, selected_classes)
            st.markdown("---")

        if "Comparação Categórica entre Categorias" in selected_visualizations:
            if len(selected_classes) > 1 and selected_cat_features:
                st.subheader("📊 Comparação Categórica entre as Categorias")
                for feature in selected_cat_features:
                    plot_comparative_categorical_distribution(
                        df,
                        feature,
                        c_type,
                        selected_classes,
                        classification_types,
                        selected_classes,
                    )
            else:
                st.info(
                    "Para comparação, selecione múltiplas categorias de nota e ao menos uma feature categórica."
                )

        if "Teste Qui-Quadrado" in selected_visualizations and selected_cat_features:
            st.markdown("#### Teste Qui-Quadrado de Associação")
            for feature in selected_cat_features:
                display_chi_squared_test(
                    df, feature, c_type, selected_classes_input, selected_classes
                )
            st.markdown("---")

        for cls in selected_classes:
            st.markdown(f"### 🎯 Detalhes para a Categoria de Nota: **'{cls}'**")
            df_filtered = df[df[c_type].astype(str) == str(cls)].copy()

            if df_filtered.empty:
                st.write(f"Sem dados para a categoria '{cls}'.")
                continue

            # Garante que a NOTA_GERAL seja calculada para as estatísticas
            cols_for_mean = [
                col
                for col in FEATURES_FOR_CLUSTERING_EVAL
                if col in df_filtered.columns
            ]
            if cols_for_mean:
                df_filtered["NOTA_GERAL"] = df_filtered[cols_for_mean].mean(axis=1)

            with st.container():
                if (
                    "Plots: Features Categóricas" in selected_visualizations
                    and selected_cat_features
                ):
                    st.markdown("#### 📊 Distribuição de Features Categóricas")
                    for feature in selected_cat_features:
                        plot_categorical_feature(
                            df,
                            df_filtered,
                            feature,
                            c_type,
                            cls,
                            current_filters,
                            classification_types,
                            selected_classes,
                        )

                if (
                    "Plots: Features Numéricas" in selected_visualizations
                    and selected_num_features
                ):
                    st.markdown("#### 📈 Distribuição de Features Numéricas")
                    for feature in selected_num_features:
                        plot_numerical_feature(
                            df, df_filtered, feature, c_type, cls, current_filters
                        )

                if "Teste ANOVA (Seleção Manual)" in selected_visualizations:
                    st.markdown("#### 🧪 Teste ANOVA (Seleção Manual)")
                    display_anova_test(
                        df_filtered,
                        c_type,
                        cls,
                        cat_features,
                        num_features,
                        COLUNAS_NOTAS,
                    )

                numerical_features_to_analyze = sorted(
                    list(
                        set(
                            COLUNAS_NOTAS
                            + ["NU_NOTA_REDACAO", "NOTA_GERAL"]
                            + selected_num_features
                        )
                    )
                )

                if (
                    "Estatísticas Gerais (Features Numéricas)"
                    in selected_visualizations
                ):
                    st.markdown("#### Estatísticas Descritivas Gerais")
                    display_general_numerical_stats(
                        df_filtered, cls, numerical_features_to_analyze
                    )

                if any(
                    s.startswith("Estatísticas Agrupadas")
                    for s in selected_visualizations
                ):
                    st.markdown("#### Estatísticas Agrupadas")
                    grouping_cat_feature = st.selectbox(
                        f"Agrupar dados da categoria '{cls}' por:",
                        options=["Nenhuma"] + cat_features,
                        key=f"grouping_cat_{c_type}_{cls}",
                    )
                    if grouping_cat_feature != "Nenhuma":
                        if (
                            "Estatísticas Agrupadas (Contagens)"
                            in selected_visualizations
                        ):
                            display_grouped_categorical_counts(
                                df_filtered, cls, grouping_cat_feature
                            )
                        if (
                            "Estatísticas Agrupadas (Médias/Medianas)"
                            in selected_visualizations
                        ):
                            display_grouped_numerical_stats(
                                df_filtered,
                                cls,
                                grouping_cat_feature,
                                numerical_features_to_analyze,
                            )

            st.markdown("---")

    st.info("Use a barra lateral para refinar a análise.")


if __name__ == "__main__":
    main()
