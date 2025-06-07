import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import numpy as np


def display_chi_squared_test(
    df, feature, c_type, selected_classes_input, selected_classes
):
    """Displays Chi-Squared test results for a given categorical feature."""
    st.markdown(
        f"##### Teste Qui-Quadrado para '{feature}' vs. '{c_type.replace('_', ' ')}'"
    )

    contingency_table_full = pd.crosstab(df[feature], df[c_type])

    if "Todas" not in selected_classes_input and selected_classes:
        if c_type in contingency_table_full.columns:
            contingency_table = contingency_table_full[selected_classes]
        else:
            contingency_table = contingency_table_full
    else:
        contingency_table = contingency_table_full

    if (
        contingency_table.empty
        or contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
        or (contingency_table == 0).any().any()
    ):
        st.info(
            f"Dados insuficientes ou tabela de contingência com células zero/dimensões insuficientes para o teste Qui-Quadrado de '{feature}' vs. '{c_type.replace('_', ' ')}'. Certifique-se de que há variação suficiente nas categorias e que a amostra não é muito pequena para as categorias selecionadas."
        )
    else:
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            st.write(f"**Valor Qui-Quadrado:** {chi2:.2f}")
            st.write(f"**p-valor:** {p_value:.5f}")
            st.write(f"**Graus de liberdade:** {dof}")

            if p_value < 0.05:
                st.success(
                    f"Há uma associação estatisticamente significativa entre '{feature}' e '{c_type.replace('_', ' ')}' (p < 0.05)."
                )
            else:
                st.info(
                    f"Não há evidência de associação estatisticamente significativa entre '{feature}' e '{c_type.replace('_', ' ')}' (p >= 0.05)."
                )
            st.markdown(f"**Tabela de Contingência (Contagens Observadas):**")
            st.dataframe(contingency_table)
            with st.expander("Ver Valores Esperados"):
                st.dataframe(
                    pd.DataFrame(
                        expected,
                        index=contingency_table.index,
                        columns=contingency_table.columns,
                    )
                )
        except ValueError as ve:
            st.warning(
                f"Não foi possível realizar o teste Qui-Quadrado para '{feature}' vs. '{c_type.replace('_', ' ')}' devido a: {ve}. Isso pode ocorrer se houver categorias com zero observações na tabela de contingência após a filtragem. Tente selecionar mais categorias de nota ou revisar os filtros."
            )
        except Exception as e:
            st.error(
                f"Ocorreu um erro inesperado ao calcular o Qui-Quadrado para '{feature}' vs. '{c_type.replace('_', ' ')}': {e}"
            )


def display_anova_test(
    df_filtered, c_type, cls, cat_features, num_features, COLUNAS_NOTAS
):
    """Displays ANOVA test results based on manual selection."""
    anova_num_feature_options = sorted(
        list(
            set(
                num_features
                + COLUNAS_NOTAS
                + [
                    "NOTA_GERAL_COM_REDACAO",
                    "NOTA_GERAL_SEM_REDACAO",
                    "Q006",
                    "Q022",
                    "Q024",
                    "Q008",
                ]
            )
        )
    )
    anova_num_feature = st.selectbox(
        f"Selecione a Feature Numérica (dependente) para ANOVA na categoria '{cls}':",
        options=["Nenhuma"]
        + [
            f
            for f in anova_num_feature_options
            if f in df_filtered.columns
            and pd.api.types.is_numeric_dtype(df_filtered[f])
        ],
        key=f"anova_manual_num_feature_{c_type}_{cls}",
    )

    anova_cat_feature = st.selectbox(
        f"Selecione a Feature Categórica (independente/agrupamento) para ANOVA na categoria '{cls}':",
        options=["Nenhuma"] + [f for f in cat_features if f in df_filtered.columns],
        key=f"anova_manual_cat_feature_{c_type}_{cls}",
    )

    if anova_num_feature != "Nenhuma" and anova_cat_feature != "Nenhuma":
        st.markdown(
            f"##### ANOVA para '{anova_num_feature}' agrupado por '{anova_cat_feature}'"
        )

        df_anova = df_filtered[[anova_num_feature, anova_cat_feature]].dropna()

        if df_anova.empty:
            st.warning(
                f"Não há dados suficientes para realizar ANOVA com as seleções '{anova_num_feature}' e '{anova_cat_feature}' para a categoria '{cls}'."
            )
        else:
            try:
                groups = [
                    df_anova[anova_num_feature][
                        df_anova[anova_cat_feature] == category
                    ].dropna()
                    for category in df_anova[anova_cat_feature].unique()
                ]

                valid_groups = [g for g in groups if g.count() > 1]

                if len(valid_groups) < 2:
                    st.warning(
                        f"Não há grupos suficientes (mínimo de 2 grupos com mais de uma observação) para realizar a ANOVA para '{anova_num_feature}' agrupado por '{anova_cat_feature}' na categoria '{cls}'."
                    )
                else:
                    f_statistic, p_value = f_oneway(*valid_groups)

                    st.write(f"**F-Estatística:** {f_statistic:.2f}")
                    st.write(f"**p-valor:** {p_value:.5f}")

                    if p_value < 0.05:
                        st.success(
                            f"Há uma diferença estatisticamente significativa nas médias de '{anova_num_feature}' entre os grupos de '{anova_cat_feature}' (p < 0.05)."
                        )
                    else:
                        st.info(
                            f"Não há evidência de diferença estatisticamente significativa nas médias de '{anova_num_feature}' entre os grupos de '{anova_cat_feature}' (p >= 0.05)."
                        )

                    st.markdown(f"**Estatísticas Descritivas por Grupo:**")
                    grouped_stats = (
                        df_anova.groupby(anova_cat_feature)[anova_num_feature]
                        .agg(["count", "mean", "std"])
                        .reset_index()
                    )
                    st.dataframe(grouped_stats)

            except ValueError as ve:
                st.warning(
                    f"Erro ao realizar ANOVA: {ve}. Isso pode ocorrer se os grupos tiverem variância zero ou não houver dados suficientes."
                )
            except Exception as e:
                st.error(
                    f"Ocorreu um erro inesperado ao calcular ANOVA para '{anova_num_feature}' agrupado por '{anova_cat_feature}': {e}"
                )
    else:
        st.info(
            "Selecione uma feature numérica e uma feature categórica para realizar o Teste ANOVA."
        )


def display_general_numerical_stats(df_filtered, cls, numerical_features_to_analyze):
    """Displays general descriptive statistics for numerical features."""
    if not numerical_features_to_analyze:
        st.info(
            "Nenhuma feature numérica (incluindo notas gerais e questões socioeconômicas) disponível ou selecionada para análise estatística detalhada."
        )
    else:
        st.markdown("#### 📜 Estatísticas Gerais para Features Numéricas na Categoria")
        if not df_filtered.empty:
            cols_in_df_filtered = [
                col
                for col in numerical_features_to_analyze
                if col in df_filtered.columns
            ]
            if cols_in_df_filtered:
                try:
                    stats_df = df_filtered[cols_in_df_filtered].describe().transpose()
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
                    st.error(f"Erro ao calcular estatísticas gerais para '{cls}': {e}")
            else:
                st.warning(
                    f"Nenhuma das features numéricas selecionadas para estatísticas está presente nos dados filtrados para a categoria '{cls}'."
                )
        else:
            st.warning(
                f"Sem dados em `df_filtered` para a categoria '{cls}' para calcular estatísticas gerais."
            )


def display_grouped_categorical_counts(df_filtered, cls, grouping_cat_feature):
    """Displays counts for a grouped categorical feature."""
    if grouping_cat_feature not in df_filtered.columns:
        st.warning(
            f"Feature de agrupamento '{grouping_cat_feature}' não encontrada nos dados filtrados para '{cls}'."
        )
        return

    st.markdown(
        f"##### Contagem de Valores para **'{grouping_cat_feature}'** na Categoria **'{cls}'**"
    )
    if not df_filtered[grouping_cat_feature].empty:
        try:
            counts_df = df_filtered[grouping_cat_feature].value_counts().reset_index()
            counts_df.columns = [grouping_cat_feature, "Contagem"]
            st.dataframe(counts_df)
        except Exception as e:
            st.error(f"Erro ao calcular contagens para '{grouping_cat_feature}': {e}")
    else:
        st.info(
            f"Não há dados para contar em '{grouping_cat_feature}' para a categoria '{cls}'."
        )


def display_grouped_numerical_stats(
    df_filtered, cls, grouping_cat_feature, numerical_features_to_analyze
):
    """Displays mean/median statistics for numerical features grouped by a categorical feature."""
    if grouping_cat_feature not in df_filtered.columns:
        st.warning(
            f"Feature de agrupamento '{grouping_cat_feature}' não encontrada nos dados filtrados para '{cls}'."
        )
        return

    for num_feat_to_stat in numerical_features_to_analyze:
        if num_feat_to_stat in df_filtered.columns:
            st.markdown(
                f"##### Estatísticas para **'{num_feat_to_stat}'** agrupado por **'{grouping_cat_feature}'**"
            )
            try:
                if pd.api.types.is_numeric_dtype(df_filtered[num_feat_to_stat]):
                    if df_filtered[grouping_cat_feature].notna().any():
                        grouped_stats = (
                            df_filtered.groupby(grouping_cat_feature, observed=True)[
                                num_feat_to_stat
                            ]
                            .agg(["count", "mean", "median", "min", "max", "std"])
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
