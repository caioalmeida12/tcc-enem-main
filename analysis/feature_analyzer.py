import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
import plotly.express as px


COLUNAS_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
]


def display_score_histogram(df, classification_type, selected_classes):
    """Exibe um histograma de notas empilhado por classificação."""
    st.markdown("#### 📊 Histograma de Notas por Classificação")

    score_options = COLUNAS_NOTAS + ["NU_NOTA_REDACAO", "NOTA_GERAL"]
    if "NOTA_GERAL" in df.columns:
        score_options.append("NOTA_GERAL")

    selected_score = st.selectbox(
        "Selecione a nota para o histograma:",
        options=score_options,
        key=f"hist_score_selector_{classification_type}",
    )

    df_plot = df[
        df[classification_type].astype(str).isin(map(str, selected_classes))
    ].copy()
    df_plot[selected_score] = pd.to_numeric(df_plot[selected_score], errors="coerce")
    df_plot.dropna(subset=[selected_score, classification_type], inplace=True)

    if df_plot.empty:
        st.warning(
            f"Não há dados disponíveis para a nota '{selected_score}' com os filtros atuais."
        )
        return

    min_score, max_score = df_plot[selected_score].min(), df_plot[selected_score].max()
    bins = np.arange(0, max_score + 10, 10)
    labels = [f"{int(i)}-{int(i+10)-1}" for i in bins[:-1]]

    df_plot["faixa_nota"] = pd.cut(
        df_plot[selected_score],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    df_plot[classification_type] = df_plot[classification_type].astype("category")

    fig = px.histogram(
        df_plot,
        x="faixa_nota",
        color=classification_type,
        barmode="stack",
        title=f"Distribuição de '{selected_score}' por '{classification_type}'",
        labels={
            "faixa_nota": f"Faixa de Nota ({selected_score})",
            "count": "Quantidade de Alunos",
            classification_type: "Classificação",
        },
        category_orders={"faixa_nota": labels},
    )

    fig.update_layout(
        xaxis_title=f"Faixa de Nota ({selected_score})",
        yaxis_title="Quantidade de Alunos",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_chi_squared_test(
    df, feature, c_type, selected_classes_input, selected_classes
):
    """Exibe os resultados do teste Qui-Quadrado para uma dada feature categórica."""
    st.markdown(
        f"##### Teste Qui-Quadrado para '{feature}' vs. '{c_type.replace('_', ' ')}'"
    )

    # Cria a tabela de contingência com todos os dados
    contingency_table_full = pd.crosstab(df[feature], df[c_type])

    # Filtra a tabela se o usuário não selecionou "Todas" as classes
    if "Todas" not in selected_classes_input and selected_classes:
        # Pega apenas as colunas que existem na tabela completa
        valid_classes = [
            c for c in selected_classes if c in contingency_table_full.columns
        ]
        if valid_classes:
            contingency_table = contingency_table_full[valid_classes]
        else:
            st.warning(
                f"Nenhuma das classes selecionadas {selected_classes} foi encontrada nos dados para o teste de '{feature}'."
            )
            return
    else:
        contingency_table = contingency_table_full

    # Validações para realizar o teste
    if (
        contingency_table.empty
        or contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        st.info(
            f"Dados insuficientes para o teste Qui-Quadrado de '{feature}'. São necessárias pelo menos 2 linhas e 2 colunas na tabela de contingência."
        )
        return

    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        st.write(f"**Valor Qui-Quadrado:** {chi2:.2f}")
        st.write(f"**p-valor:** {p_value:.5f}")
        st.write(f"**Graus de liberdade:** {dof}")

        if p_value < 0.05:
            st.success(
                f"Há uma associação estatisticamente significativa entre '{feature}' e a classificação (p < 0.05)."
            )
        else:
            st.info(
                f"Não há evidência de associação estatisticamente significativa entre '{feature}' e a classificação (p >= 0.05)."
            )
        st.markdown(f"**Tabela de Contingência (Observado):**")
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
            f"Não foi possível realizar o teste Qui-Quadrado para '{feature}'. Motivo: {ve}. Isso pode ocorrer se houver categorias com zero observações."
        )


def display_anova_test(
    df_filtered, c_type, cls, cat_features, num_features, COLUNAS_NOTAS
):
    """Exibe os resultados do teste ANOVA baseado na seleção manual."""
    anova_num_feature_options = sorted(
        list(set(num_features + COLUNAS_NOTAS + ["NOTA_GERAL"]))
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
        f"Selecione a Feature Categórica (independente) para ANOVA na categoria '{cls}':",
        options=["Nenhuma"] + [f for f in cat_features if f in df_filtered.columns],
        key=f"anova_manual_cat_feature_{c_type}_{cls}",
    )

    if anova_num_feature != "Nenhuma" and anova_cat_feature != "Nenhuma":
        st.markdown(
            f"##### ANOVA para '{anova_num_feature}' agrupado por '{anova_cat_feature}'"
        )

        df_anova = df_filtered[[anova_num_feature, anova_cat_feature]].dropna()

        if df_anova.empty or df_anova[anova_cat_feature].nunique() < 2:
            st.warning(
                "Não há dados suficientes ou grupos (mínimo 2) para realizar ANOVA com as seleções atuais."
            )
            return

        try:
            groups = [
                df_anova[anova_num_feature][df_anova[anova_cat_feature] == category]
                for category in df_anova[anova_cat_feature].unique()
            ]

            valid_groups = [g for g in groups if len(g) > 1]

            if len(valid_groups) < 2:
                st.warning(
                    "São necessários pelo menos 2 grupos com mais de uma observação para realizar a ANOVA."
                )
                return

            f_statistic, p_value = f_oneway(*valid_groups)

            st.write(f"**F-Estatística:** {f_statistic:.2f}")
            st.write(f"**p-valor:** {p_value:.5f}")

            if p_value < 0.05:
                st.success(
                    f"Há uma diferença estatisticamente significativa nas médias de '{anova_num_feature}' entre os grupos de '{anova_cat_feature}'."
                )
            else:
                st.info(
                    f"Não há evidência de diferença significativa nas médias de '{anova_num_feature}' entre os grupos de '{anova_cat_feature}'."
                )

            st.markdown(f"**Estatísticas Descritivas por Grupo:**")
            st.dataframe(
                df_anova.groupby(anova_cat_feature)[anova_num_feature]
                .agg(["count", "mean", "std"])
                .reset_index()
            )

        except ValueError as ve:
            st.warning(f"Erro ao realizar ANOVA: {ve}.")


def display_general_numerical_stats(df_filtered, cls, numerical_features_to_analyze):
    """Exibe estatísticas descritivas gerais para features numéricas."""
    st.markdown("#### 📜 Estatísticas Gerais para Features Numéricas")

    cols_in_df = [
        col for col in numerical_features_to_analyze if col in df_filtered.columns
    ]

    if not cols_in_df:
        st.info(
            "Nenhuma feature numérica selecionada está disponível nos dados filtrados."
        )
        return

    try:
        stats_df = df_filtered[cols_in_df].describe().transpose()
        st.dataframe(stats_df)
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas gerais para '{cls}': {e}")


def display_grouped_categorical_counts(df_filtered, cls, grouping_cat_feature):
    """Exibe contagens para uma feature categórica agrupada."""
    st.markdown(
        f"##### Contagem para **'{grouping_cat_feature}'** na Categoria **'{cls}'**"
    )
    try:
        counts_df = df_filtered[grouping_cat_feature].value_counts().reset_index()
        counts_df.columns = [grouping_cat_feature, "Contagem"]
        st.dataframe(counts_df)
    except Exception as e:
        st.error(f"Erro ao calcular contagens para '{grouping_cat_feature}': {e}")


def display_grouped_numerical_stats(
    df_filtered, cls, grouping_cat_feature, numerical_features_to_analyze
):
    """Exibe estatísticas de média/mediana para features numéricas agrupadas por uma feature categórica."""
    for num_feat in numerical_features_to_analyze:
        if num_feat in df_filtered.columns and pd.api.types.is_numeric_dtype(
            df_filtered[num_feat]
        ):
            st.markdown(
                f"##### Estatísticas de '{num_feat}' por '{grouping_cat_feature}'"
            )
            try:
                grouped_stats = (
                    df_filtered.groupby(grouping_cat_feature, observed=True)[num_feat]
                    .agg(["count", "mean", "median", "std"])
                    .reset_index()
                )
                st.dataframe(grouped_stats)
            except Exception as e:
                st.error(f"Erro ao calcular estatísticas para '{num_feat}': {e}")
