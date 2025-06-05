import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
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

# Lista de features categóricas pré-definidas para a ANOVA comparativa
# Certifique-se de que estas colunas existem no seu DataFrame e são categóricas.
ANOVA_PREDEFINED_CAT_FEATURES = [
    "TP_COR_RACA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_SEXO",
    "TP_LINGUA",
    "TP_ESCOLA",
    # Adicione outras features categóricas que você queira comparar automaticamente aqui
]


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

    # --- NOVO: Multiselect para seleção de visualizações ---
    visualization_options = [
        "Comparação Categórica entre Categorias Selecionadas",
        "Plots: Features Categóricas",
        "Teste Qui-Quadrado",
        "Teste ANOVA (Seleção Manual)",  # Renomeado para clareza
        "Análise ANOVA Comparativa (Pré-definida)",  # Nova opção
        "Plots: Features Numéricas",
        "Estatísticas Gerais para Features Numéricas na Categoria",
        "Estatísticas Agrupadas por Feature Categórica (contagens)",
        "Estatísticas Agrupadas por Feature Categórica (médias/medianas)",
    ]

    if "selected_visualizations" not in st.session_state:
        st.session_state.selected_visualizations = [
            "Estatísticas Gerais para Features Numéricas na Categoria"
        ]

    selected_visualizations = st.sidebar.multiselect(
        "Selecione as visualizações a exibir:",
        options=visualization_options,
        default=st.session_state.selected_visualizations,
        key="visualization_selector",
    )
    # --- FIM NOVO: Multiselect para seleção de visualizações ---

    current_filters = {
        "classification_types": classification_types,
        "selected_classifications": selected_classes,
        "selected_categorical_features": selected_cat_features,
        "selected_numerical_features": selected_num_features,
    }

    for c_type in classification_types:
        st.subheader(f"🔍 Classificação: **{c_type.replace('_', ' ')}**")

        if (
            "Comparação Categórica entre Categorias Selecionadas"
            in selected_visualizations
        ):
            if len(selected_classes) > 1 and selected_cat_features:
                st.subheader("📊 Comparação Categórica entre Categorias Selecionadas")
                for feature in selected_cat_features:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    df_filtered_compare = df[
                        df[c_type].astype(str).isin(selected_classes)
                    ]

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
                            val_series = hue_data[hue_data[feature] == cat_val][
                                "Proporção"
                            ]
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
                    ax.legend(
                        title="Categoria", bbox_to_anchor=(1.05, 1), loc="upper left"
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
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
                        st.markdown(
                            f"##### Teste Qui-Quadrado para '{feature}' vs. '{c_type.replace('_', ' ')}'"
                        )

                        contingency_table_full = pd.crosstab(df[feature], df[c_type])

                        if "Todas" not in selected_classes_input and selected_classes:
                            if c_type in contingency_table_full.columns:
                                contingency_table = contingency_table_full[
                                    selected_classes
                                ]
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
                                chi2, p_value, dof, expected = chi2_contingency(
                                    contingency_table
                                )
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
                                st.markdown(
                                    f"**Tabela de Contingência (Contagens Observadas):**"
                                )
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
                elif "Teste Qui-Quadrado" in selected_visualizations:
                    st.info(
                        "Selecione features categóricas para realizar o teste Qui-Quadrado."
                    )

            # --- Teste ANOVA (Seleção Manual) ---
            if "Teste ANOVA (Seleção Manual)" in selected_visualizations:
                st.markdown("#### 🧪 Teste ANOVA (Seleção Manual)")
                st.write(
                    "Compare as médias de uma feature numérica entre grupos definidos por uma feature categórica."
                )

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
                        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
                    ],
                    key=f"anova_manual_num_feature_{c_type}_{cls}",
                )

                anova_cat_feature = st.selectbox(
                    f"Selecione a Feature Categórica (independente/agrupamento) para ANOVA na categoria '{cls}':",
                    options=["Nenhuma"] + [f for f in cat_features if f in df.columns],
                    key=f"anova_manual_cat_feature_{c_type}_{cls}",
                )

                if anova_num_feature != "Nenhuma" and anova_cat_feature != "Nenhuma":
                    st.markdown(
                        f"##### ANOVA para '{anova_num_feature}' agrupado por '{anova_cat_feature}'"
                    )

                    df_anova = df[df[c_type].astype(str) == str(cls)][
                        [anova_num_feature, anova_cat_feature]
                    ].dropna()

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
                                    df_anova.groupby(anova_cat_feature)[
                                        anova_num_feature
                                    ]
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
                elif "Teste ANOVA (Seleção Manual)" in selected_visualizations:
                    st.info(
                        "Selecione uma feature numérica e uma feature categórica para realizar o Teste ANOVA."
                    )
            # --- FIM Teste ANOVA (Seleção Manual) ---

            # --- NOVO: Análise ANOVA Comparativa (Pré-definida) ---
            if "Análise ANOVA Comparativa (Pré-definida)" in selected_visualizations:
                st.markdown(
                    "#### 📊 Análise ANOVA Comparativa (Variáveis Pré-definidas)"
                )
                st.write(
                    "Resultados da ANOVA para a feature numérica selecionada, comparada com um conjunto fixo de variáveis classificatórias."
                )

                # Options for numerical features (dependent variable)
                anova_comparative_num_feature_options = sorted(
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

                selected_anova_num_feature = st.selectbox(
                    f"Selecione a Feature Numérica (dependente) para Análise ANOVA Comparativa na categoria '{cls}':",
                    options=["Nenhuma"]
                    + [
                        f
                        for f in anova_comparative_num_feature_options
                        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
                    ],
                    key=f"anova_comparative_num_feature_{c_type}_{cls}",
                )

                if selected_anova_num_feature != "Nenhuma":
                    results_list = []

                    for cat_feat in ANOVA_PREDEFINED_CAT_FEATURES:
                        if cat_feat not in df.columns:
                            results_list.append(
                                {
                                    "Variável Classificatória": cat_feat,
                                    "Valor F": "N/A",
                                    "p-valor": "N/A",
                                    "Associação Significativa (p < 0.05)": "Coluna não encontrada",
                                    "Observações": "N/A",
                                }
                            )
                            continue

                        # Use o df_filtered para garantir que a ANOVA é para a categoria de nota atual
                        df_anova_comp = df_filtered[
                            [selected_anova_num_feature, cat_feat]
                        ].dropna()

                        if df_anova_comp.empty:
                            results_list.append(
                                {
                                    "Variável Classificatória": cat_feat,
                                    "Valor F": "N/A",
                                    "p-valor": "N/A",
                                    "Associação Significativa (p < 0.05)": "Dados Insuficientes",
                                    "Observações": "Não há dados após filtrar NaNs para esta combinação.",
                                }
                            )
                            continue

                        try:
                            groups = [
                                df_anova_comp[selected_anova_num_feature][
                                    df_anova_comp[cat_feat] == category
                                ].dropna()
                                for category in df_anova_comp[cat_feat].unique()
                            ]

                            valid_groups = [
                                g for g in groups if g.count() > 1
                            ]  # Groups must have at least 2 observations

                            if len(valid_groups) < 2:
                                results_list.append(
                                    {
                                        "Variável Classificatória": cat_feat,
                                        "Valor F": "N/A",
                                        "p-valor": "N/A",
                                        "Associação Significativa (p < 0.05)": "Grupos Insuficientes",
                                        "Observações": "Menos de 2 grupos com dados suficientes para ANOVA.",
                                    }
                                )
                            else:
                                f_statistic, p_value = f_oneway(*valid_groups)

                                is_significant = "Sim" if p_value < 0.05 else "Não"
                                obs = f"Média(s): {df_anova_comp.groupby(cat_feat)[selected_anova_num_feature].mean().round(2).to_dict()}"

                                results_list.append(
                                    {
                                        "Variável Classificatória": cat_feat,
                                        "Valor F": f"{f_statistic:.2f}",
                                        "p-valor": f"{p_value:.5f}",
                                        "Associação Significativa (p < 0.05)": is_significant,
                                        "Observações": obs,
                                    }
                                )

                        except ValueError as ve:
                            results_list.append(
                                {
                                    "Variável Classificatória": cat_feat,
                                    "Valor F": "Erro",
                                    "p-valor": "Erro",
                                    "Associação Significativa (p < 0.05)": "Erro",
                                    "Observações": f"Erro ao calcular: {ve}",
                                }
                            )
                        except Exception as e:
                            results_list.append(
                                {
                                    "Variável Classificatória": cat_feat,
                                    "Valor F": "Erro",
                                    "p-valor": "Erro",
                                    "Associação Significativa (p < 0.05)": "Erro",
                                    "Observações": f"Erro inesperado: {e}",
                                }
                            )

                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        st.dataframe(results_df.set_index("Variável Classificatória"))
                    else:
                        st.info(
                            "Nenhum resultado de ANOVA gerado para as variáveis pré-definidas."
                        )
                else:
                    st.info(
                        "Selecione uma Feature Numérica para iniciar a Análise ANOVA Comparativa."
                    )

            # --- FIM NOVO: Análise ANOVA Comparativa (Pré-definida) ---

            if "Plots: Features Numéricas" in selected_visualizations:
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
                elif "Plots: Features Numéricas" in selected_visualizations:
                    st.info("Selecione features numéricas para ver os plots numéricos.")

            # --- INÍCIO DA SEÇÃO DE ESTATÍSTICAS DETALHADAS ---
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
                                    df_filtered[cols_in_df_filtered]
                                    .describe()
                                    .transpose()
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
                                    [
                                        col
                                        for col in cols_to_show
                                        if col in stats_df.columns
                                    ]
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
                        if grouping_cat_feature not in df_filtered.columns:
                            st.warning(
                                f"Feature de agrupamento '{grouping_cat_feature}' não encontrada nos dados filtrados para '{cls}'."
                            )
                        else:
                            if (
                                "Estatísticas Agrupadas por Feature Categórica (contagens)"
                                in selected_visualizations
                            ):
                                st.markdown(
                                    f"##### Contagem de Valores para **'{grouping_cat_feature}'** na Categoria **'{cls}'**"
                                )
                                if not df_filtered[grouping_cat_feature].empty:
                                    try:
                                        counts_df = (
                                            df_filtered[grouping_cat_feature]
                                            .value_counts()
                                            .reset_index()
                                        )
                                        counts_df.columns = [
                                            grouping_cat_feature,
                                            "Contagem",
                                        ]
                                        st.dataframe(counts_df)
                                    except Exception as e:
                                        st.error(
                                            f"Erro ao calcular contagens para '{grouping_cat_feature}': {e}"
                                        )
                                else:
                                    st.info(
                                        f"Não há dados para contar em '{grouping_cat_feature}' para a categoria '{cls}'."
                                    )

                            if (
                                "Estatísticas Agrupadas por Feature Categórica (médias/medianas)"
                                in selected_visualizations
                            ):
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
                    elif (
                        "Estatísticas Agrupadas por Feature Categórica (contagens)"
                        in selected_visualizations
                        or "Estatísticas Agrupadas por Feature Categórica (médias/medianas)"
                        in selected_visualizations
                    ):
                        st.info(
                            "Selecione uma feature categórica para ver as estatísticas agrupadas."
                        )
            # --- END OF DETAILED STATISTICS SECTION ---

    st.info(
        "Use a barra lateral para refinar a análise e selecionar as visualizações desejadas."
    )


if __name__ == "__main__":
    main()
