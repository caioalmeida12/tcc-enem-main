import streamlit as st
import pandas as pd
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)


def evaluate_davies_bouldin(df, features_for_clustering_eval):
    """Calculates and displays Davies-Bouldin Index."""
    st.header("✨ Avaliação de Agrupamento: Índice Davies-Bouldin")
    st.markdown(
        """
        O Índice Davies-Bouldin avalia a qualidade dos agrupamentos gerados pelo algoritmo Birch.
        **Valores mais baixos indicam agrupamentos melhores** (mais coesos internamente e mais separados entre si).
        """
    )

    results_db = []
    for cls_type_db in [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
    ]:
        if cls_type_db not in df.columns:
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "N/A",
                    "Observações": f"Coluna '{cls_type_db}' não encontrada.",
                }
            )
            continue

        data_for_db = df[
            [cls_type_db] + [f for f in features_for_clustering_eval if f in df.columns]
        ].dropna()

        if data_for_db.empty:
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "N/A",
                    "Observações": "Dados insuficientes para avaliação.",
                }
            )
            continue

        X_db = data_for_db[
            [f for f in features_for_clustering_eval if f in data_for_db.columns]
        ]
        labels_db = data_for_db[cls_type_db]

        n_clusters = len(labels_db.unique())
        if n_clusters < 2:
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "N/A",
                    "Observações": f"Apenas {n_clusters} cluster(s) encontrado(s). Requer 2 ou mais clusters.",
                }
            )
            continue

        if X_db.shape[0] < (n_clusters + 1):
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "N/A",
                    "Observações": f"Número de amostras ({X_db.shape[0]}) insuficiente para o número de clusters ({n_clusters}).",
                }
            )
            continue

        try:
            score = davies_bouldin_score(X_db, labels_db)
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": f"{score:.4f}",
                    "Observações": "Avaliação concluída.",
                }
            )
        except ValueError as ve:
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "Erro",
                    "Observações": f"Erro de cálculo: {ve}. Verifique se há variância zero em algum cluster ou número de amostras muito pequeno.",
                }
            )
        except Exception as e:
            results_db.append(
                {
                    "Tipo de Classificação": cls_type_db,
                    "Índice Davies-Bouldin": "Erro",
                    "Observações": f"Erro inesperado: {e}",
                }
            )

    if results_db:
        st.dataframe(pd.DataFrame(results_db))
    else:
        st.info("Nenhum resultado de Davies-Bouldin gerado.")


def evaluate_silhouette_score(
    df, features_for_clustering_eval, sampling_percentage, min_samples_for_sampling
):
    """Calculates and displays Silhouette Score."""
    st.header("✨ Avaliação de Agrupamento: Silhouette Score")
    st.markdown(
        """
        O Silhouette Score avalia a qualidade dos agrupamentos gerados pelo algoritmo Birch.
        **Valores mais próximos de 1 indicam agrupamentos melhores** (bem definidos e separados).
        Valores próximos de 0 indicam clusters sobrepostos. Valores negativos indicam má atribuição.
        """
    )

    results_silhouette = []
    for cls_type_sil in [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
    ]:
        if cls_type_sil not in df.columns:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "N/A",
                    "Observações": f"Coluna '{cls_type_sil}' não encontrada.",
                }
            )
            continue

        data_for_sil = df[
            [cls_type_sil]
            + [f for f in features_for_clustering_eval if f in df.columns]
        ].dropna()

        if data_for_sil.empty:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "N/A",
                    "Observações": "Dados insuficientes para avaliação.",
                }
            )
            continue

        current_num_samples = data_for_sil.shape[0]
        sampling_applied = False
        if current_num_samples > min_samples_for_sampling:
            sample_size = max(2, int(current_num_samples * sampling_percentage))
            if sample_size < current_num_samples:
                data_for_sil_sampled = data_for_sil.sample(
                    n=sample_size, random_state=42
                )
                X_sil = data_for_sil_sampled[
                    [
                        f
                        for f in features_for_clustering_eval
                        if f in data_for_sil_sampled.columns
                    ]
                ]
                labels_sil = data_for_sil_sampled[cls_type_sil]
                sampling_applied = True
                st.info(
                    f"Calculando Silhouette Score em uma amostra de {sample_size} de {current_num_samples} pontos para '{cls_type_sil}'."
                )
            else:
                X_sil = data_for_sil[
                    [
                        f
                        for f in features_for_clustering_eval
                        if f in data_for_sil.columns
                    ]
                ]
                labels_sil = data_for_sil[cls_type_sil]
        else:
            X_sil = data_for_sil[
                [f for f in features_for_clustering_eval if f in data_for_sil.columns]
            ]
            labels_sil = data_for_sil[cls_type_sil]

        n_clusters_sil = len(labels_sil.unique())
        if n_clusters_sil < 2 or n_clusters_sil >= X_sil.shape[0]:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "N/A",
                    "Observações": f"Número de clusters ({n_clusters_sil}) ou amostras ({X_sil.shape[0]}) insuficiente para o cálculo. Requer 2 <= clusters < amostras.",
                }
            )
            continue

        if X_sil.shape[0] <= 1:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "N/A",
                    "Observações": "Apenas uma amostra ou menos após a filtragem.",
                }
            )
            continue

        try:
            score_sil = silhouette_score(X_sil, labels_sil)
            obs_text = (
                "Avaliação concluída (amostra utilizada)."
                if sampling_applied
                else "Avaliação concluída."
            )
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": f"{score_sil:.4f}",
                    "Observações": obs_text,
                }
            )
        except ValueError as ve:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "Erro",
                    "Observações": f"Erro de cálculo: {ve}. Pode ser devido a cluster com variância zero ou poucas amostras.",
                }
            )
        except Exception as e:
            results_silhouette.append(
                {
                    "Tipo de Classificação": cls_type_sil,
                    "Silhouette Score": "Erro",
                    "Observações": f"Erro inesperado: {e}",
                }
            )

    if results_silhouette:
        st.dataframe(pd.DataFrame(results_silhouette))
    else:
        st.info("Nenhum resultado de Silhouette Score gerado.")


def evaluate_calinski_harabasz_score(df, features_for_clustering_eval):
    """Calculates and displays Calinski-Harabasz Score."""
    st.header("✨ Avaliação de Agrupamento: Índice Calinski-Harabasz")
    st.markdown(
        """
        O Índice Calinski-Harabasz (também conhecido como Variance Ratio Criterion)
        avalia a qualidade dos agrupamentos. **Valores mais altos indicam agrupamentos melhores**
        (densos e bem separados).
        """
    )

    results_ch = []
    for cls_type_ch in [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
    ]:
        if cls_type_ch not in df.columns:
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": "N/A",
                    "Observações": f"Coluna '{cls_type_ch}' não encontrada.",
                }
            )
            continue

        data_for_ch = df[
            [cls_type_ch] + [f for f in features_for_clustering_eval if f in df.columns]
        ].dropna()

        if data_for_ch.empty:
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": "N/A",
                    "Observações": "Dados insuficientes para avaliação.",
                }
            )
            continue

        X_ch = data_for_ch[
            [f for f in features_for_clustering_eval if f in data_for_ch.columns]
        ]
        labels_ch = data_for_ch[cls_type_ch]

        n_clusters_ch = len(labels_ch.unique())
        if n_clusters_ch < 2 or n_clusters_ch >= X_ch.shape[0]:
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": "N/A",
                    "Observações": f"Número de clusters ({n_clusters_ch}) ou amostras ({X_ch.shape[0]}) insuficiente para o cálculo. Requer 2 <= clusters < amostras.",
                }
            )
            continue

        try:
            score_ch = calinski_harabasz_score(X_ch, labels_ch)
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": f"{score_ch:.4f}",
                    "Observações": "Avaliação concluída.",
                }
            )
        except ValueError as ve:
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": "Erro",
                    "Observações": f"Erro de cálculo: {ve}. Pode ser devido a clusters com variância zero ou poucas amostras. Tente garantir que cada cluster tenha pelo menos 2 amostras e que as colunas numéricas tenham variância.",
                }
            )
        except Exception as e:
            results_ch.append(
                {
                    "Tipo de Classificação": cls_type_ch,
                    "Calinski-Harabasz Score": "Erro",
                    "Observações": f"Erro inesperado: {e}",
                }
            )

    if results_ch:
        st.dataframe(pd.DataFrame(results_ch))
    else:
        st.info("Nenhum resultado de Calinski-Harabasz Score gerado.")
