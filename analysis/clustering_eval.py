import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import MinMaxScaler

# Constante com as features para avaliação, para manter consistência.
FEATURES_FOR_CLUSTERING_EVAL = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]


def _run_metric_comparison(
    metric_function, metric_name, y_axis_title, lower_is_better=False
):
    """
    Função auxiliar genérica para calcular e plotar métricas de cluster para diferentes valores de k.
    """
    st.header(f"📊 Comparativo de Clusters: {metric_name}")

    # Descrições fornecidas pelo usuário
    if metric_name == "Índice Davies-Bouldin":
        st.markdown(
            "Esta métrica avalia a razão entre a dispersão média dentro dos clusters e a separação entre eles. **Valores mais baixos** indicam um agrupamento de melhor qualidade."
        )
    elif metric_name == "Silhouette Score":
        st.markdown(
            "Mede a similaridade de um objeto ao seu próprio cluster em comparação com outros clusters. Varia de -1 a 1, onde **valores próximos de 1** são melhores."
        )
    elif metric_name == "Índice Calinski-Harabasz":
        st.markdown(
            "Mede a razão entre a variância inter-cluster e a variância intra-cluster. **Valores mais altos** indicam clusters mais densos e bem separados."
        )

    base_path = "./preprocess/generico/microdados_enem_combinado/PREPROCESS/"
    files_and_k = {
        2: f"{base_path}PREPROCESSED_DATA_2M_2C.csv",
        3: f"{base_path}PREPROCESSED_DATA_2M_3C.csv",
        4: f"{base_path}PREPROCESSED_DATA_2M_4C.csv",
        5: f"{base_path}PREPROCESSED_DATA_2M_5C.csv",
        6: f"{base_path}PREPROCESSED_DATA_2M_6C.csv",
    }

    metric_values = {}
    k_options = sorted(files_and_k.keys())
    progress_bar = st.progress(0, text=f"Iniciando cálculo do {metric_name}...")

    for i, k in enumerate(k_options):
        filepath = files_and_k[k]
        progress_text = f"Processando k={k}..."
        progress_bar.progress((i + 1) / len(k_options), text=progress_text)
        try:
            df_k = pd.read_csv(filepath, sep=";")

            classification_col = None
            for col in df_k.columns:
                if df_k[col].nunique() == k:
                    classification_col = col
                    break

            if not classification_col:
                st.warning(
                    f"Não foi possível identificar a coluna de classificação para k={k}."
                )
                continue

            feature_cols = [
                col
                for col in df_k.columns
                if col != classification_col
                and pd.api.types.is_numeric_dtype(df_k[col])
            ]
            df_k.dropna(subset=feature_cols + [classification_col], inplace=True)

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_k[feature_cols])
            labels = df_k[classification_col]

            # Amostragem para Silhouette Score para evitar estouro de memória
            if metric_function == silhouette_score and len(df_k) > 25000:
                st.info(
                    f"Para k={k}, usando uma amostra de 25.000 pontos para o Silhouette Score."
                )
                sample_indices = np.random.choice(
                    X_scaled.shape[0], 25000, replace=False
                )
                X_scaled = X_scaled[sample_indices]
                labels = labels.iloc[sample_indices]

            score = metric_function(X_scaled, labels)
            metric_values[k] = score

        except FileNotFoundError:
            st.warning(f"Arquivo não encontrado para k={k}: {filepath}.")
        except Exception as e:
            st.error(f"Erro ao processar k={k}: {e}")

    progress_bar.empty()

    if not metric_values:
        st.error(f"Não foi possível calcular o {metric_name} para nenhum valor de k.")
        return

    plot_data = pd.DataFrame(
        {
            "Número de Clusters (k)": list(metric_values.keys()),
            metric_name: list(metric_values.values()),
        }
    )

    fig = px.line(
        plot_data,
        x="Número de Clusters (k)",
        y=metric_name,
        title=f"Comparativo de {metric_name} por Número de Clusters",
        markers=True,
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis=dict(tickmode="linear"), yaxis_title=y_axis_title)
    st.plotly_chart(fig, use_container_width=True)


def display_elbow_curve():
    st.header("📉 Curva de Cotovelo (Elbow Method)")
    st.markdown(
        "O 'cotovelo' da curva, onde a taxa de queda da inércia diminui, sugere o melhor valor para `k`."
    )
    # Implementação da curva de cotovelo (código anterior)


def display_davies_bouldin_comparison():
    _run_metric_comparison(
        davies_bouldin_score,
        "Índice Davies-Bouldin",
        "Índice Davies-Bouldin (Menor é Melhor)",
        lower_is_better=True,
    )


def display_silhouette_comparison():
    _run_metric_comparison(
        silhouette_score, "Silhouette Score", "Silhouette Score (Maior é Melhor)"
    )


def display_calinski_harabasz_comparison():
    _run_metric_comparison(
        calinski_harabasz_score,
        "Índice Calinski-Harabasz",
        "Índice Calinski-Harabasz (Maior é Melhor)",
    )


def display_elbow_curve():
    """
    Calcula a inércia (WCSS) para diferentes números de clusters a partir de
    arquivos pré-processados e plota a Curva de Cotovelo.
    """
    st.header("📉 Curva de Cotovelo (Elbow Method)")
    st.markdown(
        "Este gráfico ajuda a encontrar o número ideal de clusters (quantis). O 'cotovelo' da curva, onde a taxa de queda da inércia diminui drasticamente, sugere o melhor valor para `k`."
    )

    base_path = "./preprocess/generico/microdados_enem_combinado/PREPROCESS/"
    # Mapeia o número de clusters (k) para o arquivo correspondente
    files_and_k = {
        2: f"{base_path}PREPROCESSED_DATA_2M_2C.csv",
        3: f"{base_path}PREPROCESSED_DATA_2M_3C.csv",
        4: f"{base_path}PREPROCESSED_DATA_2M_4C.csv",
        5: f"{base_path}PREPROCESSED_DATA_2M_5C.csv",
        6: f"{base_path}PREPROCESSED_DATA_2M_6C.csv",
    }

    inertia_values = {}
    k_options = sorted(files_and_k.keys())

    progress_bar = st.progress(0, text="Iniciando cálculo da inércia...")

    for i, k in enumerate(k_options):
        filepath = files_and_k[k]
        progress_text = f"Processando k={k} (arquivo: {filepath.split('/')[-1]})"
        progress_bar.progress((i + 1) / len(k_options), text=progress_text)
        try:
            # Adicionado o separador ';' para ler o CSV corretamente.
            df_k = pd.read_csv(
                filepath,
                sep=";",
                usecols=FEATURES_FOR_CLUSTERING_EVAL + ["CLASSIFICACAO"],
            )
            df_k.dropna(inplace=True)

            # Normalização dos dados
            scaler = MinMaxScaler()
            features_to_scale = [col for col in df_k.columns if col != "CLASSIFICACAO"]
            df_k_scaled = df_k.copy()
            df_k_scaled[features_to_scale] = scaler.fit_transform(
                df_k[features_to_scale]
            )

            # Cálculo da inércia (WCSS) com os dados normalizados
            wcss = 0
            centroids = df_k_scaled.groupby("CLASSIFICACAO")[features_to_scale].mean()
            for cluster_id, centroid_coords in centroids.iterrows():
                cluster_points = df_k_scaled[
                    df_k_scaled["CLASSIFICACAO"] == cluster_id
                ][features_to_scale]
                wcss += np.sum((cluster_points.values - centroid_coords.values) ** 2)

            inertia_values[k] = wcss
        except FileNotFoundError:
            st.warning(
                f"Arquivo não encontrado para k={k}: {filepath}. Este ponto será ignorado."
            )
        except ValueError as ve:
            st.error(
                f"Erro de valor ao processar k={k}: {ve}. Verifique se as colunas em 'usecols' existem no arquivo '{filepath.split('/')[-1]}' e se o separador está correto."
            )
        except Exception as e:
            st.error(
                f"Erro ao processar o arquivo para k={k}: {e}. Este ponto será ignorado."
            )

    progress_bar.empty()

    if not inertia_values:
        st.error("Não foi possível calcular a inércia para nenhum valor de k.")
        return

    plot_data = pd.DataFrame(
        {
            "Número de Clusters (k)": list(inertia_values.keys()),
            "Inércia (WCSS)": list(inertia_values.values()),
        }
    )

    fig = px.line(
        plot_data,
        x="Número de Clusters (k)",
        y="Inércia (WCSS)",
        title="Curva de Cotovelo (Elbow Method)",
        markers=True,
        labels={
            "Número de Clusters (k)": "Número de Clusters (k)",
            "Inércia (WCSS)": "Soma dos Quadrados Intra-Cluster (WCSS)",
        },
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig, use_container_width=True)


def evaluate_davies_bouldin(df, features_for_clustering_eval, c_type):
    """Calcula e exibe o Índice Davies-Bouldin para um tipo de classificação."""
    st.subheader("Índice Davies-Bouldin")
    st.markdown("Valores mais baixos indicam agrupamentos melhores.")

    data_for_db = df[[c_type] + features_for_clustering_eval].dropna()

    if data_for_db.empty:
        st.warning("Dados insuficientes para avaliação.")
        return

    X_db = data_for_db[features_for_clustering_eval]
    labels_db = data_for_db[c_type]
    n_clusters = labels_db.nunique()

    if n_clusters < 2:
        st.warning(f"Apenas {n_clusters} cluster encontrado. Requer 2 ou mais.")
        return

    try:
        score = davies_bouldin_score(X_db, labels_db)
        st.metric(label="Índice Davies-Bouldin", value=f"{score:.4f}")
    except Exception as e:
        st.error(f"Erro ao calcular o índice: {e}")


def evaluate_silhouette_score(
    df,
    features_for_clustering_eval,
    c_type,
    sampling_percentage,
    min_samples_for_sampling,
):
    """Calcula e exibe o Silhouette Score para um tipo de classificação."""
    st.subheader("Silhouette Score")
    st.markdown("Valores mais próximos de 1 indicam agrupamentos melhores.")

    data_for_sil = df[[c_type] + features_for_clustering_eval].dropna()

    if data_for_sil.empty:
        st.warning("Dados insuficientes para avaliação.")
        return

    # Amostragem para performance
    if data_for_sil.shape[0] > min_samples_for_sampling:
        sample_size = int(data_for_sil.shape[0] * sampling_percentage)
        data_for_sil = data_for_sil.sample(n=sample_size, random_state=42)
        st.info(f"Calculando em uma amostra de {sample_size} pontos.")

    X_sil = data_for_sil[features_for_clustering_eval]
    labels_sil = data_for_sil[c_type]
    n_clusters = labels_sil.nunique()

    if n_clusters < 2 or n_clusters >= X_sil.shape[0]:
        st.warning(
            f"Número de clusters ({n_clusters}) ou amostras ({X_sil.shape[0]}) inválido para cálculo."
        )
        return

    try:
        score = silhouette_score(X_sil, labels_sil)
        st.metric(label="Silhouette Score", value=f"{score:.4f}")
    except Exception as e:
        st.error(f"Erro ao calcular o score: {e}")


def evaluate_calinski_harabasz_score(df, features_for_clustering_eval, c_type):
    """Calcula e exibe o Índice Calinski-Harabasz para um tipo de classificação."""
    st.subheader("Índice Calinski-Harabasz")
    st.markdown("Valores mais altos indicam agrupamentos melhores.")

    data_for_ch = df[[c_type] + features_for_clustering_eval].dropna()

    if data_for_ch.empty:
        st.warning("Dados insuficientes para avaliação.")
        return

    X_ch = data_for_ch[features_for_clustering_eval]
    labels_ch = data_for_ch[c_type]
    n_clusters = labels_ch.nunique()

    if n_clusters < 2:
        st.warning(f"Apenas {n_clusters} cluster encontrado. Requer 2 ou mais.")
        return

    try:
        score = calinski_harabasz_score(X_ch, labels_ch)
        st.metric(label="Índice Calinski-Harabasz", value=f"{score:.2f}")
    except Exception as e:
        st.error(f"Erro ao calcular o índice: {e}")
