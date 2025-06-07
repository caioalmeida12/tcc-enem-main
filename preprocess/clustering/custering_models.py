import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, Birch
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def classificar_com_agrupamento_hierarquico(
    df: pd.DataFrame, n_clusters: int
) -> pd.DataFrame:
    """
    Classifica as notas gerais usando agrupamento hierárquico.
    A quantidade de clusters é definida pelo parâmetro `n_clusters`.
    """
    copia_df = df.copy()

    score_cols = ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]
    df_for_clustering = copia_df.dropna(subset=score_cols).copy()

    if df_for_clustering.empty:
        print("Nenhum dado válido para agrupamento hierárquico.")
        copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = np.nan
        copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = np.nan
        return copia_df

    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df_for_clustering["CLUSTER_LABEL"] = cluster.fit_predict(
        df_for_clustering[score_cols]
    )

    copia_df = copia_df.merge(
        df_for_clustering[["CLUSTER_LABEL"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df.drop(columns=["CLUSTER_LABEL"], inplace=True)

    return copia_df


def classificar_com_dbscan(
    df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5
) -> pd.DataFrame:
    """
    Classifica as notas gerais usando agrupamento DBSCAN.
    Os parâmetros `eps` (distância máxima entre duas amostras para uma ser considerada na vizinhança da outra)
    e `min_samples` (número de amostras em uma vizinhança para um ponto ser considerado um ponto central)
    podem ser ajustados.
    """
    copia_df = df.copy()

    score_cols = ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]

    df_for_clustering = copia_df.dropna(subset=score_cols).copy()

    if df_for_clustering.empty:
        print("Nenhum dado válido para agrupamento DBSCAN.")
        copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = np.nan
        copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = np.nan
        return copia_df

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_for_clustering["CLUSTER_LABEL"] = dbscan.fit_predict(
        df_for_clustering[score_cols]
    )

    copia_df = copia_df.merge(
        df_for_clustering[["CLUSTER_LABEL"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df.drop(columns=["CLUSTER_LABEL"], inplace=True)

    return copia_df


def classificar_com_birch(
    df: pd.DataFrame,
    n_clusters: int,
    threshold: float = 0.5,
    branching_factor: int = 50,
) -> pd.DataFrame:
    """
    Classifica as notas gerais usando o algoritmo BIRCH.
    Os parâmetros `n_clusters`, `threshold` e `branching_factor` podem ser ajustados.
    `threshold`: o raio do subcluster em torno do centroid.
    `branching_factor`: o número máximo de subclusters CF em cada nó.
    """
    copia_df = df.copy()

    score_cols = ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]
    df_for_clustering = copia_df.dropna(subset=score_cols).copy()

    if df_for_clustering.empty:
        print("Nenhum dado válido para agrupamento BIRCH.")
        copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = np.nan
        copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = np.nan
        return copia_df

    birch_model = Birch(
        n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor
    )

    df_for_clustering["CLUSTER_LABEL"] = birch_model.fit_predict(
        df_for_clustering[score_cols]
    )

    copia_df = copia_df.merge(
        df_for_clustering[["CLUSTER_LABEL"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    copia_df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = copia_df["CLUSTER_LABEL"]
    copia_df.drop(columns=["CLUSTER_LABEL"], inplace=True)

    return copia_df
