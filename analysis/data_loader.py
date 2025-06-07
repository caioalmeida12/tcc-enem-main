import pandas as pd
import streamlit as st
import numpy as np
from utils import COLUNAS_NOTAS  # Assuming COLUNAS_NOTAS is defined in utils.py


def load_preprocessed_data(path):
    """
    Loads preprocessed data from a CSV file, explicitly converting
    known numerical columns to numeric types.
    """
    try:
        df = pd.read_csv(path, sep=";", dtype=str)

        numeric_cols_to_convert = [
            "NU_NOTA_CN",
            "NU_NOTA_CH",
            "NU_NOTA_LC",
            "NU_NOTA_MT",
            "NU_NOTA_COMP1",
            "NU_NOTA_COMP2",
            "NU_NOTA_COMP3",
            "NU_NOTA_COMP4",
            "NU_NOTA_COMP5",
            "NU_NOTA_REDACAO",
            "NOTA_GERAL_COM_REDACAO",
            "NOTA_GERAL_SEM_REDACAO",
            "Q006",
            "Q022",
            "Q024",
            "Q008",
        ]

        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo não encontrado no caminho: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar ou pré-processar os dados: {e}")
        st.stop()


def identify_feature_types(df):
    """
    Identifies categorical and numerical features in the DataFrame.
    Converts object columns to category if they have few unique values.
    """
    cat_features = []
    num_features = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_features.append(col)
        elif df[col].nunique() < 50 and df[col].dtype == "object":
            df[col] = df[col].astype("category")
            cat_features.append(col)
        elif df[col].dtype == "object":
            cat_features.append(col)

    exclude_from_features = [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
        "NOTA_GERAL_COM_REDACAO",
        "NOTA_GERAL_SEM_REDACAO",
    ] + COLUNAS_NOTAS

    cat_features = [f for f in cat_features if f not in exclude_from_features]
    num_features = [f for f in num_features if f not in exclude_from_features]

    return df, sorted(cat_features), sorted(num_features)
