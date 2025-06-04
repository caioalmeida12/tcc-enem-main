import numpy as np
import pandas as pd
import os
import csv
from toolz import pipe

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

"""
Esse script tem como objetivo realizar o pré-processamento dos microdados do Enem 2023.
O pré-processamento consiste em:
1. Carregar os microdados do Enem 2023 (./microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv).
2. Selecionar todas as colunas do CSV (padrão).
3. Remover valores inválidos (nulos, ausentes, etc).
4. Criar novas colunas a partir das colunas existentes.
5. Salvar os dados pré-processados em um arquivo CSV (./microdados_enem_2023/PREPROCESS/PREPROCESSED_DATA.csv).
"""

anos_para_carregar = ["2021", "2022", "2023"]

colunas_notas = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

colunas_interesse_original = []


def carregar_arquivo(caminho: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV em um DataFrame do Pandas.
    """
    try:
        df = pd.read_csv(
            caminho, delimiter=";", encoding="latin-1", skipinitialspace=True
        )
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro ao carregar o arquivo {caminho}: {e}")
        return pd.DataFrame()


def pegar_colunas_de_interresse(df: pd.DataFrame, colunas: list = None) -> pd.DataFrame:
    """
    Seleciona apenas as colunas de interesse de um DataFrame.
    Se nenhuma coluna for especificada, todas as colunas serão selecionadas.
    """
    if colunas is None:
        return df.copy()
    else:
        colunas_existentes = [col for col in colunas if col in df.columns]
        return df[colunas_existentes].copy()


def remover_colunas_especificas(
    df: pd.DataFrame, colunas_para_remover: list
) -> pd.DataFrame:
    """
    Remove colunas específicas de um DataFrame.
    """
    df_copy = df.copy()
    existing_columns_to_drop = [
        col for col in colunas_para_remover if col in df_copy.columns
    ]
    return df_copy.drop(columns=existing_columns_to_drop)


def remover_linhas_com_valores_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com quaisquer valores nulos (NaN) de um DataFrame.
    Pandas handle various representations of missing data as NaN.
    """
    return df.dropna().copy()


def remover_linhas_com_notas_zeradas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas onde qualquer uma das notas é zero.
    Converte as colunas de nota para numérico antes de verificar.
    """
    copia_df = df.copy()
    for col in colunas_notas:
        if col in copia_df.columns:
            copia_df[col] = pd.to_numeric(
                copia_df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    for col in colunas_notas:
        if col in copia_df.columns:
            copia_df = copia_df[copia_df[col].notna() & (copia_df[col] > 0)]

    return copia_df


def converter_opcoes_letra_para_numero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas com opções de letra (e.g., Q001, Q002) e classificações para números.
    """
    copia_df = df.copy()

    potential_letter_cols = [
        col
        for col in copia_df.columns
        if pd.api.types.is_string_dtype(copia_df[col])
        and copia_df[col].str.isalpha().any()
        and copia_df[col].astype(str).str.len().max() == 1
    ]

    colunas_classificacao = [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
    ]

    for col in potential_letter_cols + colunas_classificacao:
        if col in copia_df.columns and pd.api.types.is_string_dtype(copia_df[col]):
            copia_df[col] = copia_df[col].apply(
                lambda x: (
                    str(ord(x.upper()) - ord("A"))
                    if isinstance(x, str) and len(x) == 1 and x.isalpha()
                    else x
                )
            )
            copia_df[col] = pd.to_numeric(copia_df[col], errors="coerce")
    return copia_df


def criar_novas_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria as colunas de nota geral com e sem redação.
    """
    copia_df = df.copy()
    for col in colunas_notas:
        if col in copia_df.columns:
            copia_df[col] = pd.to_numeric(copia_df[col], errors="coerce")

    copia_df.dropna(
        subset=[col for col in colunas_notas if col in copia_df.columns], inplace=True
    )

    if not copia_df.empty:
        existing_colunas_notas = [
            col for col in colunas_notas if col in copia_df.columns
        ]
        if existing_colunas_notas:
            copia_df["NOTA_GERAL_COM_REDACAO"] = (
                copia_df[existing_colunas_notas].mean(axis=1).round(2)
            )
        else:
            copia_df["NOTA_GERAL_COM_REDACAO"] = np.nan

        cols_sem_redacao = [
            col for col in existing_colunas_notas if col != "NU_NOTA_REDACAO"
        ]
        if cols_sem_redacao:
            copia_df["NOTA_GERAL_SEM_REDACAO"] = (
                copia_df[cols_sem_redacao].mean(axis=1).round(2)
            )
        else:
            copia_df["NOTA_GERAL_SEM_REDACAO"] = np.nan

    return copia_df


def criar_coluna_classificacao(df: pd.DataFrame) -> pd.DataFrame:
    def classificar(nota):
        if pd.isna(nota):
            return np.nan
        if nota <= 400:
            return "400 ou menos"
        elif nota < 500:
            return "400-499"
        elif nota < 600:
            return "500-599"
        elif nota < 700:
            return "600-699"
        else:
            return "700 ou mais"

    if "NOTA_GERAL_COM_REDACAO" in df.columns:
        df["CLASSIFICACAO_NOTA_GERAL_COM_REDACAO"] = df["NOTA_GERAL_COM_REDACAO"].apply(
            classificar
        )
    if "NOTA_GERAL_SEM_REDACAO" in df.columns:
        df["CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO"] = df["NOTA_GERAL_SEM_REDACAO"].apply(
            classificar
        )
    return df


def normalizar_valores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os valores numéricos usando MinMaxScaler do scikit-learn.
    """
    copia_df = df.copy()

    numeric_cols = copia_df.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        scaler = MinMaxScaler()
        copia_df[numeric_cols] = scaler.fit_transform(copia_df[numeric_cols])
        copia_df[numeric_cols] = copia_df[numeric_cols].round(2)

    return copia_df


def salvar_arquivo_preprocessado(df: pd.DataFrame, caminho: str):
    """
    Salva o DataFrame pré-processado em um arquivo CSV.
    """
    if df.empty:
        print("Nenhum dado para salvar.")
        return

    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    df.to_csv(caminho, sep=";", index=False, encoding="utf-8")
    print(f"Arquivo salvo com sucesso em: {caminho}")


def main():
    caminho_base = (
        "./preprocess/generico/microdados_enem_{ano}/DADOS/MICRODADOS_ENEM_{ano}.csv"
    )
    caminhos_dos_arquivos = [caminho_base.format(ano=ano) for ano in anos_para_carregar]

    caminho_saida = "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"

    todos_os_arquivos = [carregar_arquivo(caminho) for caminho in caminhos_dos_arquivos]

    todos_os_arquivos = [df for df in todos_os_arquivos if not df.empty]

    if not todos_os_arquivos:
        print("Nenhum arquivo de dados válido foi carregado. Saindo.")
        return

    dataframe_combinado = pd.concat(todos_os_arquivos, ignore_index=True, sort=False)
    colunas_para_remover_exemplo = [
        "NU_INSCRICAO",
        "IN_TREINEIRO",
        "NO_MUNICIPIO_ESC",
        "NO_MUNICIPIO_PROVA",
        "TX_RESPOSTAS_CN",
        "TX_GABARITO_CN",
        "TX_RESPOSTAS_CH",
        "TX_GABARITO_CH",
        "TX_RESPOSTAS_LC",
        "TX_GABARITO_LC",
        "TX_RESPOSTAS_MT",
        "TX_GABARITO_MT",
    ]

    processed_df = pipe(
        dataframe_combinado,
        pegar_colunas_de_interresse,
        lambda df: remover_colunas_especificas(df, colunas_para_remover_exemplo),
        remover_linhas_com_valores_invalidos,
        remover_linhas_com_notas_zeradas,
        criar_novas_colunas,
        criar_coluna_classificacao,
        # converter_opcoes_letra_para_numero,
        # normalizar_valores,
    )

    salvar_arquivo_preprocessado(processed_df, caminho_saida)
    print("Pré-processamento concluído com sucesso.")


if __name__ == "__main__":
    main()
