import numpy as np
import pandas as pd
import os
from toolz import pipe

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

# Importar as funções de clustering do novo módulo
from preprocess.clustering.clustering_models import (
    classificar_com_agrupamento_hierarquico,
    classificar_com_dbscan,
    classificar_com_birch,
)


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

# Dicionário de mapeamento para variáveis categóricas
mapeamento_categorias = {
    "TP_FAIXA_ETARIA": {
        "1": 16.5,  # Menor de 17 anos (estimativa)
        "2": 17,
        "3": 18,
        "4": 19,
        "5": 20,
        "6": 21,
        "7": 22,
        "8": 23,
        "9": 24,
        "10": 25,
        "11": 28,  # Entre 26 e 30 anos (midpoint)
        "12": 33,  # Entre 31 e 35 anos (midpoint)
        "13": 38,  # Entre 36 e 40 anos (midpoint)
        "14": 43,  # Entre 41 e 45 anos (midpoint)
        "15": 48,  # Entre 46 e 50 anos (midpoint)
        "16": 53,  # Entre 51 e 55 anos (midpoint)
        "17": 58,  # Entre 56 e 60 anos (midpoint)
        "18": 63,  # Entre 61 e 65 anos (midpoint)
        "19": 68,  # Entre 66 e 70 anos (midpoint)
        "20": 70.5,  # Maior de 70 anos (estimativa)
    },
    "TP_SEXO": {"M": 0, "F": 1},
    "TP_ESTADO_CIVIL": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    },
    "TP_COR_RACA": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
    },
    "TP_NACIONALIDADE": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    },
    "TP_ST_CONCLUSAO": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    },
    "TP_ANO_CONCLUIU": {
        "0": 0,
        "1": 2022,
        "2": 2021,
        "3": 2020,
        "4": 2019,
        "5": 2018,
        "6": 2017,
        "7": 2016,
        "8": 2015,
        "9": 2014,
        "10": 2013,
        "11": 2012,
        "12": 2011,
        "13": 2010,
        "14": 2009,
        "15": 2008,
        "16": 2007,
        "17": 2006.5,  # Antes de 2007 (estimativa)
    },
    "TP_ESCOLA": {"1": 1, "2": 2, "3": 3},
    "TP_ENSINO": {"1": 1, "2": 2},
    "IN_TREINEIRO": {"0": 0, "1": 1},
    "TP_DEPENDENCIA_ADM_ESC": {"1": 1, "2": 2, "3": 3, "4": 4},
    "TP_LOCALIZACAO_ESC": {"1": 1, "2": 2},
    "TP_SIT_FUNC_ESC": {"1": 1, "2": 2, "3": 3, "4": 4},
    "TP_PRESENCA_CN": {"0": 0, "1": 1, "2": 2},
    "TP_PRESENCA_CH": {"0": 0, "1": 1, "2": 2},
    "TP_PRESENCA_LC": {"0": 0, "1": 1, "2": 2},
    "TP_PRESENCA_MT": {"0": 0, "1": 1, "2": 2},
    "TP_LINGUA": {"0": 0, "1": 1},
    "TP_STATUS_REDACAO": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
    },
    "Q001": {
        "A": 0,
        "B": 3,  # Não completou a 4ª série/5º ano (midpoint de 1-4)
        "C": 6.5,  # Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano (midpoint de 5-8)
        "D": 10.5,  # Completou a 8ª série/9º ano, mas não completou o Ensino Médio (midpoint de 9-12)
        "E": 14,  # Completou o Ensino Médio, mas não completou a Faculdade (12 + 2 anos de faculdade estimados)
        "F": 18,  # Completou a Faculdade, mas não completou a Pós-graduação (16 + 2 anos de pós estimados)
        "G": 20,  # Completou a Pós-graduação (18 + 2 anos de pós estimados)
        "H": np.nan,  # Não sei
    },
    "Q002": {
        "A": 0,
        "B": 3,
        "C": 6.5,
        "D": 10.5,
        "E": 14,
        "F": 18,
        "G": 20,
        "H": np.nan,
    },
    "Q003": {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": np.nan,  # Não sei
    },
    "Q004": {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": np.nan,
    },
    "Q005": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,
        "17": 17,
        "18": 18,
        "19": 19,
        "20": 20,
    },
    "Q006": {
        "A": 0,  # Nenhuma Renda
        "B": 660.00,  # Até R$ 1.320,00 (midpoint)
        "C": 1650.00,  # De R$ 1.320,01 até R$ 1.980,00 (midpoint)
        "D": 2310.50,  # De R$ 1.980,01 até R$ 2.640,00 (midpoint)
        "E": 2970.50,  # De R$ 2.640,01 até R$ 3.300,00 (midpoint)
        "F": 3630.50,  # De R$ 3.300,01 até R$ 3.960,00 (midpoint)
        "G": 4620.50,  # De R$ 3.960,01 até R$ 5.280,00 (midpoint)
        "H": 5940.50,  # De R$ 5.280,01 até R$ 6.600,00 (midpoint)
        "I": 7260.50,  # De R$ 6.600,01 até R$ 7.920,00 (midpoint)
        "J": 8580.50,  # De R$ 7.920,01 até R$ 9240,00 (midpoint)
        "K": 9900.50,  # De R$ 9.240,01 até R$ 10.560,00 (midpoint)
        "L": 11220.50,  # De R$ 10.560,01 até R$ 11.880,00 (midpoint)
        "M": 12540.50,  # De R$ 11.880,01 até R$ 13.200,00 (midpoint)
        "N": 14520.50,  # De R$ 13.200,01 até R$ 15.840,00 (midpoint)
        "O": 17820.50,  # De R$ 15.840,01 até R$19.800,00 (midpoint)
        "P": 23100.50,  # De R$ 19.800,01 até R$ 26.400,00 (midpoint)
        "Q": 27000.00,  # Acima de R$ 26.400,00 (estimativa)
    },
    "Q007": {"A": 0, "B": 1, "C": 2, "D": 3},
    "Q008": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q009": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q010": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q011": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q012": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q013": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q014": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q015": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q016": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q017": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q018": {"A": 0, "B": 1},
    "Q019": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q020": {"A": 0, "B": 1},
    "Q021": {"A": 0, "B": 1},
    "Q022": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q023": {"A": 0, "B": 1},
    "Q024": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q025": {"A": 0, "B": 1},
}


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
    Converte colunas com opções de letra e classificações para números,
    utilizando o dicionário de mapeamento global `mapeamento_categorias`.
    Para faixas numéricas, converte para o ponto médio.
    """
    copia_df = df.copy()

    for col, mapping in mapeamento_categorias.items():
        if col in copia_df.columns and pd.api.types.is_string_dtype(copia_df[col]):
            # Use .loc to avoid SettingWithCopyWarning
            copia_df.loc[:, col] = copia_df[col].map(mapping).fillna(copia_df[col])
            copia_df.loc[:, col] = pd.to_numeric(copia_df[col], errors="coerce")
        elif col in copia_df.columns and pd.api.types.is_numeric_dtype(copia_df[col]):
            # Handle numeric columns that might have categorical meaning (e.g., TP_FAIXA_ETARIA)
            # Ensure they are mapped if specified in the dictionary
            copia_df.loc[:, col] = (
                copia_df[col].astype(str).map(mapping).fillna(copia_df[col])
            )
            copia_df.loc[:, col] = pd.to_numeric(copia_df[col], errors="coerce")

    # Handle CLASSIFICACAO_NOTA_GERAL_COM_REDACAO and CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO if they exist
    colunas_classificacao = [
        "CLASSIFICACAO_NOTA_GERAL_COM_REDACAO",
        "CLASSIFICACAO_NOTA_GERAL_SEM_REDACAO",
    ]
    for col in colunas_classificacao:
        if col in copia_df.columns and pd.api.types.is_string_dtype(copia_df[col]):
            copia_df.loc[:, col] = copia_df[col].apply(
                lambda x: (
                    str(ord(x.upper()) - ord("A"))
                    if isinstance(x, str) and len(x) == 1 and x.isalpha()
                    else x
                )
            )
            copia_df.loc[:, col] = pd.to_numeric(copia_df[col], errors="coerce")

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


def plot_dendrogram(df: pd.DataFrame):
    """
    Gera e exibe o dendrograma para auxiliar na escolha do número de clusters.
    """
    score_cols = ["NOTA_GERAL_COM_REDACAO", "NOTA_GERAL_SEM_REDACAO"]
    df_for_clustering = df.dropna(subset=score_cols).copy()

    if df_for_clustering.empty:
        print("Nenhum dado válido para gerar o dendrograma.")
        return

    # Gera a matriz de ligação
    # Usando 'ward' linkage para minimizar a variância dentro de cada cluster
    Z = linkage(df_for_clustering[score_cols], method="ward")

    # Plota o dendrograma
    plt.figure(figsize=(15, 7))
    plt.title("Dendrograma para Agrupamento Hierárquico das Notas")
    plt.xlabel("Número de Amostras ou (Índice da Amostra)")
    plt.ylabel("Distância")
    dendrogram(
        Z,
        leaf_rotation=90.0,  # Rotaciona os rótulos do eixo x
        leaf_font_size=8.0,  # Tamanho da fonte para os rótulos do eixo x
        truncate_mode="lastp",  # Mostra apenas os últimos p clusters mesclados
        p=30,  # Mostra os últimos 30 clusters mesclados (ajuste conforme necessário)
        show_leaf_counts=True,  # Mostra a contagem de folhas em cada nó
        show_contracted=True,  # Mostra parênteses com contagem de folhas em nós contraídos
    )
    plt.grid(True)
    plt.show()


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


def sample_dataframe(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Aplica amostragem sobre o DataFrame.
    `percentage` deve ser um valor entre 0 e 1.
    """
    if not 0 <= percentage <= 1:
        raise ValueError("A porcentagem de amostragem deve estar entre 0 e 1.")

    if df.empty:
        print("DataFrame vazio, sem amostragem para aplicar.")
        return df

    return df.sample(frac=percentage, random_state=42).reset_index(drop=True)


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
    # Caminhos para os arquivos de microdados do ENEM
    caminho_base = (
        "./preprocess/generico/microdados_enem_{ano}/DADOS/MICRODADOS_ENEM_{ano}.csv"
    )
    caminhos_dos_arquivos = [caminho_base.format(ano=ano) for ano in anos_para_carregar]

    # Caminho de saída para o arquivo pré-processado
    caminho_saida = "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"

    # Porcentagem de amostragem (0.05 = 5% dos dados)
    sampling_percentage = 1

    # Carrega todos os arquivos de microdados
    todos_os_arquivos = [carregar_arquivo(caminho) for caminho in caminhos_dos_arquivos]
    todos_os_arquivos = [df for df in todos_os_arquivos if not df.empty]

    if not todos_os_arquivos:
        print("Nenhum arquivo de dados válido foi carregado. Saindo.")
        return

    # Concatena os DataFrames carregados
    dataframe_combinado = pd.concat(todos_os_arquivos, ignore_index=True, sort=False)

    # Colunas a serem removidas do DataFrame
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

    # Pipeline de pré-processamento
    processed_df = pipe(
        dataframe_combinado,
        pegar_colunas_de_interresse,
        lambda df: remover_colunas_especificas(df, colunas_para_remover_exemplo),
        remover_linhas_com_valores_invalidos,
        remover_linhas_com_notas_zeradas,
        converter_opcoes_letra_para_numero,  # Adicionado ao pipeline
        lambda df: sample_dataframe(df, sampling_percentage),
        criar_novas_colunas,
        lambda df: classificar_com_birch(df, n_clusters=4),  # Usando a função importada
    )

    # Salva o DataFrame pré-processado
    salvar_arquivo_preprocessado(processed_df, caminho_saida)
    print("Pré-processamento concluído com sucesso.")


if __name__ == "__main__":
    main()
