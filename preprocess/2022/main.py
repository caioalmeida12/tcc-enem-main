import numpy as np
import pandas as pd
import os
import csv
from toolz import pipe

"""
Esse script tem como objetivo realizar o pré-processamento dos microdados do Enem 2022.
O pré-processamento consiste em:
1. Carregar os microdados do Enem 2022 (./microdados_enem_2022/DADOS/MICRODADOS_ENEM_2022.csv).
2. Selecionar apenas as colunas de interesse.
3. Remover valores inválidos (nulos, ausentes, etc).
4. Criar novas colunas a partir das colunas existentes.
5. Salvar os dados pré-processados em um arquivo CSV (./microdados_enem_2022/PREPROCESS/PREPROCESSED_DATA.csv).
"""

colunas_notas = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

colunas_interesse = [
    # Fator 1
    "Q012",
    "Q013",
    "Q014",
    "Q015",
    "Q016",
    "Q018",
    "Q019",
    "Q020",
    "Q021",
    "Q025",
    # Fator 2
    "Q002",
    "Q003",
    "Q004",
    # Fator 3
    "Q008",
    "Q009",
    "Q022",
    # Fator 4
    "TP_DEPENDENCIA_ADM_ESC",
    "Q006",
    "Q010",
    "Q018",
    "Q024",
]


def carregar_arquivo(caminho: str):
    dados: list[dict[str, str]] = []

    with open(caminho, newline="", encoding="latin-1") as file:
        conteudo = csv.DictReader(file, delimiter=";", skipinitialspace=True)
        conteudo.fieldnames = [name.strip() for name in conteudo.fieldnames]

        for i, row in enumerate(conteudo):
            dados.append(row)

    return dados


def pegar_colunas_de_interresse(arquivo: list[dict[str, str]]):
    arquivo_filtrado: list[dict[str, str]] = []

    for row in arquivo:
        row_filtrada: dict[str, str] = {}

        for key in row.keys():
            if key in colunas_interesse or key in colunas_notas:
                row_filtrada[key] = row[key]

        arquivo_filtrado.append(row_filtrada)

    return arquivo_filtrado


def remover_linhas_com_valores_invalidos(arquivo: list[dict[str, str]]):
    arquivo_validado: list[dict[str | any, str | any]] = []

    for row in arquivo:
        if all(v is not None and str(v).strip() != "" for v in row.values()):
            arquivo_validado.append(row)

    return arquivo_validado


def remover_linhas_com_notas_zeradas(arquivo: list[dict[str, str]]):
    arquivo_filtrado: list[dict[str, str]] = []

    for row in arquivo:
        notas_validas = all(
            float(row[nota].replace(",", ".") if "," in row[nota] else row[nota]) > 0
            for nota in colunas_notas
        )

        if notas_validas:
            arquivo_filtrado.append(row)

    return arquivo_filtrado


def calcular_nota_geral_sem_redacao(row: dict[str, str]):
    nota_total = sum(float(row[nota]) for nota in colunas_notas)

    nota_total -= float(row["NU_NOTA_REDACAO"])

    return round(nota_total / len(colunas_notas) - 1, 2)


def calcular_nota_geral_com_redacao(row: dict[str, str]):
    nota_total = sum(float(row[nota]) for nota in colunas_notas)

    return round(nota_total / len(colunas_notas), 2)


def criar_novas_colunas(arquivo: list[dict[str, str]]):
    for row in arquivo:
        row["NOTA_GERAL_COM_REDACAO"] = str(calcular_nota_geral_com_redacao(row))
        row["NOTA_GERAL_SEM_REDACAO"] = str(calcular_nota_geral_sem_redacao(row))

    return arquivo


def salvar_arquivo_preprocessado(dados: list[dict[str, str]], caminho: str):
    if not dados:
        print("Nenhum dado para salvar.")
        return

    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dados[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(dados)

    print(f"Arquivo salvo com sucesso em: {caminho}")


def main():
    caminho_saida = (
        "./preprocess/2022/microdados_enem_2022/PREPROCESS/PREPROCESSED_DATA_100K.csv"
    )

    preprocessados = pipe(
        "./preprocess/2022/microdados_enem_2022/DADOS/MICRODADOS_ENEM_2022_100K.csv",
        carregar_arquivo,
        pegar_colunas_de_interresse,
        remover_linhas_com_valores_invalidos,
        remover_linhas_com_notas_zeradas,
        criar_novas_colunas,
    )

    salvar_arquivo_preprocessado(preprocessados, caminho_saida)


if __name__ == "__main__":
    main()
