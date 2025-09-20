import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import MiniBatchKMeans


# ==============================================================================
# SEÇÃO DE CONFIGURAÇÃO
# ==============================================================================

# 1. Anos dos dados do ENEM a serem carregados e processados.
ANOS_PARA_CARREGAR = ["2023", "2022", "2021"]

# 2. Caminho base para os arquivos de dados. Use '{ano}' como um placeholder.
CAMINHO_BASE_DADOS = (
    "./preprocess/generico/microdados_enem_{ano}/DADOS/MICRODADOS_ENEM_{ano}.csv"
)

# 3. Caminho para salvar o arquivo de saída pré-processado.
# Este será o prefixo; o número de clusters será adicionado.
CAMINHO_SAIDA_PREFIXO = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA_2M"
)

# 4. Percentual de amostragem a ser aplicado (1.0 para 100%, 0.1 para 10%).
PERCENTUAL_AMOSTRAGEM = 1

# 5. Lista de colunas que devem ser selecionadas do dataset original.
# Apenas estas colunas serão carregadas na memória após a seleção.
COLUNAS_SELECIONADAS = [
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_LINGUA",
    "TP_DEPENDENCIA_ADM_ESC",
    "Q006",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
    "IN_TREINEIRO",  # Adicionada coluna de treineiro
    "TP_PRESENCA_CN",  # Adicionada presença na prova de Ciências da Natureza
    "TP_PRESENCA_CH",  # Adicionada presença na prova de Ciências Humanas
    "TP_PRESENCA_LC",  # Adicionada presença na prova de Linguagens e Códigos
    "TP_PRESENCA_MT",  # Adicionada presença na prova de Matemática
]

# 6. Colunas de notas usadas para calcular a média e verificar valores zerados.
COLUNAS_DE_NOTAS = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

# 7. Mapeamento para preenchimento de valores ausentes (NaN).
# Formato: {'NOME_DA_COLUNA': valor_de_preenchimento}
MAPEAMENTO_PREENCHIMENTO_NA = {
    # Exemplo: Preencher notas ausentes com 0, se desejado.
    "TP_DEPENDENCIA_ADM_ESC": 0,
    "NU_NOTA_CN": 0,
    "NU_NOTA_CH": 0,
    "NU_NOTA_LC": 0,
    "NU_NOTA_MT": 0,
    "NU_NOTA_REDACAO": 0,
}

# 8. Mapeamento para converter colunas de texto para números. (NOVA CONFIGURAÇÃO)
# Formato: {'NOME_DA_COLUNA': {'valor_texto_1': valor_num_1, ...}}
MAPEAMENTO_CATEGORICO_PARA_NUMERICO = {
    "TP_SEXO": {"F": 0, "M": 1},
    "Q006": {
        "A": 1,  # Nenhuma Renda
        "B": 2,  # Até R$ 1.212,00
        "C": 3,  # De R$ 1.212,01 até R$ 1.818,00
        "D": 4,  # De R$ 1.818,01 até R$ 2.424,00
        "E": 5,  # De R$ 2.424,01 até R$ 3.030,00
        "F": 6,  # De R$ 3.030,01 até R$ 3.636,00
        "G": 7,  # De R$ 3.636,01 até R$ 4.848,00
        "H": 8,  # De R$ 4.848,01 até R$ 6.060,00
        "I": 9,  # De R$ 6.060,01 até R$ 7.272,00
        "J": 10,  # De R$ 7.272,01 até R$ 8.484,00
        "K": 11,  # De R$ 8.484,01 até R$ 9.696,00
        "L": 12,  # De R$ 9.696,01 até R$ 10.908,00
        "M": 13,  # De R$ 10.908,01 até R$ 12.120,00
        "N": 14,  # De R$ 12.120,01 até R$ 14.544,00
        "O": 15,  # De R$ 14.544,01 até R$ 18.180,00
        "P": 16,  # De R$ 18.180,01 até R$ 24.240,00
        "Q": 17,  # Acima de R$ 24.240,00
    },
    # Mapeamento da Quantidade de Banheiros (Questão 008)
    "Q008": {
        "A": 0,  # Não
        "B": 1,  # Sim, um
        "C": 2,  # Sim, dois
        "D": 3,  # Sim, três
        "E": 4,  # Sim, quatro ou mais
    },
    # Mapeamento da Quantidade de Celulares (Questão 022)
    "Q022": {
        "A": 0,  # Não
        "B": 1,  # Sim, um
        "C": 2,  # Sim, dois
        "D": 3,  # Sim, três
        "E": 4,  # Sim, quatro ou mais
    },
    # Mapeamento da Quantidade de Computadores (Questão 024)
    "Q024": {
        "A": 0,  # Não
        "B": 1,  # Sim, um
        "C": 2,  # Sim, dois
        "D": 3,  # Sim, três
        "E": 4,  # Sim, quatro ou mais
    },
}


# ==============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ==============================================================================


def carregar_dados(
    anos: list,
    caminho_base: str,
    colunas_selecionadas_para_contagem: list,
    mapeamento_categorico_para_numerico: dict,
) -> pd.DataFrame:
    """
    Carrega, combina e retorna os dataframes do ENEM para os anos especificados.
    Após o carregamento e combinação, mostra a contagem inicial de valores para
    as colunas selecionadas, incluindo o valor original e o valor convertido (se aplicável).
    """
    todos_os_dfs = []
    print("Iniciando carregamento dos arquivos...")
    for ano in anos:
        caminho_arquivo = caminho_base.format(ano=ano)
        try:
            df = pd.read_csv(
                caminho_arquivo,
                delimiter=";",
                encoding="latin-1",
                skipinitialspace=True,
            )
            df.columns = df.columns.str.strip()
            todos_os_dfs.append(df)
            print(f"   - Arquivo de {ano} carregado com sucesso ({len(df)} linhas).")
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado em {caminho_arquivo}")
        except Exception as e:
            print(f"ERRO ao carregar o arquivo {caminho_arquivo}: {e}")

    if not todos_os_dfs:
        print("Nenhum arquivo de dados foi carregado. Encerrando.")
        return pd.DataFrame()

    df_combinado = pd.concat(todos_os_dfs, ignore_index=True, sort=False)
    print(f"Total de linhas combinadas: {len(df_combinado)}")

    print("\n[Contagem Inicial de Valores por Coluna Selecionada]")
    for col in colunas_selecionadas_para_contagem:
        if col in df_combinado.columns:
            print(f"\n--- Coluna: {col} ---")
            if col in mapeamento_categorico_para_numerico:
                # Get the mapping for the current column
                col_map = mapeamento_categorico_para_numerico[col]

                # Get counts of original values including NaNs
                original_counts = df_combinado[col].value_counts(dropna=False)

                # Iterate through these counts to display original/converted/count
                for original_val, count in original_counts.items():
                    # Check if the original_val is NaN
                    if pd.isna(original_val):
                        converted_val_str = "NaN"  # Represent NaN as string "NaN"
                    # Check if the original_val is in the mapping to convert it
                    elif original_val in col_map:
                        converted_val_str = str(col_map[original_val])
                    else:
                        # If an original value exists but is not in the mapping, keep its original string representation
                        converted_val_str = "Não Mapeado"

                    print(f"   {original_val} / {converted_val_str}: {count}")
            else:
                # For numerical or non-mapped categorical columns, just print value_counts
                print(df_combinado[col].value_counts(dropna=False))
        else:
            print(
                f"--- AVISO: Coluna '{col}' não encontrada no DataFrame combinado para contagem inicial."
            )

    return df_combinado


def aplicar_amostragem(df: pd.DataFrame, percentual: float) -> pd.DataFrame:
    """
    Aplica amostragem sobre o DataFrame se o percentual for menor que 1.0.
    """
    if not 0 <= percentual <= 1:
        raise ValueError("O percentual de amostragem deve estar entre 0 e 1.")

    linhas_antes = len(df)
    print(f"\n[Amostragem] Entrando com {linhas_antes} linhas.")

    if percentual < 1.0:
        print(f"Aplicando amostragem de {percentual:.2%} dos dados...")
        df_amostrado = df.sample(frac=percentual, random_state=42).reset_index(
            drop=True
        )
        linhas_depois = len(df_amostrado)
        removidas = linhas_antes - linhas_depois
        print(f"   - {removidas} linhas removidas pela amostragem.")
        print(f"[Amostragem] Saindo com {linhas_depois} linhas.")
        return df_amostrado

    print("   - Nenhuma amostragem aplicada (percentual é 100%).")
    print(f"[Amostragem] Saindo com {linhas_antes} linhas.")
    return df


def selecionar_colunas(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """
    Seleciona apenas as colunas de interesse de um DataFrame.
    """
    print(f"\n[Seleção de Colunas] Entrando com {len(df)} linhas.")
    print("Selecionando colunas de interesse...")
    colunas_existentes = [col for col in colunas if col in df.columns]
    df_selecionado = df[colunas_existentes].copy()
    print(f"[Seleção de Colunas] Saindo com {len(df_selecionado)} linhas.")
    return df_selecionado


def converter_categorico_para_numerico(
    df: pd.DataFrame, mapeamento: dict
) -> pd.DataFrame:
    """
    Converte colunas categóricas (texto) para valores numéricos com base em um dicionário.
    """
    print(f"\n[Conversão Categórica] Entrando com {len(df)} linhas.")
    if not mapeamento:
        print("   - Nenhum mapeamento para conversão foi fornecido.")
        print(f"[Conversão Categórica] Saindo com {len(df)} linhas.")
        return df

    print("Convertendo colunas categóricas para numéricas...")
    df_convertido = df.copy()
    for coluna, mapa in mapeamento.items():
        if coluna in df_convertido.columns:
            df_convertido[coluna] = df_convertido[coluna].replace(mapa)
            print(f"   - Coluna '{coluna}' convertida.")
        else:
            print(f"   - AVISO: Coluna '{coluna}' para conversão não encontrada.")

    print(f"[Conversão Categórica] Saindo com {len(df_convertido)} linhas.")
    return df_convertido


def preencher_valores_ausentes(
    df: pd.DataFrame, mapa_preenchimento: dict
) -> pd.DataFrame:
    """
    Preenche valores nulos (NaN) com base em um dicionário de mapeamento.
    Detalha a quantidade de linhas que tiveram valores preenchidos.
    """
    linhas_antes = len(df)
    print(f"\n[Preenchimento de Nulos] Entrando com {linhas_antes} linhas.")

    if mapa_preenchimento:
        print("Preenchendo valores ausentes...")

        # Criar um DataFrame booleano indicando quais valores são NaN antes do preenchimento
        nan_antes = df.isna()

        df.fillna(value=mapa_preenchimento, inplace=True)

        # Comparar o estado de NaN antes e depois para encontrar as células preenchidas
        preenchidos_por_coluna = (nan_antes & ~df.isna()).sum()

        total_celulas_preenchidas = preenchidos_por_coluna.sum()

        # Contar o número de linhas que tiveram *pelo menos um* valor preenchido
        linhas_com_preenchimento = (nan_antes & ~df.isna()).any(axis=1).sum()

        if total_celulas_preenchidas > 0:
            print(
                f"   - {total_celulas_preenchidas} valores (células) foram preenchidos no total."
            )
            print(
                f"   - {linhas_com_preenchimento} linhas tiveram pelo menos um valor preenchido."
            )
            print("   - Detalhe de valores preenchidos por coluna:")
            for col, count in preenchidos_por_coluna.items():
                if count > 0:
                    print(f"     - '{col}': {count} valores")
        else:
            print(
                "   - Nenhum valor ausente foi encontrado e preenchido com o mapeamento fornecido."
            )
    else:
        print("   - Nenhum mapeamento para preenchimento de nulos foi fornecido.")

    print(f"[Preenchimento de Nulos] Saindo com {len(df)} linhas.")
    return df


def remover_treineiros_e_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas que correspondem a treineiros ou a alunos ausentes em qualquer uma das provas.
    """
    linhas_antes = len(df)
    print(f"\n[Remoção de Treineiros e Ausentes] Entrando com {linhas_antes} linhas.")

    df_filtrado = df.copy()

    # Remover treineiros
    if "IN_TREINEIRO" in df_filtrado.columns:
        treineiros_removidos = df_filtrado[df_filtrado["IN_TREINEIRO"] == 1].shape[0]
        df_filtrado = df_filtrado[df_filtrado["IN_TREINEIRO"] == 0]
        print(f"   - {treineiros_removidos} linhas de treineiros removidas.")
    else:
        print(
            "   - AVISO: Coluna 'IN_TREINEIRO' não encontrada para remoção de treineiros."
        )

    # Remover ausentes em qualquer uma das provas
    colunas_presenca = [
        "TP_PRESENCA_CN",
        "TP_PRESENCA_CH",
        "TP_PRESENCA_LC",
        "TP_PRESENCA_MT",
    ]

    # Filtrar apenas as colunas de presença que realmente existem no DataFrame
    colunas_presenca_existentes = [
        col for col in colunas_presenca if col in df_filtrado.columns
    ]

    if colunas_presenca_existentes:
        # 0 = Ausente, 1 = Presente, 2 = Eliminado
        # Queremos manter apenas os presentes (1)
        condicao_presenca = pd.Series(True, index=df_filtrado.index)
        for col in colunas_presenca_existentes:
            # Garante que a coluna é numérica para a comparação
            df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors="coerce")
            # Um aluno é considerado ausente se TP_PRESENCA for 0 ou NaN (após coerce de não-numéricos)
            # Queremos manter apenas os que são 1 (presentes)
            condicao_presenca = condicao_presenca & (df_filtrado[col] == 1)

        ausentes_removidos = df_filtrado[~condicao_presenca].shape[0]
        df_filtrado = df_filtrado[condicao_presenca]
        print(
            f"   - {ausentes_removidos} linhas de ausentes em alguma prova removidas."
        )
    else:
        print(
            "   - AVISO: Nenhuma coluna de presença encontrada para remoção de ausentes."
        )

    linhas_depois = len(df_filtrado)
    removidas_total = linhas_antes - linhas_depois
    print(
        f"   - Total de {removidas_total} linhas removidas (treineiros e/ou ausentes)."
    )
    print(f"[Remoção de Treineiros e Ausentes] Saindo com {linhas_depois} linhas.")

    # Remover as colunas de presença após o filtro, se elas existirem
    df_filtrado.drop(columns=colunas_presenca_existentes, errors="ignore", inplace=True)
    print(f"   - Colunas de presença removidas: {colunas_presenca_existentes}")

    # Remover a coluna IN_TREINEIRO após o filtro, se existir
    if "IN_TREINEIRO" in df_filtrado.columns:
        df_filtrado.drop(columns=["IN_TREINEIRO"], errors="ignore", inplace=True)
        print(f"   - Coluna 'IN_TREINEIRO' removida.")

    return df_filtrado


def adicionar_nota_geral(df: pd.DataFrame, colunas_notas: list) -> pd.DataFrame:
    """
    Calcula e adiciona a coluna 'NOTA_GERAL' como a média das notas especificadas.
    """
    print(f"\n[Criação da Nota Geral] Entrando com {len(df)} linhas.")
    print("Criando a coluna 'NOTA_GERAL'...")
    df_com_nota = df.copy()
    notas_existentes = [col for col in colunas_notas if col in df_com_nota.columns]
    df_com_nota["NOTA_GERAL"] = df_com_nota[notas_existentes].mean(axis=1).round(2)
    print(f"[Criação da Nota Geral] Saindo com {len(df_com_nota)} linhas.")
    return df_com_nota


def adicionar_classificacao(
    df: pd.DataFrame, coluna_base: str, n_grupos: int
) -> pd.DataFrame:
    """
    Cria uma classificação utilizando MiniBatch K-Means e a adiciona ao DataFrame original.
    Os dados são escalados antes do agrupamento.
    """
    print(f"\n[Criação da Classificação] Entrando com {len(df)} linhas.")
    print("Criando a coluna 'CLASSIFICACAO' com MiniBatch K-Means...")
    df_final = df.copy()

    # Seleciona apenas a coluna numérica para agrupamento
    if coluna_base not in df_final.columns or not pd.api.types.is_numeric_dtype(
        df_final[coluna_base]
    ):
        print(
            f"ERRO: A coluna base '{coluna_base}' não é numérica ou não existe. Incapaz de agrupar."
        )
        df_final["CLASSIFICACAO"] = "Indefinida"
        return df_final

    # Escalamento dos dados: crucial para algoritmos baseados em distância como K-Means.
    # MinMaxScaler é usado, mas considere StandardScaler se a distribuição dos dados
    # for mais próxima de uma normal e você quiser que outliers não distorçam a escala.
    print("   - Escalando a coluna base para agrupamento...")
    scaler = MinMaxScaler()  # Você pode testar StandardScaler() aqui também
    data_for_clustering = scaler.fit_transform(df_final[[coluna_base]])

    # Inicializa e treina o modelo MiniBatch K-Means
    print(f"   - Executando MiniBatch K-Means com {n_grupos} clusters...")
    mbkmeans_model = MiniBatchKMeans(
        n_clusters=n_grupos,
        batch_size=256,  # Ajuste este valor conforme a memória e performance
        random_state=42,
        n_init="auto",
    )

    # As previsões serão usadas como a classificação
    df_final["CLASSIFICACAO"] = mbkmeans_model.fit_predict(data_for_clustering)

    # Opcional: Verificar o balanceamento dos clusters
    unique_labels, counts = np.unique(df_final["CLASSIFICACAO"], return_counts=True)
    print("\n   - Tamanho dos clusters finais (MiniBatch K-Means):")
    for label, count in zip(unique_labels, counts):
        print(
            f"     - Cluster {label}: {count} pontos ({count/len(df_final)*100:.2f}%)"
        )

    print(f"[Criação da Classificação] Saindo com {len(df_final)} linhas.")
    return df_final


def salvar_dados(df: pd.DataFrame, caminho: str):
    """
    Salva o DataFrame final em um arquivo CSV.
    """
    if df.empty:
        print("\nNenhum dado para salvar.")
        return

    print(f"\nSalvando arquivo pré-processado em: {caminho}")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, sep=";", index=False, encoding="utf-8")
    print("Arquivo salvo com sucesso!")


def run_multiple_clusterings(
    anos_para_carregar: list,
    caminho_base_dados: str,
    caminho_saida_prefixo: str,
    percentual_amostragem: float,
    colunas_selecionadas: list,
    colunas_de_notas: list,
    mapeamento_preenchimento_na: dict,
    mapeamento_categorico_para_numerico: dict,
    range_n_clusters: range,
    coluna_base_classificacao: str = "NOTA_GERAL",
):
    """
    Roda o pipeline de pré-processamento múltiplas vezes, variando o número de clusters,
    e salva cada resultado em um arquivo separado.
    """
    print("\n--- INICIANDO EXECUÇÕES MÚLTIPLAS DE AGRUPAMENTO ---")

    # Passo 1: Carregar os arquivos (feito apenas uma vez para todos os runs)
    # df_bruto é o DataFrame original combinado de todos os anos
    df_bruto = carregar_dados(
        anos_para_carregar,
        caminho_base_dados,
        colunas_selecionadas,
        mapeamento_categorico_para_numerico,
    )  # Passando as colunas selecionadas e o mapeamento para a contagem
    if df_bruto.empty:
        print("Dados brutos não carregados. Encerrando execuções múltiplas.")
        return

    # A partir daqui, as etapas de pré-processamento (exceto a classificação)
    # são executadas uma vez no DataFrame bruto para evitar reprocessamento desnecessário
    # para cada variação de clusters.
    print(
        "\n--- Executando etapas de pré-processamento inicial (comum a todos os runs) ---"
    )

    # Passo 2: Aplicar amostragem configurável
    df_amostrado = aplicar_amostragem(df_bruto, percentual_amostragem)

    # Passo 3: Selecionar apenas as colunas desejadas
    df_selecionado = selecionar_colunas(df_amostrado, colunas_selecionadas)

    # Passo 4: Remover treineiros e ausentes
    df_filtrado_treineiro_ausente = remover_treineiros_e_ausentes(df_selecionado)

    # Passo 5: Preencher valores não definidos
    df_preenchido = preencher_valores_ausentes(
        df_filtrado_treineiro_ausente, mapeamento_preenchimento_na
    )

    # PASSO AJUSTADO: Converter colunas categóricas para numéricas AGORA (após preenchimento)
    df_convertido = converter_categorico_para_numerico(
        df_preenchido, mapeamento_categorico_para_numerico
    )

    # Passo 7: Criar a coluna de nota geral
    df_base_para_cluster = adicionar_nota_geral(df_convertido, colunas_de_notas)

    print(
        "\n--- Etapas iniciais de pré-processamento concluídas. Iniciando agrupamentos. ---"
    )

    for num_clusters in range_n_clusters:
        print(f"\n\n=========== EXECUTANDO PARA {num_clusters} CLUSTERS =============")

        # Passo 8: Criar a coluna de classificação para o número atual de clusters
        # Usamos uma cópia do df_base_para_cluster para não interferir nas próximas iterações
        df_com_classificacao = adicionar_classificacao(
            df_base_para_cluster.copy(), coluna_base_classificacao, num_clusters
        )

        # Atualizar o caminho de saída com o número de clusters
        caminho_saida_atual = f"{caminho_saida_prefixo}_{num_clusters}C.csv"

        # Passo 9: Salvar o arquivo final para esta iteração
        salvar_dados(df_com_classificacao, caminho_saida_atual)

        print(
            f"=========== FIM DA EXECUÇÃO PARA {num_clusters} CLUSTERS ============\n"
        )

    print(
        "\n--- TODAS AS EXECUÇÕES MÚLTIPLAS DE AGRUPAMENTO CONCLUÍDAS COM SUCESSO ---"
    )


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================


def main():
    """
    Orquestra o pipeline de pré-processamento de ponta a ponta.
    """
    print("--- INICIANDO SCRIPT DE PRÉ-PROCESSAMENTO DO ENEM ---")

    # Você pode descomentar esta seção para rodar o pipeline para múltiplos clusters
    # e comentar a seção "Execução única" abaixo.
    # -------------------------------------------------------------------------
    # Execução para Múltiplos Clusters (de 2 a 6)
    # -------------------------------------------------------------------------
    # run_multiple_clusterings(
    #     anos_para_carregar=ANOS_PARA_CARREGAR,
    #     caminho_base_dados=CAMINHO_BASE_DADOS,
    #     caminho_saida_prefixo=CAMINHO_SAIDA_PREFIXO,
    #     percentual_amostragem=PERCENTUAL_AMOSTRAGEM,
    #     colunas_selecionadas=COLUNAS_SELECIONADAS,
    #     colunas_de_notas=COLUNAS_DE_NOTAS,
    #     mapeamento_preenchimento_na=MAPEAMENTO_PREENCHIMENTO_NA,
    #     mapeamento_categorico_para_numerico=MAPEAMENTO_CATEGORICO_PARA_NUMERICO,
    #     range_n_clusters=range(2, 7),  # Roda de 2 a 6 (o 7 não é incluído)
    #     coluna_base_classificacao="NOTA_GERAL",
    # )
    # Fim da execução para múltiplos clusters.
    # Se você quiser apenas esta execução, pode retornar aqui:
    # return
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Execução Única (Descomente esta seção se não usar a de múltiplos clusters)
    # -------------------------------------------------------------------------
    print("--- Execução de pipeline único ---")

    # Passando as colunas selecionadas e o mapeamento para a contagem inicial
    df_bruto = carregar_dados(
        ANOS_PARA_CARREGAR,
        CAMINHO_BASE_DADOS,
        COLUNAS_SELECIONADAS,
        MAPEAMENTO_CATEGORICO_PARA_NUMERICO,
    )
    if df_bruto.empty:
        return

    df_amostrado = aplicar_amostragem(df_bruto, PERCENTUAL_AMOSTRAGEM)
    df_selecionado = selecionar_colunas(df_amostrado, COLUNAS_SELECIONADAS)
    df_filtrado_treineiro_ausente = remover_treineiros_e_ausentes(df_selecionado)
    df_preenchido = preencher_valores_ausentes(
        df_filtrado_treineiro_ausente, MAPEAMENTO_PREENCHIMENTO_NA
    )
    # PASSO AJUSTADO: Converter colunas categóricas para numéricas AGORA (após preenchimento)
    df_convertido = converter_categorico_para_numerico(
        df_preenchido, MAPEAMENTO_CATEGORICO_PARA_NUMERICO
    )
    df_com_nota = adicionar_nota_geral(df_convertido, COLUNAS_DE_NOTAS)

    # Para a execução única, você precisa definir NUMERO_DE_GRUPOS_CLASSIFICACAO
    # na seção de configuração no topo do arquivo.
    df_final = adicionar_classificacao(df_com_nota, "NOTA_GERAL", 3)

    # Para a execução única, o caminho de saída precisa ser definido na config.
    # Você pode definir CAMINHO_SAIDA = f"{CAMINHO_SAIDA_PREFIXO}_UNICO.csv" se quiser.
    salvar_dados(
        df_final,
        "./preprocess/generico/microdados_enem_combinado/DADOS/PREPROCESSED_DATA.csv",
    )

    print("\n--- PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO ---")
    if not df_final.empty:
        print(f"\nResumo do DataFrame Final ({len(df_final)} linhas):")
        print(df_final.head())
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()
