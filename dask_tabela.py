import dask.dataframe as dd
import pandas as pd
import dask
import numpy as np  # Importa a biblioteca NumPy


# --- FUNÇÃO AUXILIAR PARA AJUSTAR AS PORCENTAGENS ---
def ajustar_porcentagens_para_100(series: pd.Series) -> pd.Series:
    """
    Ajusta uma Série de porcentagens para que a soma seja exatamente 100.00
    usando o método do maior resto (largest remainder method).

    Args:
        series: Uma Série pandas contendo as porcentagens de uma categoria.

    Returns:
        Uma nova Série pandas com as porcentagens ajustadas.
    """
    # Trabalha com valores escalados em 10000 (para 2 casas decimais de precisão)
    scaled_values = series * 100
    floor_values = np.floor(scaled_values).astype(int)
    remainders = scaled_values - floor_values

    # Calcula a diferença para chegar a 100.00 (ou 10000 na escala)
    total_floor = floor_values.sum()
    difference = 10000 - total_floor

    # Distribui a diferença para os valores com os maiores restos
    if difference > 0:
        top_indices = remainders.nlargest(int(difference)).index
        floor_values.loc[top_indices] += 1

    # Converte de volta para porcentagens e retorna
    return floor_values / 100.0


# Supondo que o arquivo de dados esteja no caminho especificado
try:
    df = dd.read_csv(
        "./preprocess/generico/microdados_enem_combinado/DADOS/PREPROCESSED_DATA.csv",
        sep=";",
        blocksize="64MB",
    )
except FileNotFoundError:
    print("Arquivo PREPROCESSED_DATA.csv não encontrado. Verifique o caminho.")
    exit()

# --- DEFINIÇÃO DAS COLUNAS ---
colunas_numericas = [
    "NOTA_GERAL",
    "NU_NOTA_CH",
    "NU_NOTA_CN",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
    "Q006",
]
colunas_categoricas = [
    "TP_COR_RACA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_SEXO",
    "TP_LINGUA",
    "TP_ESCOLA",
]

# --- ESTRUTURAS PARA MONTAR A TABELA FINAL ---
dados_finais = {}

mapeamento_geral = {
    "TP_COR_RACA": {
        0: "Não declarado",
        1: "Branca",
        2: "Preta",
        3: "Parda",
        4: "Amarela",
        5: "Indígena",
    },
    "TP_DEPENDENCIA_ADM_ESC": {
        1: "Federal",
        2: "Estadual",
        3: "Municipal",
        4: "Privada",
    },
    "TP_SEXO": {"M": "Masculino", "F": "Feminino"},
    "TP_LINGUA": {0: "Inglês", 1: "Espanhol"},
    "TP_ESCOLA": {1: "Não respondeu", 2: "Pública", 3: "Privada"},
}

# Mapeamento para nomes mais amigáveis das estatísticas
mapeamento_estatisticas = {
    "count": "Contagem",
    "mean": "Média",
    "std": "Desvio Padrão",
    "min": "Mínimo",
    "25%": "25º Percentil (Q1)",
    "50%": "Mediana (50º)",
    "75%": "75º Percentil (Q3)",
    "max": "Máximo",
}

nomes_das_metricas = ["Distribuição Geral", "Perfil 0", "Perfil 1", "Perfil 2"]

# --- PROCESSAMENTO DE COLUNAS NUMÉRICAS ---
print("Processando colunas numéricas...")
# Usa o método describe() para calcular todas as estatísticas de uma vez
desc_geral_dask = df[colunas_numericas].describe()


# ***** INÍCIO DA ALTERAÇÃO *****
# Para o groupby, usamos .apply() para garantir compatibilidade
def describe_on_group(group):
    """Aplica o .describe() do pandas a um grupo de dados."""
    return group.describe()


# Dask precisa saber a estrutura de saída da função .apply().
# Criamos um DataFrame "modelo" (meta) com a estrutura esperada.
meta_df = pd.DataFrame(
    {col: pd.Series(dtype=float) for col in colunas_numericas}
).describe()

# Aplicamos a função a cada grupo, passando as colunas numéricas
desc_por_categoria_dask = df.groupby("CLASSIFICACAO")[colunas_numericas].apply(
    describe_on_group, meta=meta_df
)
# ***** FIM DA ALTERAÇÃO *****

# Computa os resultados
desc_geral, desc_por_categoria = dask.compute(desc_geral_dask, desc_por_categoria_dask)


# Adiciona cada estatística calculada à estrutura de dados final
for col in colunas_numericas:
    for stat_key, stat_name in mapeamento_estatisticas.items():
        chave_coluna = (col, stat_name)

        # O índice de `desc_por_categoria` é um MultiIndex (CLASSIFICACAO, stat_key)
        # Por isso, precisamos verificar a existência da tupla completa
        dados_finais[chave_coluna] = [
            desc_geral.loc[stat_key, col],
            (
                desc_por_categoria.loc[(0, stat_key), col]
                if (0, stat_key) in desc_por_categoria.index
                else 0
            ),
            (
                desc_por_categoria.loc[(1, stat_key), col]
                if (1, stat_key) in desc_por_categoria.index
                else 0
            ),
            (
                desc_por_categoria.loc[(2, stat_key), col]
                if (2, stat_key) in desc_por_categoria.index
                else 0
            ),
        ]

# --- PROCESSAMENTO DE COLUNAS CATEGÓRICAS ---
print("Processando colunas categóricas...")
for col in colunas_categoricas:
    print(f"  - Analisando '{col}'...")
    # Dask computations
    contagens_agrupadas_dask = df.groupby("CLASSIFICACAO")[col].value_counts()
    totais_agrupados_dask = df.groupby("CLASSIFICACAO")[col].count()
    contagens_geral_dask = df[col].value_counts()
    total_geral_dask = df[col].count()

    contagens_agrupadas, totais_agrupados, contagens_geral, total_geral = dask.compute(
        contagens_agrupadas_dask,
        totais_agrupados_dask,
        contagens_geral_dask,
        total_geral_dask,
    )

    # Cálculos com os resultados do Dask (Pandas)
    percentuais_agrupados = (contagens_agrupadas / totais_agrupados) * 100
    percentuais_geral = (contagens_geral / total_geral) * 100

    tabela_percentuais = percentuais_agrupados.unstack(level="CLASSIFICACAO").fillna(0)
    tabela_percentuais.columns = [f"Perfil {c}" for c in tabela_percentuais.columns]
    tabela_percentuais["Distribuição Geral"] = percentuais_geral

    # Aplica a função de ajuste para cada coluna de perfil/distribuição
    for profile_col in tabela_percentuais.columns:
        tabela_percentuais[profile_col] = ajustar_porcentagens_para_100(
            tabela_percentuais[profile_col]
        )

    mapeamento_coluna = mapeamento_geral.get(col, {})
    for codigo, dados_linha in tabela_percentuais.iterrows():
        nome_subcategoria = mapeamento_coluna.get(codigo, str(codigo))
        chave_coluna = (col, nome_subcategoria)

        dados_finais[chave_coluna] = [
            dados_linha.get("Distribuição Geral", 0),
            dados_linha.get("Perfil 0", 0),
            dados_linha.get("Perfil 1", 0),
            dados_linha.get("Perfil 2", 0),
        ]

# --- MONTAGEM E EXPORTAÇÃO DA TABELA FINAL ---
print("Montando e salvando a tabela final...")

tabela_final = pd.DataFrame(dados_finais, index=nomes_das_metricas)

tabela_final.columns = pd.MultiIndex.from_tuples(
    tabela_final.columns, names=["Categoria", "Subcategoria"]
)

# Formatação final dos valores (agora mais genérica)
for cat in colunas_numericas:
    # Obtém todas as subcategorias para a categoria numérica atual (Média, Mediana, etc.)
    subcategorias = tabela_final[cat].columns
    for subcat in subcategorias:
        # Formata todas as subcategorias numéricas como float com 2 casas decimais
        # A 'Contagem' será formatada como float, o que é aceitável aqui.
        tabela_final[(cat, subcat)] = tabela_final[(cat, subcat)].map("{:,.2f}".format)

for cat in colunas_categoricas:
    subcategorias = tabela_final[cat].columns
    for subcat in subcategorias:
        tabela_final[(cat, subcat)] = tabela_final[(cat, subcat)].map("{:,.2f}%".format)

try:
    tabela_final.to_csv(
        "tabela_analise_descritiva_transposta.csv", sep=";", encoding="utf-8-sig"
    )
    print(
        "\nTabela transposta salva com sucesso em 'tabela_analise_descritiva_transposta.csv'"
    )
except Exception as e:
    print(f"\nOcorreu um erro ao salvar o arquivo: {e}")
