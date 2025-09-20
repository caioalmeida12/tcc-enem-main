import dask.dataframe as dd
import dask
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
import time
import numpy as np  # Importar numpy para cálculos de effect size

# --- CONFIGURAÇÕES ---
DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/DADOS/PREPROCESSED_DATA.csv"
)
OUTPUT_FILE = "resultados_testes_estatisticos.txt"
GROUPING_VARIABLE = "CLASSIFICACAO"

# --- DEFINIÇÃO DAS COLUNAS PARA ANÁLISE ---
# Variáveis para o teste de Qui-Quadrado (associação entre categóricas)
colunas_para_qui_quadrado = [
    "TP_COR_RACA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_SEXO",
    "TP_LINGUA",
    "TP_ESCOLA",
]

# Variáveis para o teste de Kruskal-Wallis (comparação de medianas entre grupos)
colunas_para_kruskal_wallis = [
    "NOTA_GERAL",
    "NU_NOTA_CH",
    "NU_NOTA_CN",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
    "Q006",  # Renda
]


def formatar_resultados(p_valor, alpha=0.05):
    """Formata a interpretação do p-valor."""
    if p_valor < alpha:
        return f"p-valor = {p_valor:.5f} -> Significativo (Rejeita H0)"
    else:
        return f"p-valor = {p_valor:.5f} -> Não Significativo (Falha em rejeitar H0)"


# --- Funções para cálculo de Effect Size ---
def cramers_v(chi2, n, r, k):
    """
    Calcula o V de Cramér.
    chi2: estatística qui-quadrado
    n: tamanho total da amostra
    r: número de linhas da tabela de contingência
    k: número de colunas da tabela de contingência
    """
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    return v


def eta_squared_h(H_stat, N, k):
    """
    Calcula Eta Quadrado H para Kruskal-Wallis.
    H_stat: estatística H do Kruskal-Wallis
    N: tamanho total da amostra
    k: número de grupos
    """
    # Esta é uma aproximação e pode não ser o eta^2 exato para Kruskal-Wallis
    # Uma forma mais robusta envolveria a soma dos ranks ao quadrado,
    # mas para simplicidade e considerando as limitações do H,
    # esta fórmula baseada em L. B. Osborne (2008) ou similar é comumente usada para estimar.
    # Alternativamente, para uma medida mais direta, pode-se calcular a diferença de medianas.
    return (H_stat - k + 1) / (N - k) if (N - k) > 0 else np.nan


def main():
    """
    Função principal para carregar os dados, executar os testes estatísticos
    e salvar os resultados.
    """
    start_time = time.time()
    print("Iniciando a análise estatística com Dask...")

    try:
        df = dd.read_csv(DATA_PATH, sep=";", blocksize="64MB")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de dados não encontrado em '{DATA_PATH}'.")
        return

    # Abre o arquivo de saída para escrever os resultados
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("         ANÁLISE ESTATÍSTICA AUTOMATIZADA COM DASK\n")
        f.write("=" * 80 + "\n\n")

        # --- 1. TESTE QUI-QUADRADO DE ASSOCIAÇÃO ---
        f.write("-" * 80 + "\n")
        f.write(
            f" 1. Teste Qui-Quadrado (χ²): Associação entre Features Categóricas e '{GROUPING_VARIABLE}'\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            "Hipótese Nula (H0): Não há associação entre a variável e a classificação (são independentes).\n"
            "Medida de Magnitude de Efeito: V de Cramér (0.1: pequeno, 0.3: médio, 0.5: grande).\n\n"
        )

        print("Executando Testes Qui-Quadrado...")
        for col in colunas_para_qui_quadrado:
            try:
                # Dask não tem crosstab, então construímos a tabela de contingência manualmente
                contingency_table_dask = (
                    df.groupby([GROUPING_VARIABLE, col]).size().compute()
                )
                contingency_table = contingency_table_dask.unstack(
                    level=GROUPING_VARIABLE
                ).fillna(0)

                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                    resultado = "Dados insuficientes para o teste (menos de 2x2)."
                    f.write(f"- {col}:\n   {resultado}\n\n")
                else:
                    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
                    resultado_p = formatar_resultados(p_val)

                    # --- Adição do V de Cramér ---
                    n_total = contingency_table.sum().sum()
                    r, k = contingency_table.shape
                    v_cramer = cramers_v(chi2_stat, n_total, r, k)

                    f.write(f"- {col}:\n")
                    f.write(f"   Estatística χ² = {chi2_stat:.2f}, {resultado_p}\n")
                    f.write(f"   V de Cramér = {v_cramer:.3f} (Magnitude do Efeito)\n")
                    # --- Sugestão de inspeção visual ---
                    f.write(
                        f"   Tabela de Contingência (amostra):\n{contingency_table.head().to_string()}\n\n"
                    )

                print(f"   - Teste para '{col}' concluído.")

            except Exception as e:
                f.write(f"- {col}:\n   ERRO ao processar: {e}\n\n")
                print(f"   - ERRO no teste para '{col}'.")

        # --- 2. TESTE DE KRUSKAL-WALLIS ---
        f.write("\n" + "-" * 80 + "\n")
        f.write(
            f" 2. Teste de Kruskal-Wallis: Comparação de Distribuições Numéricas por '{GROUPING_VARIABLE}'\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            "Hipótese Nula (H0): A distribuição da variável é a mesma em todos os grupos de classificação.\n"
            "Medida de Magnitude de Efeito: Diferença de Medianas e Eta Quadrado H.\n\n"
        )

        print("\nExecutando Testes de Kruskal-Wallis...")
        # Obtém as classes únicas da variável de agrupamento
        classes = df[GROUPING_VARIABLE].unique().compute().tolist()
        classes.sort()

        for col in colunas_para_kruskal_wallis:
            try:
                # Cria uma lista de Dask Series, uma para cada grupo/classe
                grupos_dask = [
                    df[df[GROUPING_VARIABLE] == cls][col].dropna() for cls in classes
                ]

                # Executa o .compute() para todos os grupos de uma vez (mais eficiente)
                grupos_computados = dask.compute(*grupos_dask)

                # Valida se há dados suficientes nos grupos
                grupos_validos = [g for g in grupos_computados if len(g) > 1]

                if len(grupos_validos) < 2:
                    resultado = "Dados insuficientes (menos de 2 grupos com dados)."
                    f.write(f"- {col}:\n   {resultado}\n\n")
                else:
                    h_stat, p_val = kruskal(*grupos_validos)
                    resultado_p = formatar_resultados(p_val)

                    f.write(f"- {col}:\n")
                    f.write(f"   Estatística H = {h_stat:.2f}, {resultado_p}\n")

                    # --- Adição de Medidas de Magnitude de Efeito (diferença de medianas e eta^2) ---
                    f.write("   Medianas por Grupo:\n")
                    total_n = 0
                    for i, cls in enumerate(classes):
                        if (
                            len(grupos_computados[i]) > 1
                        ):  # Verifica se o grupo tem dados válidos
                            median_val = np.median(grupos_computados[i])
                            count_val = len(grupos_computados[i])
                            f.write(
                                f"     {GROUPING_VARIABLE} '{cls}': Mediana = {median_val:.2f}, N = {count_val}\n"
                            )
                            total_n += count_val

                    if (
                        total_n > 0
                    ):  # Para evitar divisão por zero se todos os grupos forem vazios
                        eta2_h = eta_squared_h(h_stat, total_n, len(grupos_validos))
                        f.write(
                            f"   Eta Quadrado H = {eta2_h:.3f} (Magnitude do Efeito)\n"
                        )

                    # --- Sugestão de inspeção visual (Exemplo de dispersão, se aplicável) ---
                    f.write(
                        f"   (Recomendado: Visualizar boxplots ou distribuições para entender as diferenças)\n\n"
                    )

                print(f"   - Teste para '{col}' concluído.")

            except Exception as e:
                f.write(f"- {col}:\n   ERRO ao processar: {e}\n\n")
                print(f"   - ERRO no teste para '{col}'.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nAnálise concluída em {total_time:.2f} segundos.")
    print(f"Resultados salvos em '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
