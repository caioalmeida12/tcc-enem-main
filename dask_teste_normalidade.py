import dask.dataframe as dd
from scipy.stats import anderson
import time

# --- CONFIGURAÇÕES ---

# 1. Caminho para o arquivo de dados pré-processado.
DATA_PATH = (
    "./preprocess/generico/microdados_enem_combinado/DADOS/PREPROCESSED_DATA.csv"
)

# 2. Caminho para salvar o relatório de normalidade.
OUTPUT_FILE = "relatorio_teste_de_normalidade.txt"

# 3. Lista de colunas numéricas para verificar a normalidade.
COLUNAS_NUMERICAS = [
    "NOTA_GERAL",
    "NU_NOTA_CH",
    "NU_NOTA_CN",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
    "Q006",  # Renda
]

# 4. Tamanho da amostra a ser extraída para realizar os testes.
# Testes de normalidade são computacionalmente intensivos.
# Uma amostra aleatória grande é suficiente para uma boa inferência.
SAMPLE_SIZE = 6000000

# 5. Nível de significância para o teste.
ALPHA = 0.05


def interpretar_anderson_darling(resultado_anderson, alpha=0.05):
    """
    Interpreta o resultado do teste de Anderson-Darling.

    A hipótese nula (H0) é que os dados seguem uma distribuição normal.
    Se a estatística do teste for MAIOR que o valor crítico, rejeitamos H0.

    Args:
        resultado_anderson: O objeto de resultado retornado por scipy.stats.anderson.
        alpha: O nível de significância (ex: 0.05 para 5%).

    Returns:
        Uma string com a interpretação do resultado.
    """
    # Mapeia o alpha para o índice correspondente no array de valores críticos
    niveis_significancia = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
    if alpha not in niveis_significancia:
        raise ValueError(f"Nível de significância '{alpha}' não suportado pelo teste.")

    indice_critico = niveis_significancia[alpha]
    valor_critico = resultado_anderson.critical_values[indice_critico]
    estatistica = resultado_anderson.statistic

    if estatistica > valor_critico:
        conclusao = "REJEITADA. Os dados NÃO parecem seguir uma distribuição normal."
    else:
        conclusao = (
            "NÃO REJEITADA. Não há evidências para descartar a normalidade dos dados."
        )

    return (
        f"Estatística do Teste (A²): {estatistica:.4f}\n"
        f"  - Valor Crítico ({int(alpha*100)}%): {valor_critico:.4f}\n"
        f"  - Conclusão: A hipótese nula de normalidade é {conclusao}"
    )


def main():
    """
    Função principal para carregar os dados, testar a normalidade e salvar os resultados.
    """
    start_time = time.time()
    print("Iniciando o teste de normalidade com Dask...")

    try:
        df = dd.read_csv(DATA_PATH, sep=";", blocksize="64MB")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de dados não encontrado em '{DATA_PATH}'.")
        return

    # Calcula a fração de amostragem necessária
    total_rows = len(df)
    if total_rows == 0:
        print("ERRO: O arquivo de dados está vazio.")
        return

    sample_fraction = min(SAMPLE_SIZE / total_rows, 1.0)

    print(f"Extraindo uma amostra de {SAMPLE_SIZE} linhas ({sample_fraction:.2%})...")
    # Extrai uma amostra aleatória e a carrega na memória como um DataFrame do Pandas
    # Isso é necessário porque os testes do SciPy não operam diretamente em Dask DataFrames.
    df_sample = df.sample(frac=sample_fraction, random_state=42).compute()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        RELATÓRIO DE TESTE DE NORMALIDADE (ANDERSON-DARLING)\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Hipótese Nula (H0): A amostra de dados provém de uma população com distribuição normal.\n"
        )
        f.write(f"Nível de Significância (alpha): {ALPHA}\n")
        f.write("-" * 80 + "\n\n")

        print("Realizando os testes de normalidade...")
        for col in COLUNAS_NUMERICAS:
            f.write(f"Variável: {col}\n")
            f.write("-" * (len(col) + 10) + "\n")

            try:
                # Remove valores nulos ou infinitos antes do teste
                dados_validos = df_sample[col].dropna()
                if len(dados_validos) < 10:
                    f.write("  - Resultado: Dados insuficientes para o teste.\n\n")
                    continue

                # Realiza o teste de Anderson-Darling
                resultado = anderson(dados_validos)
                interpretacao = interpretar_anderson_darling(resultado, alpha=ALPHA)
                f.write(f"  - {interpretacao}\n\n")

            except Exception as e:
                f.write(f"  - ERRO ao processar a variável: {e}\n\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nAnálise de normalidade concluída em {total_time:.2f} segundos.")
    print(f"Resultados salvos em '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
