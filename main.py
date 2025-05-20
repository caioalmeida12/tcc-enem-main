import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Caminho do arquivo pré-processado
CAMINHO_DADOS = (
    "./preprocess/generico/microdados_enem_combinado/PREPROCESS/PREPROCESSED_DATA.csv"
)

# Variáveis globais para os conjuntos de treino/teste
X_train = X_test = y_train = y_test = None


def carregar_dados(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()
    return pd.read_csv(caminho, delimiter=";", encoding="utf-8")


def preparar_dados(df: pd.DataFrame, coluna_alvo: str = "NOTA_GERAL_SEM_REDACAO"):
    global X_train, X_test, y_train, y_test

    if df.empty:
        print("DataFrame está vazio. Abortando preparação de dados.")
        return

    y = df[coluna_alvo]
    X = df.drop(columns=[col for col in df.columns if "NOTA" in col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def treinar_regressao_linear():
    if X_train is None:
        print("Os dados não foram preparados.")
        return

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    avaliar_modelo(y_test, y_pred, "Regressão Linear")


def treinar_svr():
    if X_train is None:
        print("Os dados não foram preparados.")
        return

    modelo = SVR()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    avaliar_modelo(y_test, y_pred, "SVR")


def treinar_rfr():
    if X_train is None:
        print("Os dados não foram preparados.")
        return

    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    avaliar_modelo(y_test, y_pred, "Random Forest Regressor")


def avaliar_modelo(y_true, y_pred, titulo):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Resultado do modelo {titulo}:")
    print(f"R²: {r2:.4f}")
    print(f"Erro Médio Absoluto (MAE): {mae:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color="red", linestyle="--")
    plt.xlabel("Nota Real")
    plt.ylabel("Nota Predita")
    plt.title(f"{titulo}: Nota Real vs Predita")
    plt.grid(True)
    plt.show()


def main():
    df = carregar_dados(CAMINHO_DADOS)
    preparar_dados(df)
    # treinar_regressao_linear()
    # treinar_svr()
    treinar_rfr()


if __name__ == "__main__":
    main()
