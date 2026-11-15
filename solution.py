import numpy as np
import pandas as pd


def calcular_regressao(arquivo_csv):
  df = pd.read_csv(arquivo_csv)

  area = df["area_util_m2"].to_list()
  quartos = df["quantidade_quartos"].to_list()
  preco = df["preco_venda"].to_list()

  uns = [1] * len(area)

  X = np.column_stack((uns, area, quartos))
  y = np.array(preco)

  X_T = X.T
  beta = np.linalg.inv(X_T @ X) @ X_T @ y

  return beta.tolist()


betas = calcular_regressao("imoveis_lar_perfeito.csv")
print(f"Lista retornada: {betas}")
print(f"Preco = {betas[0]:.2f} + {betas[1]:.2f}*Area + {betas[2]:.2f}*Quartos")
