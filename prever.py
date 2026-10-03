import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelagem import priorizar

RAIZ = Path(__file__).resolve().parent


def executar():
    parser = argparse.ArgumentParser(description="Pontuar clientes e selecionar uma fila de contato.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("saida", type=Path)
    parser.add_argument("--capacidade", type=float, default=.1)
    args = parser.parse_args()
    pacote = joblib.load(RAIZ / "resultados/modelo.joblib")
    x = pd.read_csv(args.entrada)
    if set(x.columns) != set(pacote["colunas"]):
        raise ValueError(f"Colunas esperadas: {pacote['colunas']}")
    x = x[pacote["colunas"]].astype(float)
    if not np.isfinite(x.to_numpy()).all() or (x < 0).any().any():
        raise ValueError("Valores de entrada inválidos.")
    p = pacote["modelo"].predict_proba(x)[:, 1]
    resultado = pd.DataFrame({"linha_entrada": np.arange(len(x)), "probabilidade_churn": p, "priorizado": False})
    resultado.loc[priorizar(p, args.capacidade), "priorizado"] = True
    resultado.sort_values("probabilidade_churn", ascending=False).to_csv(args.saida, index=False)
    print(f"{len(x)} clientes pontuados; capacidade aplicada sem usar desfechos.")


if __name__ == "__main__":
    executar()
