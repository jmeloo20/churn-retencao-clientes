import sqlite3
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent


def executar():
    dados = pd.read_csv(RAIZ / "resultados/particoes.csv")
    with sqlite3.connect(":memory:") as conexao:
        dados.to_sql("clientes", conexao, index=False)
        resultado = pd.read_sql_query((RAIZ / "sql/particoes.sql").read_text(), conexao)
    if resultado.clientes.sum() != len(dados) or resultado.cancelamentos.sum() != dados.churn.sum():
        raise ValueError("Divergência de totais em SQL.")
    if resultado.perfis.sum() != dados.grupo_perfil.nunique():
        raise ValueError("Perfis atravessam partições.")
    resultado.to_csv(RAIZ / "resultados/auditoria_sql.csv", index=False)
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    executar()
