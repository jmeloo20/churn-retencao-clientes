import csv
import json
from pathlib import Path


def executar():
    raiz = Path(__file__).resolve().parent
    resultados = raiz / "resultados"
    resumo = json.loads((resultados / "resumo.json").read_text(encoding="utf-8"))
    with (resultados / "ranking_teste.csv").open(encoding="utf-8", newline="") as arquivo:
        clientes = [
            {"id": int(linha["linha_fonte"]), "risco": float(linha["probabilidade"]), "observado": int(linha["churn_observado"])}
            for linha in csv.DictReader(arquivo)
        ]
    destino = raiz / "painel" / "dados.js"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text("const estudo = " + json.dumps({"resumo": resumo, "clientes": clientes}, ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")
    print(f"Painel atualizado: {len(clientes)} registros de teste")


if __name__ == "__main__":
    executar()
