import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

RAIZ = Path(__file__).resolve().parent


def verificar():
    fonte = json.loads((RAIZ / "fonte.json").read_text(encoding="utf-8"))
    for nome, esperado in fonte["sha256"].items():
        arquivo = RAIZ / "dados" / nome
        if not arquivo.exists():
            raise FileNotFoundError(f"Execute python dados.py para baixar {nome}.")
        if hashlib.sha256(arquivo.read_bytes()).hexdigest() != esperado:
            raise ValueError(f"Integridade divergente em {nome}; revise a origem.")


def baixar():
    fonte = json.loads((RAIZ / "fonte.json").read_text(encoding="utf-8"))
    with urlopen(fonte["url"], timeout=120) as resposta:
        conteudo = resposta.read()
    with ZipFile(io.BytesIO(conteudo)) as pacote:
        arquivos = {Path(n).name: pacote.read(n) for n in pacote.namelist() if not n.endswith("/")}
    for nome, esperado in fonte["sha256"].items():
        if hashlib.sha256(arquivos[nome]).hexdigest() != esperado:
            raise ValueError("A fonte mudou. Revisão manual necessária.")
    (RAIZ / "dados").mkdir(exist_ok=True)
    for nome in fonte["sha256"]:
        (RAIZ / "dados" / nome).write_bytes(arquivos[nome])
    verificar()
    print("Bases verificadas por SHA-256.")


if __name__ == "__main__":
    baixar()
