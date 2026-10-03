import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

NOMES = {"Call Failure": "falhas_chamada", "Complains": "reclamacao", "Subscription Length": "meses_assinatura", "Charge Amount": "faixa_cobranca", "Seconds of Use": "segundos_uso", "Frequency of use": "numero_chamadas", "Frequency of SMS": "numero_sms", "Distinct Called Numbers": "numeros_distintos", "Tariff Plan": "plano"}


def preparar(original):
    dados = original.copy()
    dados.columns = [" ".join(c.split()) for c in dados.columns]
    if not (set(NOMES) | {"Churn"}).issubset(dados.columns):
        raise ValueError("Colunas obrigatórias ausentes.")
    x = dados[list(NOMES)].rename(columns=NOMES).astype(float)
    y = dados.Churn.astype(int)
    if not np.isfinite(x.to_numpy()).all() or (x < 0).any().any() or not y.isin([0, 1]).all():
        raise ValueError("Dados inválidos.")
    grupos = pd.factorize(pd.MultiIndex.from_frame(x), sort=True)[0]
    return x, y, grupos


def dividir(x, y, grupos):
    desenvolvimento, teste = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, grupos))
    treino_local, calibracao_local = next(StratifiedGroupKFold(4, shuffle=True, random_state=43).split(x.iloc[desenvolvimento], y.iloc[desenvolvimento], grupos[desenvolvimento]))
    return desenvolvimento[treino_local], desenvolvimento[calibracao_local], teste


def priorizar(probabilidades, fracao=.1):
    probabilidades = np.asarray(probabilidades, dtype=float)
    if not 0 < fracao <= 1 or not len(probabilidades) or not np.isfinite(probabilidades).all():
        raise ValueError("Capacidade ou probabilidades inválidas.")
    desempate = np.random.default_rng(42).random(len(probabilidades))
    ordem = np.lexsort((desempate, -probabilidades))
    return ordem[:max(1, int(np.ceil(len(probabilidades) * fracao)))]


def metricas_carteira(real, probabilidades, fracao=.1):
    real = np.asarray(real)
    indices = priorizar(probabilidades, fracao)
    encontrados = int(real[indices].sum())
    prevalencia = float(real.mean())
    precisao = float(real[indices].mean())
    return {"contatos": len(indices), "cancelamentos_encontrados": encontrados, "precisao": precisao, "recall": encontrados / real.sum() if real.sum() else 0., "lift": precisao / prevalencia if prevalencia else 0.}


def intervalo_lift(real, probabilidades, grupos, repeticoes=2000):
    real, probabilidades, grupos = map(np.asarray, [real, probabilidades, grupos])
    unicos = np.unique(grupos)
    mapa = {g: np.flatnonzero(grupos == g) for g in unicos}
    rng = np.random.default_rng(42)
    resultados = []
    for _ in range(repeticoes):
        indices = np.concatenate([mapa[g] for g in rng.choice(unicos, len(unicos), replace=True)])
        if real[indices].sum():
            resultados.append(metricas_carteira(real[indices], probabilidades[indices])["lift"])
    return np.quantile(resultados, [.025, .975])
