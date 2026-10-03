import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dados import verificar
from modelagem import dividir, intervalo_lift, metricas_carteira, preparar, priorizar

RAIZ = Path(__file__).resolve().parent


def executar():
    verificar()
    original = pd.read_csv(RAIZ / "dados/Customer Churn.csv")
    x, y, grupos = preparar(original)
    treino, calibracao, teste = dividir(x, y, grupos)
    saida = RAIZ / "resultados"
    saida.mkdir(exist_ok=True)
    particao = pd.Series(index=x.index, dtype="object")
    for nome, indices in [("treino", treino), ("calibracao", calibracao), ("teste", teste)]:
        particao.loc[indices] = nome
    pd.DataFrame({"linha_fonte": x.index, "grupo_perfil": grupos, "particao": particao, "churn": y}).to_csv(saida / "particoes.csv", index=False)
    candidatos = {
        "logistica": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)), {"logisticregression__C": [.1, 1., 10.], "logisticregression__class_weight": [None, "balanced"]}),
        "boosting": (HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, early_stopping=False, random_state=42), {"max_leaf_nodes": [7, 15], "l2_regularization": [1., 10.]})
    }
    modelos, registros, grades = {}, [], []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=44)
    for nome, (modelo, parametros) in candidatos.items():
        busca = GridSearchCV(modelo, parametros, scoring="average_precision", cv=cv, n_jobs=1, error_score="raise")
        busca.fit(x.iloc[treino], y.iloc[treino], groups=grupos[treino])
        modelos[nome] = busca.best_estimator_
        registros.append({"modelo": nome, "ap_cv": float(busca.best_score_), "desvio_folds": float(busca.cv_results_["std_test_score"][busca.best_index_]), "parametros": json.dumps(busca.best_params_)})
        grades.append(pd.DataFrame(busca.cv_results_).assign(modelo=nome))
    comparacao = pd.DataFrame(registros).sort_values("ap_cv", ascending=False)
    escolhido = comparacao.iloc[0].modelo
    comparacao.to_csv(saida / "validacao_cruzada.csv", index=False)
    pd.concat(grades, ignore_index=True).to_csv(saida / "busca_completa.csv", index=False)
    selecao = {"modelo": escolhido, "parametros": json.loads(comparacao.iloc[0].parametros), "criterio": "Maior average precision em CV agrupada no treino", "capacidade_principal": .1, "calibracao": "sigmoid em conjunto separado"}
    (saida / "selecao.json").write_text(json.dumps(selecao, indent=2))
    modelo = CalibratedClassifierCV(FrozenEstimator(modelos[escolhido]), method="sigmoid")
    modelo.fit(x.iloc[calibracao], y.iloc[calibracao])
    probabilidades = modelo.predict_proba(x.iloc[teste])[:, 1]
    predicoes = {"prior_treino": np.full(len(teste), y.iloc[treino].mean()), **{n: m.predict_proba(x.iloc[teste])[:, 1] for n, m in modelos.items()}, "selecionado_calibrado": probabilidades}
    resultados = []
    for nome, p in predicoes.items():
        resultados.append({"modelo": nome, "average_precision": average_precision_score(y.iloc[teste], p), "roc_auc": roc_auc_score(y.iloc[teste], p), "brier": brier_score_loss(y.iloc[teste], p), **metricas_carteira(y.iloc[teste], p)})
    pd.DataFrame(resultados).to_csv(saida / "avaliacao_teste.csv", index=False)
    carteira = metricas_carteira(y.iloc[teste], probabilidades)
    intervalo = intervalo_lift(y.iloc[teste], probabilidades, grupos[teste])
    selecionados = priorizar(probabilidades)
    fila = pd.DataFrame({"linha_fonte": teste, "churn_observado": y.iloc[teste].to_numpy(), "probabilidade": probabilidades, "priorizado": False})
    fila.loc[selecionados, "priorizado"] = True
    fila.sort_values("probabilidade", ascending=False).to_csv(saida / "ranking_teste.csv", index=False)
    pd.DataFrame([{"capacidade": f, **metricas_carteira(y.iloc[teste], probabilidades, f)} for f in [.05, .1, .2, .3]]).to_csv(saida / "capacidade_contato.csv", index=False)
    importancia = permutation_importance(modelo, x.iloc[teste], y.iloc[teste], scoring="average_precision", n_repeats=20, random_state=42, n_jobs=1)
    pd.DataFrame({"variavel": x.columns, "queda_ap_media": importancia.importances_mean, "desvio_repeticoes": importancia.importances_std}).sort_values("queda_ap_media", ascending=False).to_csv(saida / "importancia.csv", index=False)
    faixas = pd.cut(probabilidades, np.linspace(0, 1, 11), include_lowest=True)
    pd.DataFrame({"faixa": faixas, "p": probabilidades, "y": y.iloc[teste].to_numpy()}).groupby("faixa", observed=True).agg(n=("y", "size"), probabilidade_media=("p", "mean"), fracao_churn=("y", "mean")).to_csv(saida / "calibracao.csv")
    resumo = {"clientes": len(y), "churn_total": int(y.sum()), "prevalencia_total": float(y.mean()), "perfis_distintos": int(len(np.unique(grupos))), "linhas_por_particao": particao.value_counts().to_dict(), "prevalencia_por_particao": pd.DataFrame({"particao": particao, "y": y}).groupby("particao").y.mean().to_dict(), "modelo": escolhido, "ap_teste": float(average_precision_score(y.iloc[teste], probabilidades)), "brier_teste": float(brier_score_loss(y.iloc[teste], probabilidades)), "carteira_10pct": carteira, "ic95_lift_cluster": intervalo.tolist(), "campos_excluidos": ["Status", "Customer Value", "Age", "Age Group"], "semente": 42}
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    joblib.dump({"modelo": modelo, "colunas": list(x.columns)}, saida / "modelo.joblib")
    x.iloc[teste[:5]].to_csv(RAIZ / "exemplo_entrada.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for nome, p in predicoes.items():
        precisao, recall, _ = precision_recall_curve(y.iloc[teste], p)
        eixos[0].plot(recall, precisao, label=nome)
    eixos[0].set(xlabel="Recall", ylabel="Precisão", title="Discriminação no teste reservado")
    eixos[0].legend(fontsize=8)
    observado, previsto = calibration_curve(y.iloc[teste], probabilidades, n_bins=8, strategy="quantile")
    eixos[1].plot(previsto, observado, "o-", color="#236e94")
    eixos[1].plot([0, 1], [0, 1], "--", color="#777777")
    eixos[1].set(xlabel="Probabilidade média prevista", ylabel="Fração de cancelamentos", title="Confiabilidade das probabilidades", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(saida / "diagnostico.png", dpi=180)
    plt.close(fig)
    texto = f'''# Retenção: resultados e decisão

O estudo usa {len(y):,} clientes, com {y.mean():.1%} de cancelamento na base. O modelo **{escolhido}** foi escolhido somente pela validação cruzada do treino; uma amostra separada foi usada para calibrar as probabilidades.

Na amostra de teste, priorizar os 10% com maior risco selecionou **{carteira['contatos']} clientes**, dos quais **{carteira['cancelamentos_encontrados']} cancelaram**. Precisão = **{carteira['precisao']:.1%}**, recall = **{carteira['recall']:.1%}** e lift sobre a prevalência do teste = **{carteira['lift']:.2f} vezes**. O IC bootstrap exploratório de 95% do lift, por perfil, foi [{intervalo[0]:.2f}, {intervalo[1]:.2f}]. Average precision = {resumo['ap_teste']:.3f}; Brier = {resumo['brier_teste']:.3f}.

## Decisão apoiada

A fila permite priorizar uma capacidade limitada de contato. Lift mede concentração de futuros cancelamentos identificados; não mede clientes salvos. Não há campanha observada, custo de contato, margem ou resposta a incentivo nesta base. Nenhuma receita recuperada foi calculada.

## Próximo experimento

Sortear clientes elegíveis entre ação de retenção e controle, antes do contato. Definir previamente horizonte de cancelamento, orçamento e métricas de efeito adverso. Medir diferença de cancelamento entre grupos e custo por retenção incremental, em vez de atribuir ao modelo toda permanência posterior. Risco elevado não significa maior chance de responder à ação.

## Limitações

A base é histórica, de uma operadora iraniana. Há janela de nove meses de atributos e três meses de planejamento, mas não datas individuais para teste fora do tempo. Perfis observáveis idênticos permanecem no mesmo conjunto, com repetições preservadas porque representam clientes. A unidade do bootstrap é o perfil; a incerteza do treinamento não está incluída. Há poucos dados para transportar desempenho a outra operadora.

Status, Customer Value e idade não entram no modelo. Status pode ser informativo, mas foi excluído por cautela sobre disponibilidade operacional; a fonte diz que todos os preditores pertencem à janela inicial. Não se afirma que há vazamento comprovado nesse campo. A fórmula de Customer Value é insuficiente para tratá-lo como receita. Charge Amount é categoria ordinal, não dinheiro.
'''
    (saida / "relatorio.md").write_text(texto, encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    executar()
