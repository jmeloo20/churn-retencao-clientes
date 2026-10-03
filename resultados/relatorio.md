# Retenção: resultados e decisão

O estudo usa 3,150 clientes, com 15.7% de cancelamento na base. O modelo **boosting** foi escolhido somente pela validação cruzada do treino; uma amostra separada foi usada para calibrar as probabilidades.

Na amostra de teste, priorizar os 10% com maior risco selecionou **64 clientes**, dos quais **54 cancelaram**. Precisão = **84.4%**, recall = **54.5%** e lift sobre a prevalência do teste = **5.38 vezes**. O IC bootstrap exploratório de 95% do lift, por perfil, foi [4.22, 7.01]. Average precision = 0.836; Brier = 0.055.

## Decisão apoiada

A fila permite priorizar uma capacidade limitada de contato. Lift mede concentração de futuros cancelamentos identificados; não mede clientes salvos. Não há campanha observada, custo de contato, margem ou resposta a incentivo nesta base. Nenhuma receita recuperada foi calculada.

## Próximo experimento

Sortear clientes elegíveis entre ação de retenção e controle, antes do contato. Definir previamente horizonte de cancelamento, orçamento e métricas de efeito adverso. Medir diferença de cancelamento entre grupos e custo por retenção incremental, em vez de atribuir ao modelo toda permanência posterior. Risco elevado não significa maior chance de responder à ação.

## Limitações

A base é histórica, de uma operadora iraniana. Há janela de nove meses de atributos e três meses de planejamento, mas não datas individuais para teste fora do tempo. Perfis observáveis idênticos permanecem no mesmo conjunto, com repetições preservadas porque representam clientes. A unidade do bootstrap é o perfil; a incerteza do treinamento não está incluída. Há poucos dados para transportar desempenho a outra operadora.

Status, Customer Value e idade não entram no modelo. Status pode ser informativo, mas foi excluído por cautela sobre disponibilidade operacional; a fonte diz que todos os preditores pertencem à janela inicial. Não se afirma que há vazamento comprovado nesse campo. A fórmula de Customer Value é insuficiente para tratá-lo como receita. Charge Amount é categoria ordinal, não dinheiro.
