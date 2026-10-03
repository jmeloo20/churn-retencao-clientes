SELECT particao,
       COUNT(*) AS clientes,
       COUNT(DISTINCT grupo_perfil) AS perfis,
       SUM(churn) AS cancelamentos,
       AVG(churn) AS prevalencia
FROM clientes
GROUP BY particao
ORDER BY particao;
