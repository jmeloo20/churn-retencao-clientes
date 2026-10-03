import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelagem import dividir, metricas_carteira, preparar, priorizar


class TestChurn(unittest.TestCase):
    def test_capacidade_e_ranking(self):
        selecionados = priorizar([.2, .9, .4, .8], .5)
        self.assertEqual(set(selecionados), {1, 3})
        resultado = metricas_carteira([0, 1, 0, 1], [.2, .9, .4, .8], .5)
        self.assertEqual(resultado["lift"], 2)
        self.assertEqual(resultado["recall"], 1)

    def test_desempate_reprodutivel(self):
        np.testing.assert_array_equal(priorizar([.5] * 100), priorizar([.5] * 100))
        self.assertEqual(len(priorizar([.5] * 101)), 11)

    def test_rejeita_capacidade_invalida(self):
        with self.assertRaises(ValueError):
            priorizar([.2], 1.1)

    def test_grupos_nao_atravessam_particoes(self):
        original = pd.read_csv(Path(__file__).resolve().parents[1] / "dados/Customer Churn.csv")
        x, y, grupos = preparar(original)
        partes = dividir(x, y, grupos)
        self.assertEqual(len(np.unique(np.concatenate(partes))), len(original))
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(set(grupos[partes[i]]) & set(grupos[partes[j]]))
        for indices in partes:
            self.assertEqual(set(y.iloc[indices]), {0, 1})
        self.assertFalse({"Status", "Customer Value", "Churn", "Age", "Age Group"} & set(x.columns))


if __name__ == "__main__":
    unittest.main()
