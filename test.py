import unittest
import numpy as np
from main import VLite

class TestVLite(unittest.TestCase):

    def setUp(self):
        self.vlite = VLite(collection='test.pkl')
        self.data = ["7. DETERMINISTIC FRONTIER ANALYSIS\nProduction frontiers are often represented by distance, revenue, cost and/or pro\x0ct\nfunctions. These functions can sometimes be written in the form of regression\nmodels in which the explanatory variables are deterministic (i.e., not random).\nThe associated frontiers are known as deterministic frontiers. This topic explains\nhow to estimate the unknown parameters of so-called deterministic frontier mod-\nels (DFMs).\nc\rC.J. O'Donnell. All rights reserved. 1 / 32",
                    "8. STOCHASTIC FRONTIER ANALYSIS\nDistance, revenue, cost and pro\x0ct functions can always be written in the form\nof regression models with unobserved error terms representing statistical noise\nand di\x0berent types of ine\x0eciency. In practice, the noise components are almost\nalways assumed to be random variables (i.e., stochastic). The associated fron-\ntiers are known as stochastic frontiers. This topic explains how to estimate the\nunknown parameters of so-called stochastic frontier models (SFMs).\nc\rC.J. O'Donnell. All rights reserved. 1 / 24"]
        self.ids = ["1", "2"]
        self.metadata = [{"course": "Econometrics", "topic": "Deterministic Frontier Analysis"},
                        {"course": "Econometrics", "topic": "Stochastic Frontier Analysis"}]
        for id, data, metadata in zip(self.ids, self.data, self.metadata):
            self.vlite.memorize(data=data, id=id, metadata=metadata)

    def tearDown(self):
        del self.vlite

    def test_memorize(self):
        self.assertEqual(self.vlite.id_to_index[self.ids[0]], 0)
        self.assertEqual(self.vlite.metadata[self.ids[0]], self.metadata[0])
        self.assertTrue(np.array_equal(self.vlite.vectors[0], self.vlite.model.embed(self.data[0])))

    def test_remember_by_id(self):
        self.assertTrue(np.array_equal(self.vlite.remember(id=self.ids[1]), self.vlite.model.embed(self.data[1])))

    def test_remember_by_metadata(self):
        result_metadata = {"course": "Econometrics", "topic": "Deterministic Frontier Analysis"}
        self.assertTrue(np.array_equal(self.vlite.remember(metadata=result_metadata), self.vlite.model.embed(self.data[0])))

    def test_remember_top_5(self):
        data = "This is a new test query"
        results = self.vlite.remember(data=data)
        self.assertTrue(len(results), 5)
        for id, vector in results:
            self.assertIn(id, self.ids)
            self.assertTrue(np.array_equal(vector, self.vlite.model.embed(self.data[self.ids.index(id)])))


if __name__ == '__main__':
    unittest.main()
