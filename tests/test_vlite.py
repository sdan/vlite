import unittest
import numpy as np
from vlite.main import VLite
from vlite.utils import load_file
import os

class TestVLite(unittest.TestCase):
    def setUp(self):
        self.vlite = VLite("test_vlite")
        self.corpus = load_file('data/gpt-4.pdf')

    def tearDown(self):
        if os.path.exists(self.vlite.get_index_file()):
            os.remove(self.vlite.get_index_file())

    def test_ingest(self):
        for text in self.corpus:
            self.vlite.ingest(text)
        self.assertEqual(len(self.vlite), len(self.corpus))

    def test_retrieve(self):
        for text in self.corpus:
            self.vlite.ingest(text)

        query = "What is the architecture of GPT-4?"
        top_k = 3
        results = self.vlite.retrieve(query, top_k=top_k, get_similarities=True)

        self.assertEqual(len(results["texts"]), top_k)
        self.assertEqual(len(results["metadata"]), top_k)
        self.assertEqual(len(results["scores"]), top_k)


if __name__ == '__main__':
    unittest.main()