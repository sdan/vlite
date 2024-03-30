import unittest
import numpy as np
from vlite.main import VLite
import os
from vlite.utils import process_pdf
import cProfile
from pstats import Stats
import matplotlib.pyplot as plt
import time

class TestVLite(unittest.TestCase):
    test_times = {}

    def setUp(self):
        self.queries = [
            "What is the architecture of GPT-4?",
            "How does GPT-4 handle contextual understanding?",
            "What are the key improvements in GPT-4 over GPT-3?",
            "How many parameters does GPT-4 have?",
            "What are the limitations of GPT-4?",
            "What datasets were used to train GPT-4?",
            "How does GPT-4 handle longer context?",
            "What is the computational requirement for training GPT-4?",
            "What techniques were used to train GPT-4?",
            "What is the impact of GPT-4 on natural language processing?",
            "What are the use cases demonstrated in the GPT-4 paper?",
            "What are the evaluation metrics used in GPT-4's paper?",
            "What kind of ethical considerations are discussed in the GPT-4 paper?",
            "How does the GPT-4 handle tokenization?",
            "What are the novel contributions of the GPT-4 model?"
        ]
        self.corpus = process_pdf('data/gpt-4.pdf')
        self.vlite = VLite("vlite-unit")

    def tearDown(self):
        # Remove the file
        if os.path.exists('vlite-unit.npz'):
            print("[+] Removing vlite.npz")
            os.remove('vlite-unit.npz')

    def test_add(self):
        start_time = time.time()
        print("[+] Adding text to the collection...")
        for doc in self.corpus:
            self.vlite.add(doc)
        end_time = time.time()
        TestVLite.test_times["add"] = end_time - start_time

    def test_retrieve(self):
        start_time = time.time()
        for query in self.queries:
            _, top_sims = self.vlite.retrieve(query)
        end_time = time.time()
        TestVLite.test_times["retrieve"] = end_time - start_time

    @classmethod
    def tearDownClass(cls):
        print("\nTest times:")
        for test_name, test_time in cls.test_times.items():
            print(f"{test_name}: {test_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main(verbosity=2)