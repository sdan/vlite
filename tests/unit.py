import unittest
import numpy as np
from vlite.main import VLite
import os
from vlite.utils import process_pdf
import cProfile
from pstats import Stats
import matplotlib.pyplot as plt

class TestVLite(unittest.TestCase):
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
        # remove the file
        if os.path.exists('vlite-unit.npz'):
            print("[+] Removing vlite.npz")
            os.remove('vlite.npz')

    def test_add(self):
        with cProfile.Profile() as pr:
            print("[+] Adding text to the collection...")
            # print("Corpus printed below:")
            # print(self.corpus)  
            # corpus is a list of dictionaries and add expects a dictionary with keys 'text', 'id', and 'metadata'
            for doc in self.corpus:
                self.vlite.add(doc)
        stats = Stats(pr)
        stats.strip_dirs().sort_stats("time").print_stats()

    def test_retrieve(self):
        with cProfile.Profile() as pr:
            for query in self.queries:
                _, top_sims = self.vlite.retrieve(query)
        stats = Stats(pr)
        stats.strip_dirs().sort_stats("time").print_stats()

if __name__ == '__main__':
    unittest.main()
