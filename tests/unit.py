import unittest
import numpy as np
from vlite import VLite
import os
from vlite.utils import load_file
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
        self.corpus = load_file('test-data/gpt-4.pdf')

        self.vlite = VLite(DEBUG=True)

    def tearDown(self):
        # remove the file
        if os.path.exists('vlite.pkl'):
            print("[+] Removing vlite.pkl")
            os.remove('vlite.pkl')

    def test_add_vector(self):
        with cProfile.Profile() as pr:
            self.vlite.add_vector(np.random.rand(5, 384))
            self.assertEqual(self.vlite.vectors.shape[0], 5) # was initially empty
            self.vlite.add_vector(np.random.rand(4, 384))
            self.assertEqual(self.vlite.vectors.shape[0], 9) # add 4 more, s/b 9
        #stats = Stats(pr)
        #stats.strip_dirs().sort_stats("time").print_stats()
        
    def test_get_similar_vectors(self):
        with cProfile.Profile() as pr:
            self.vlite.add_vector(np.random.rand(7, 384))
            self.vlite.add_vector(np.random.rand(4, 384))
            indices, sims = self.vlite.get_similar_vectors(np.random.rand(1, 384))
            self.assertEqual(len(indices), 5)
            for index in indices:
                self.assertEqual(type(index), np.int64)
                self.assertTrue(index >= 0)
                self.assertTrue(index <= 11)
            self.assertEqual(len(sims), 5)
            for sim in sims:
                self.assertEqual(type(sim), np.float64)
                self.assertTrue(sim >= -1.0)
                self.assertTrue(sim <= 1.0)
            
        #stats = Stats(pr)
        #stats.strip_dirs().sort_stats("time").print_stats()

    def test_memorize(self):
        with cProfile.Profile() as pr:
            self.vlite.memorize(self.corpus)
        self.assertTrue(len(self.vlite.data) > 0)
        #stats = Stats(pr)
        #stats.strip_dirs().sort_stats("time").print_stats()

    def test_remember(self):
        self.vlite.memorize(self.corpus)
        with cProfile.Profile() as pr:
            for q, query in enumerate(self.queries):
                data, metadata, top_sims = self.vlite.remember(query)
                if q == 0:
                    print(f'\n{query}\n{top_sims}\n{metadata[0]}')
                    print(f'\n{query}\n{data[0][0]}')
        #stats = Stats(pr)
        #stats.strip_dirs().sort_stats("time").print_stats()

if __name__ == '__main__':
    unittest.main()
