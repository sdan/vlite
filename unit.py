import unittest
import numpy as np
from main import VLite
import os
from utils import load_file
import cProfile
from pstats import Stats

class TestVLite(unittest.TestCase):
    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

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

        self.vlite = VLite()

    def tearDown(self):
        p = Stats (self.pr)
        

        # remove the file
        if os.path.exists('vlite.pkl'):
            print("[+] Removing vlite.pkl")
            os.remove('vlite.pkl')

        p.strip_dirs()
        p.sort_stats ('cumtime')
        p.print_stats ()


    def test_add_vector(self):
        self.vlite.add_vector(np.random.rand(384, 384))
        print("[test_add_vector] Vectors:", self.vlite.vectors)
        self.assertEqual(self.vlite.vectors.shape, (384, 384))

    def test_get_similar_vectors(self):
        self.vlite.add_vector(np.random.rand(384, 384))
        print("[test_get_similar_vectors] Curr Vectors shape:",
              self.vlite.vectors.shape)
        indices, sims = self.vlite.get_similar_vectors(np.random.rand(1, 384))
        print("[test_get_similar_vectors] Indices:", indices)
        print("[test_get_similar_vectors] Sims:", sims)
        self.assertEqual(indices.shape, (5,))

    def test_memorize(self):
        self.vlite.memorize(self.corpus)
        print("[test_memorize] Vectors shape:", self.vlite.vectors.shape)
        self.assertEqual(self.vlite.vectors.shape[1], 384)
    
    def test_remember(self):
        self.vlite.memorize(self.corpus)
        print("[test_remember] Memorized shape:", self.vlite.vectors.shape)
        for query in self.queries:
            print("[test_remember] Query:", query)
            _, top_sims = self.vlite.remember(query)
            print("[test_remember] Top scores:", top_sims)
            self.assertEqual(len(top_sims), 5)

if __name__ == '__main__':
    unittest.main()
