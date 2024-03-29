import unittest
import numpy as np
from vlite.main import VLite
from vlite.utils import load_file
import os
import time

class TestVLite(unittest.TestCase):
    def setUp(self):
        self.vlite = VLite()
        self.corpus = load_file('data/gpt-4.pdf')

    def tearDown(self):
        if os.path.exists(self.vlite.get_index_file()):
            os.remove(self.vlite.get_index_file())

    def test_retrieve(self):
        print("Ingesting corpus...")
        start_time = time.time()
        for text in self.corpus:
            self.vlite.ingest(text)
        end_time = time.time()
        print(f"Ingestion time: {end_time - start_time:.2f} seconds")

        query = "What is the architecture of GPT-4?"
        top_k = 3
        
        print(f"Retrieving top {top_k} results...")
        start_time = time.time()
        results = self.vlite.retrieve(query, top_k=top_k, get_similarities=True)
        end_time = time.time()
        print(f"Retrieval time: {end_time - start_time:.2f} seconds")

        print("Results retrieved successfully.")
        print(f"Number of texts retrieved: {len(results['texts'])}")
        print(f"Number of scores retrieved: {len(results['scores'])}")
        print("Results: ", results)
        print("Query Text: ", query)

        self.assertEqual(len(results["texts"]), top_k)
        self.assertEqual(len(results["scores"]), top_k)


if __name__ == '__main__':
    unittest.main()