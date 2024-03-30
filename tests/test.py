import unittest
import numpy as np
import uuid
import time
from vlite.main import VLite

class TestVLite(unittest.TestCase):
    test_times = {}

    def setUp(self):
        self.vlite = VLite()
        self.id = str(uuid.uuid4())
        self.metadata = {"name": "test"}

    def test_add(self):
        start_time = time.time()
        test_added_text = self.vlite.add(text=self.long_data_2)
        end_time = time.time()
        TestVLite.test_times["add"] = end_time - start_time
        print(f"Added text: {test_added_text}")

    def test_retrieve_short(self):
        start_time = time.time()
        test_retrieved_text = self.vlite.retrieve(text="civil law")
        end_time = time.time()
        TestVLite.test_times["retrieve_short"] = end_time - start_time
        print(f"Retrieved text: {test_retrieved_text}")

    def test_retrieve_long(self):
        start_time = time.time()
        test_retrieve_more_text = self.vlite.retrieve(text="Overall, while common law judges are seen as central figures in shaping and interpreting the law, civil law judges traditionally have more limited roles and are viewed as functionaries. How ever, there are ongoing changes and developments in the civil law tradition that are expanding the scope and power of judges. Legal science, also known as systematic jurisprudence or conceptual jurisprudence, is a dominant school of thought within the civi l law tradition. It emerged in the 19th century and is primarily associated with German legal scholars. Legal science emphasizes the scientific study of law and seeks to discover inherent principles and relationships within legal materials. It aims to crea te a systematic and coherent legal structure by developing")
        end_time = time.time()
        TestVLite.test_times["retrieve_long"] = end_time - start_time
        print(f"Retrieved long text: {test_retrieve_more_text}")

    @classmethod
    def tearDownClass(cls):
        print("\nTest times:")
        for test_name, test_time in cls.test_times.items():
            print(f"{test_name}: {test_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main(verbosity=2)