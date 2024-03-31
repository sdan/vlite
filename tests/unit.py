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
        self.vlite = VLite("vlite-unit")

    def tearDown(self):
        # Remove the file
        if os.path.exists('vlite-unit'):
            print("[+] Removing vlite")
            os.remove('vlite-unit')

    def test_add__text(self):
        start_time = time.time()
        text = "This is a test text."
        metadata = {"source": "test"}
        self.vlite.add(text, metadata=metadata)
        self.assertEqual(self.vlite.count(), 1)
        end_time = time.time()
        TestVLite.test_times["add_single_text"] = end_time - start_time
        print(f"Count of texts in the collection: {self.vlite.count()}")

    def test_add_texts(self):
        start_time = time.time()
        text_512tokens = "underreckoning fleckiness hairstane paradigmatic eligibility sublevate xviii achylia reremice flung outpurl questing gilia unosmotic unsuckled plecopterid excludable phenazine fricando unfledgedness spiritsome incircle desmogenous subclavate redbug semihoral district chrysocolla protocoled servius readings propolises javali dujan stickman attendee hambone obtusipennate tightropes monitorially signaletics diestrums preassigning spriggy yestermorning margaritic tankfuls aseptify linearity hilasmic twinning tokonoma seminormalness cerebrospinant refroid doghouse kochab dacryocystalgia saltbushes newcomer provoker berberid platycoria overpersuaded reoverflow constrainable headless forgivably syzygal purled reese polyglottonic decennary embronze pluripotent equivocally myoblasts thymelaeaceous confervae perverted preanticipate mammalogical desalinizing tackets misappearance subflexuose concludence effluviums runtish gras cuckolded hemostasia coatroom chelidon policizer trichinised frontstall impositions unta outrance scholium fibrochondritis furcates fleaweed housefront helipads hemachate snift appellativeness knobwood superinclination tsures haberdasheries unparliamented reexecution nontangential waddied desolated subdistinctively undiscernibleness swishiest dextral progs koprino bruisingly unloanably bardash uncuckoldedunderreckoning fleckiness hairstane paradigmatic eligibility sublevate xviii achylia reremice flung outpurl questing gilia unosmotic unsuckled plecopterid excludable phenazine fricando unfledgedness spiritsome incircle desmogenous subclavate redbug semihoral district chrysocolla spriggy yestermorning margaritic tankfuls aseptify linearity hilasmic twinning tokonoma seminormalness cerebrospinant refroequivocally myoblasts thymelaeaceous confervae perverted preantiest dextral progs koprino bruisingly unloanably bardash uncuckolded"
        metadata = {"source": "test_512tokens"}
        self.vlite.add(text_512tokens, metadata=metadata)
        with open("data/text-8192tokens.txt", "r") as file:
            text_8192tokens = file.read()
        metadata = {"source": "test_8192tokens"}
        self.vlite.add(text_8192tokens, metadata=metadata)
        end_time = time.time()
        TestVLite.test_times["add_multiple_texts"] = end_time - start_time
        print(f"Count of texts in the collection: {self.vlite.count()}")
    
    def test_add_pdf(self):
        start_time = time.time()
        process_pdf('data/gpt-4.pdf')
        end_time = time.time()
        TestVLite.test_times["add_pdf"] = end_time - start_time
        # time to add 71067 tokens from the GPT-4 paper
        print(f"Time to add 71067 tokens: {TestVLite.test_times['add_pdf']} seconds")

    def test_retrieve(self):
        queries = [
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
        process_pdf('data/gpt-4.pdf')
        start_time = time.time()
        for query in self.queries:
            _, top_sims, _ = self.vlite.retrieve(query)  
            print(f"Top similarities for query '{query}': {top_sims}")
        end_time = time.time()
        TestVLite.test_times["retrieve"] = end_time - start_time
    
    def test_delete(self):
        self.vlite.add("This is a test text.")
        start_time = time.time()
        self.vlite.delete(0)
        end_time = time.time()
        TestVLite.test_times["delete"] = end_time - start_time
        print(f"Count of texts in the collection: {self.vlite.count()}")
    
    def test_update(self):
        self.vlite.add("This is a test text.")
        start_time = time.time()
        self.vlite.update(0, "This is an updated text.")
        end_time = time.time()
        TestVLite.test_times["update"] = end_time - start_time
        print(f"Count of texts in the collection: {self.vlite.count()}")

    @classmethod
    def tearDownClass(cls):
        print("\nTest times:")
        for test_name, test_time in cls.test_times.items():
            print(f"{test_name}: {test_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main(verbosity=2)