import unittest
import numpy as np
from vlite.main import VLite
import os
from vlite.utils import process_pdf
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestVLite(unittest.TestCase):
    test_times = {}
    vlite = VLite("vlite-unit")

    def test_add_text(self):
        logger.info("[TestVLite.test_add_text] Starting test")
        start_time = time.time()
        text = "This is a test text. " * 86
        metadata = {"source": "test", "author": "John Doe", "timestamp": "2023-06-08"}
        result = self.vlite.add(text, metadata=metadata, item_id="test_text_1", need_chunks=False)
        end_time = time.time()
        TestVLite.test_times["add_single_text"] = end_time - start_time
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_add_text] Count of texts in the collection: {count}")
        self.assertEqual(count, 18)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test_text_1")
        self.assertEqual(result[0][2], metadata)
        logger.info("[TestVLite.test_add_text] Test completed")

    def test_add_texts(self):
        logger.info("[TestVLite.test_add_texts] Starting test")
        start_time = time.time()
        text_512tokens = "underreckoning fleckiness hairstane paradigmatic eligibility sublevate xviii achylia reremice flung outpurl questing gilia unosmotic unsuckled plecopterid excludable phenazine fricando unfledgedness spiritsome incircle desmogenous subclavate redbug semihoral district chrysocolla protocoled servius readings propolises javali dujan stickman attendee hambone obtusipennate tightropes monitorially signaletics diestrums preassigning spriggy yestermorning margaritic tankfuls aseptify linearity hilasmic twinning tokonoma seminormalness cerebrospinant refroid doghouse kochab dacryocystalgia saltbushes newcomer provoker berberid platycoria overpersuaded reoverflow constrainable headless forgivably syzygal purled reese polyglottonic decennary embronze pluripotent equivocally myoblasts thymelaeaceous confervae perverted preanticipate mammalogical desalinizing tackets misappearance subflexuose concludence effluviums runtish gras cuckolded hemostasia coatroom chelidon policizer trichinised frontstall impositions unta outrance scholium fibrochondritis furcates fleaweed housefront helipads hemachate snift appellativeness knobwood superinclination tsures haberdasheries unparliamented reexecution nontangential waddied desolated subdistinctively undiscernibleness swishiest dextral progs koprino bruisingly unloanably bardash uncuckoldedunderreckoning fleckiness hairstane paradigmatic eligibility sublevate xviii achylia reremice flung outpurl questing gilia unosmotic unsuckled plecopterid excludable phenazine fricando unfledgedness spiritsome incircle desmogenous subclavate redbug semihoral district chrysocolla spriggy yestermorning margaritic tankfuls aseptify linearity hilasmic twinning tokonoma seminormalness cerebrospinant refroequivocally myoblasts thymelaeaceous confervae perverted preantiest dextral progs koprino bruisingly unloanably bardash uncuckolded"
        metadata = {"source": "test_512tokens", "category": "random"}
        result_512 = self.vlite.add(text_512tokens, metadata=metadata, item_id="test_text_2")
        with open(os.path.join(os.path.dirname(__file__), "data/text-8192tokens.txt"), "r") as file:
            text_8192tokens = file.read()
        metadata = {"source": "test_8192tokens", "category": "long_text"}
        result_8192 = self.vlite.add(text_8192tokens, metadata=metadata, item_id="test_text_3")
        end_time = time.time()
        TestVLite.test_times["add_multiple_texts"] = end_time - start_time
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_add_texts] Count of texts in the collection: {count}")
        self.assertEqual(count, 20)
        self.assertEqual(len(result_512), 1)
        self.assertEqual(len(result_8192), 1)
        self.assertEqual(result_512[0][0], "test_text_2")
        self.assertEqual(result_8192[0][0], "test_text_3")
        logger.info("[TestVLite.test_add_texts] Test completed")

    def test_add_pdf(self):
        logger.info("[TestVLite.test_add_pdf] Starting test")
        logger.info(f"[TestVLite.test_add_pdf] Count of chunks in the collection: {self.vlite.count()}")
        start_time = time.time()
        result = self.vlite.add(process_pdf(os.path.join(os.path.dirname(__file__), 'data/attention.pdf')), need_chunks=False, item_id="test_pdf_1")
        end_time = time.time()
        TestVLite.test_times["add_pdf"] = end_time - start_time
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_add_pdf] After count of chunks in the collection: {count}")
        logger.info(f"[TestVLite.test_add_pdf] Time to add 71067 tokens: {TestVLite.test_times['add_pdf']} seconds")
        self.assertGreater(count, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "test_pdf_1")
        logger.info("[TestVLite.test_add_pdf] Test completed")

    def test_retrieve(self):
        logger.info("[TestVLite.test_retrieve] Starting test")
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
        start_time = time.time()
        for query in queries:
            results = self.vlite.retrieve(query, top_k=3, return_scores=True)
            logger.info(f"[TestVLite.test_retrieve] Query: {query}")
            logger.info("[TestVLite.test_retrieve] Top 3 results:")
            for idx, text, metadata, score in results[:3]:
                logger.info(f"[TestVLite.test_retrieve] Text: {text[:100]}...")
                logger.info(f"[TestVLite.test_retrieve] Similarity: {score}")
                logger.info(f"[TestVLite.test_retrieve] Metadata: {metadata}")
                logger.info("[TestVLite.test_retrieve] ---")
            self.assertEqual(len(results), 3)
            self.assertIsInstance(results[0][1], str)
            self.assertIsInstance(results[0][2], dict)
        end_time = time.time()
        TestVLite.test_times["retrieve"] = end_time - start_time
        logger.info("[TestVLite.test_retrieve] Test completed")

    def test_delete(self):
        logger.info("[TestVLite.test_delete] Starting test")
        self.vlite.add("This is a test text.", item_id="test_delete_1")
        self.vlite.add("Another test text.", item_id="test_delete_2")
        start_time = time.time()
        deleted_count = self.vlite.delete(['test_delete_1', 'test_delete_2'])
        end_time = time.time()
        TestVLite.test_times["delete"] = end_time - start_time
        self.assertEqual(deleted_count, 2)
        logger.info("[TestVLite.test_delete] Test completed")

    def test_update(self):
        logger.info("[TestVLite.test_update] Starting test")
        self.vlite.add("This is a test text.", item_id="test_update_1")
        start_time = time.time()
        updated = self.vlite.update("test_update_1", text="This is an updated text.", metadata={"updated": True})
        end_time = time.time()
        TestVLite.test_times["update"] = end_time - start_time
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_update] Count of texts in the collection: {count}")
        self.assertTrue(updated)
        self.assertEqual(count, 5)
        logger.info("[TestVLite.test_update] Test completed")

    def test_get(self):
        logger.info("[TestVLite.test_get] Starting test")
        self.vlite.add("Text 1", item_id="test_get_1", metadata={"category": "A"})
        self.vlite.add("Text 2", item_id="test_get_2", metadata={"category": "B"})
        self.vlite.add("Text 3", item_id="test_get_3", metadata={"category": "A"})
        start_time = time.time()
        items = self.vlite.get(ids=["test_get_1", "test_get_3"])
        logger.info(f"[TestVLite.test_get] Items with IDs 'test_get_1' and 'test_get_3': {items}")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][1], "Text 1")
        self.assertEqual(items[1][1], "Text 3")
        items = self.vlite.get(where={"category": "A"})
        logger.info(f"[TestVLite.test_get] Items with category 'A': {items}")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][2]["category"], "A")
        self.assertEqual(items[1][2]["category"], "A")
        end_time = time.time()
        TestVLite.test_times["get"] = end_time - start_time
        logger.info("[TestVLite.test_get] Test completed")

    def test_set(self):
        logger.info("[TestVLite.test_set] Starting test")
        self.vlite.add("Original text", item_id="test_set_1", metadata={"original": True})
        start_time = time.time()
        self.vlite.set("test_set_1", text="Updated text", metadata={"updated": True})
        end_time = time.time()
        TestVLite.test_times["set"] = end_time - start_time
        items = self.vlite.get(ids=["test_set_1"])
        logger.info(f"[TestVLite.test_set] Updated item: {items}")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], "Updated text")
        self.assertEqual(items[0][2]["updated"], True)
        logger.info("[TestVLite.test_set] Test completed")

    def test_count(self):
        logger.info("[TestVLite.test_count] Starting test")
        start_time = time.time()
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_count] Count of items in the collection: {count}")
        end_time = time.time()
        TestVLite.test_times["count"] = end_time - start_time
        logger.info("[TestVLite.test_count] Test completed")

    def test_clear(self):
        logger.info("[TestVLite.test_clear] Starting test")
        start_time = time.time()
        self.vlite.clear()
        count = self.vlite.count()
        logger.info(f"[TestVLite.test_clear] Count of items after clearing the collection: {count}")
        self.assertEqual(count, 0)
        end_time = time.time()
        TestVLite.test_times["clear"] = end_time - start_time
        logger.info("[TestVLite.test_clear] Test completed")

    @classmethod
    def tearDownClass(cls):
        logger.info("[TestVLite.tearDownClass] Tearing down the test class")
        logger.info("[TestVLite.tearDownClass] Test times:")
        for test_name, test_time in cls.test_times.items():
            logger.info(f"[TestVLite.tearDownClass] {test_name}: {test_time:.4f} seconds")
        if os.path.exists('contexts/vlite-unit.ctx'):
            logger.info("[TestVLite.tearDownClass] Removing vlite ctx")
            os.remove('contexts/vlite-unit.ctx')

if __name__ == '__main__':
    unittest.main(verbosity=2)