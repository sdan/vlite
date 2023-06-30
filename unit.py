import unittest
import os
from uuid import uuid4
from main import VLite

class TestVLite(unittest.TestCase):
    def setUp(self):
        self.db = VLite(collection='test.pkl')

    def tearDown(self):
        # Clean up the test database after each test
        try:
            os.remove('test.pkl')
        except FileNotFoundError:
            pass

    def test_memorize(self):
        id = str(uuid4())
        self.db.memorize('Hello world', id=id)
        self.assertEqual(len(self.db.data), 1)
        self.assertTrue(id in self.db.data)

    def test_remember_by_id(self):
        id = str(uuid4())
        self.db.memorize('Hello world', id=id)
        result = self.db.remember(id=id)
        self.assertEqual(result['text'], 'Hello world')

    def test_remember_by_text(self):
        self.db.memorize('Hello world')
        self.db.memorize('Goodbye world')
        results = self.db.remember(text='Hello world')
        self.assertEqual(len(results), 1)

    def test_save(self):
        id = str(uuid4())
        self.db.memorize('Hello world', id=id)
        self.db.save()
        new_db = VLite(collection='test.pkl')
        self.assertEqual(len(new_db.data), 1)
        self.assertTrue(id in new_db.data)

if __name__ == '__main__':
    unittest.main()
