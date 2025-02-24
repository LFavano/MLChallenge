import unittest
import json
import numpy as np
from search_engine import load_index, search_patents
from data_processor import process_and_store_data
from sentence_transformers import SentenceTransformer
import faiss
import os

TEST_DATA_FILE = "test_data.txt"
TEST_INDEX_FILE = "test_patents.index"
TEST_METADATA_FILE = "test_metadata.json"
MODEL_NAME = "sentence-transformers/LaBSE"
model = SentenceTransformer(MODEL_NAME)


class TestPatentSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test data
        test_abstracts = [
            "A method for making a widget with improved durability.",
            "A system for processing multilingual text efficiently.",
            "A technique to enhance battery life in mobile devices."
        ]

        with open(TEST_DATA_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(test_abstracts))

        # Process and store test data
        process_and_store_data(TEST_DATA_FILE, TEST_INDEX_FILE, TEST_METADATA_FILE)

        # Load generated index and metadata
        cls.index, cls.metadata = load_index(TEST_INDEX_FILE, TEST_METADATA_FILE)

    @classmethod
    def tearDownClass(cls):
        os.remove(TEST_DATA_FILE)
        os.remove(TEST_INDEX_FILE)
        os.remove(TEST_METADATA_FILE)

    def test_load_index(self):
        self.assertIsInstance(self.index, faiss.IndexFlatL2)
        self.assertIsInstance(self.metadata, dict)
        self.assertGreater(len(self.metadata), 0)

    def test_search_patents(self):
        query = "battery efficiency"
        query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        results = search_patents(self.index, self.metadata, query_embedding, threshold=0.3)

        self.assertGreater(len(results), 0)
        self.assertIn("abstract", results[0])
        self.assertIn("score", results[0])
        self.assertTrue(isinstance(results[0]["score"], float))


if __name__ == "__main__":
    unittest.main()
