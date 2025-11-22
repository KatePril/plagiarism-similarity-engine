import unittest
from src.min_hash_generator import MinHash
from src.locality_sensitive_hashing import LSH, LSH_generator


class TestLSHGenerator(unittest.TestCase):

    @staticmethod
    def create_minhash(words, num_perms=128, seed=42):
        mh = MinHash(num_permutations=num_perms, seed=seed)
        for word in words:
            mh.update(word)
        return mh

    def test_insert_and_query_workflow(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "doc1": self.create_minhash(["machine", "learning", "algorithms"], seed=42),
            "doc2": self.create_minhash(["machine", "learning", "models"], seed=42),
            "doc3": self.create_minhash(["deep", "neural", "networks"], seed=42),
            "doc4": self.create_minhash(["database", "query", "optimization"], seed=42)
        }

        lsh = generator.generate_lsh(docs)
        query_mh = self.create_minhash(["machine", "learning", "algorithms"], seed=42)
        candidates = lsh.query(query_mh)

        self.assertEqual(len(lsh.signatures), 4)
        self.assertIn("doc1", candidates)
        for candidate in candidates:
            self.assertIn(candidate, docs.keys())

    def test_empty_minhash(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "empty_doc": MinHash(num_permutations=128)
        }
        lsh = generator.generate_lsh(docs)

        query_mh = MinHash(num_permutations=128)
        candidates = lsh.query(query_mh)

        self.assertIn("empty_doc", candidates)

    def test_single_element_minhash(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "doc1": self.create_minhash(["single"])
        }
        lsh = generator.generate_lsh(docs)

        query_mh = self.create_minhash(["single"])
        candidates = lsh.query(query_mh)

        self.assertIn("doc1", candidates)

    def test_large_number_of_documents(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        num_docs = 100
        docs = {}
        for i in range(num_docs):
            mh = self.create_minhash([f"word_{i}", f"word_{i + 1}"], seed=i)
            docs[f"doc_{i}"] = mh

        lsh = generator.generate_lsh(docs)
        query_mh = self.create_minhash(["word_50", "word_51"], seed=50)
        candidates = lsh.query(query_mh)

        self.assertEqual(len(lsh.signatures), num_docs)
        self.assertIn("doc_50", candidates)

    def test_high_similarity_documents(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        seed = 42
        docs = {
            "doc1": self.create_minhash(["a", "b", "c", "d", "e"], seed=seed),
            "doc2": self.create_minhash(["a", "b", "c", "d", "f"], seed=seed),
            "doc3": self.create_minhash(["a", "b", "c", "g", "h"], seed=seed),
        }

        lsh = generator.generate_lsh(docs)
        query_mh = self.create_minhash(["a", "b", "c", "d", "e"], seed=seed)
        candidates = lsh.query(query_mh)

        self.assertIn("doc1", candidates)
        for doc_id in ["doc1", "doc2", "doc3"]:
            similarity = query_mh.jaccard_similarity(docs[doc_id])
            if doc_id == "doc1":
                self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_low_similarity_documents(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "doc1": self.create_minhash(["aaa", "bbb", "ccc"], seed=1),
            "doc2": self.create_minhash(["xxx", "yyy", "zzz"], seed=2),
            "doc3": self.create_minhash(["ppp", "qqq", "rrr"], seed=3),
        }
        lsh = generator.generate_lsh(docs)

        query_mh = self.create_minhash(["aaa", "bbb", "ccc"], seed=1)
        candidates = lsh.query(query_mh)
        self.assertIn("doc1", candidates)

    def test_multiple_lsh_configurations(self):
        configs = [
            (32, 4),
            (8, 16),
            (64, 2),
        ]

        for num_bands, num_rows in configs:
            generator = LSH_generator(num_bands=num_bands, num_rows=num_rows)
            docs = {
                "doc1": self.create_minhash(["test", "document"], seed=42),
                "doc2": self.create_minhash(["another", "doc"], seed=42)
            }
            lsh = generator.generate_lsh(docs)
            query_mh = self.create_minhash(["test", "document"], seed=42)
            candidates = lsh.query(query_mh)

            self.assertEqual(lsh.num_bands, num_bands)
            self.assertEqual(lsh.num_rows, num_rows)
            self.assertEqual(len(lsh.signatures), 2)
            self.assertIn("doc1", candidates,
                          f"Failed for config: bands={num_bands}, rows={num_rows}")

    def test_incremental_insertion(self):
        docs = {
            "doc1": self.create_minhash(["word1", "word2"], seed=42),
            "doc2": self.create_minhash(["word3", "word4"], seed=42),
            "doc3": self.create_minhash(["word5", "word6"], seed=42)
        }

        generator = LSH_generator(num_bands=16, num_rows=8)
        lsh1 = generator.generate_lsh(docs)
        lsh2 = LSH(num_bands=16, num_rows=8)
        for doc_id, minhash in docs.items():
            lsh2.insert(doc_id, minhash)

        query_mh = self.create_minhash(["word1", "word2"], seed=42)
        candidates1 = lsh1.query(query_mh)
        candidates2 = lsh2.query(query_mh)

        self.assertEqual(len(lsh1.signatures), len(lsh2.signatures))
        self.assertEqual(set(lsh1.signatures.keys()), set(lsh2.signatures.keys()))
        self.assertEqual(candidates1, candidates2)

    def test_empty_document_collection(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {}
        lsh = generator.generate_lsh(docs)
        query_mh = self.create_minhash(["test"])
        candidates = lsh.query(query_mh)

        self.assertEqual(len(lsh.signatures), 0)
        self.assertEqual(len(candidates), 0)

    def test_single_document_collection(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "only_doc": self.create_minhash(["single", "document"], seed=42)
        }
        lsh = generator.generate_lsh(docs)
        query_mh = self.create_minhash(["single", "document"], seed=42)
        candidates = lsh.query(query_mh)

        self.assertEqual(len(lsh.signatures), 1)
        self.assertIn("only_doc", candidates)

    def test_realistic_document_collection(self):
        generator = LSH_generator(num_bands=16, num_rows=8)
        docs = {
            "article1": self.create_minhash(
                ["machine", "learning", "artificial", "intelligence", "neural", "network"],
                seed=42
            ),
            "article2": self.create_minhash(
                ["machine", "learning", "data", "science", "statistics", "analysis"],
                seed=42
            ),
            "article3": self.create_minhash(
                ["cooking", "recipe", "ingredients", "kitchen", "food", "meal"],
                seed=42
            ),
            "article4": self.create_minhash(
                ["sports", "football", "basketball", "tennis", "athlete", "competition"],
                seed=42
            )
        }
        lsh = generator.generate_lsh(docs)

        self.assertEqual(len(lsh.signatures), 4)
        query_mh = self.create_minhash(
            ["machine", "learning", "artificial", "intelligence", "neural", "network"],
            seed=42
        )
        candidates = lsh.query(query_mh)
        self.assertIn("article1", candidates)

        query_mh2 = self.create_minhash(
            ["cooking", "recipe", "ingredients", "kitchen", "food", "meal"],
            seed=42
        )
        candidates2 = lsh.query(query_mh2)
        self.assertIn("article3", candidates2)


if __name__ == '__main__':
    unittest.main(verbosity=2)