# -*- coding: utf-8 -*-
import unittest
from pymaptools.bitwise import hamming
from clustering_metrics import MinHashSignature, SimHashSignature, \
    MinHashSketchSignature, Shingler
from clustering_metrics.metrics import jaccard_similarity
from clustering_metrics.utils import randset, sigsim
from clustering_metrics.preprocess import RegexTokenizer


class TestSig(unittest.TestCase):
    def test_shingler(self):
        s = Shingler(span=5, skip=1, unique=True)
        shingles = s.get_shingles("abracadabra")
        self.assertIn(("d", "b", "a"), shingles)

        t = Shingler(span=5, skip=1, unique=False)
        shingles = t.get_shingles("abracadabra")
        self.assertEqual(("a", "r", "c"), shingles[0])

    def test_word_shingler(self):
        s = Shingler(span=5, skip=1, unique=True, tokenizer=RegexTokenizer())
        shingles = s.get_shingles("the quick brown fox jumps over a lazy dog")
        self.assertIn(("jumps", "a", "dog"), shingles)

        t = Shingler(span=5, skip=1, unique=False, tokenizer=RegexTokenizer())
        shingles = t.get_shingles("the quick brown fox jumps over a lazy dog")
        self.assertEqual(("the", "brown", "jumps"), shingles[0])

    def test_signature_length(self):
        """Signatures should have correct dimension"""
        mh = MinHashSignature(10 * 10)
        self.assertEqual(100, len(mh.get_signature(randset())))

    def test_consistent_signature(self):
        """Signatures should be consistent"""
        mh = MinHashSignature(10 * 10)
        s = randset()
        self.assertEqual(mh.get_signature(s), mh.get_signature(s))

    def test_simhash64_1(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("")
        sig2 = sh.get_signature(u"")
        self.assertEqual(sig1, sig2)

    def test_simhash64_2(self):
        sh = SimHashSignature(64)
        sig3 = sh.get_signature("abracadabra")
        sig4 = sh.get_signature(u"abracadabra")
        self.assertEqual(sig3, sig4)

    def test_simhash64_3(self):
        sh = SimHashSignature(64)
        str1 = "♡♥❤❥"
        str2 = u"♡♥❤❥"
        sig5 = sh.get_signature(str1)
        sig6 = sh.get_signature(str2)
        self.assertNotEqual(sig5, sig6)

    def test_simhash128_1(self):
        sh = SimHashSignature(128)
        sig1 = sh.get_signature("")
        sig2 = sh.get_signature(u"")
        self.assertEqual(0, sig1)
        self.assertEqual(sig1, sig2)

    def test_simhash128_2(self):
        sh = SimHashSignature(128)
        sig3 = sh.get_signature("abracadabra")
        sig4 = sh.get_signature(u"abracadabra")
        self.assertEqual(sig3, sig4)

    def test_simhash128_3(self):
        sh = SimHashSignature(128)
        str1 = "♡♥❤❥"
        str2 = u"♡♥❤❥"
        sig5 = sh.get_signature(str1)
        sig6 = sh.get_signature(str2)
        self.assertNotEqual(sig5, sig6)

    def test_simhash_similarity_1(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracadabra")
        dist = hamming(sig1, sig2)
        self.assertEqual(0, dist)

    def test_simhash_similarity_2(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("arbcd")
        dist = hamming(sig1, sig2)
        self.assertEqual(12, dist)

    def test_simhash_similarity_3(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("")
        dist = hamming(sig1, sig2)
        self.assertEqual(37, dist)

    def test_minhash_sketch_similarity_1(self):
        sh = MinHashSketchSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracadabra")
        dist = hamming(sig1, sig2)
        self.assertEqual(0, dist)

    def test_minhash_sketch_similarity_2(self):
        sh = MinHashSketchSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("arbcd")
        dist = hamming(sig1, sig2)
        self.assertEqual(0, dist)

    def test_minhash_sketch_similarity_3(self):
        sh = MinHashSketchSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("")
        dist = hamming(sig1, sig2)
        self.assertEqual(32, dist)

    def test_simhash_feature_weights_1(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra")
        sig2 = sh.get_signature("abracdabra")
        dist = hamming(sig1, sig2)
        self.assertEqual(3, dist)

    def test_simhash_feature_weights_2(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("cats", 0))
        sig2 = sh.get_signature("abracdabra", ("dogs", 0))
        dist = hamming(sig1, sig2)
        self.assertEqual(3, dist)

    def test_simhash_feature_weights_3(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("cats", 0))
        sig2 = sh.get_signature("abracadabra", ("dogs", 0))
        dist = hamming(sig1, sig2)
        self.assertEqual(0, dist)

    def test_simhash_feature_weights_4(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("ca", 4))
        sig2 = sh.get_signature("abracadabra", ("do", 4))
        dist = hamming(sig1, sig2)
        self.assertEqual(11, dist)

    def test_simhash_feature_weights_5(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("ca", 5))
        sig2 = sh.get_signature("abracadabra", ("do", 5))
        dist = hamming(sig1, sig2)
        self.assertEqual(11, dist)

    def test_simhash_feature_weights_6(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("cats", 200))
        sig2 = sh.get_signature("abracadabra", ("dogs", 200))
        dist = hamming(sig1, sig2)
        self.assertEqual(33, dist)

    def test_simhash_feature_weights_7(self):
        sh = SimHashSignature(64)
        sig1 = sh.get_signature("abracadabra", ("cats", 10))
        sig2 = sh.get_signature("abracadabra", ("cats", 10))
        dist = hamming(sig1, sig2)
        self.assertEqual(0, dist)

    def test_signature_similarity(self):
        """The probability that two sets' signatures match at some index are
        equal is equal to the Jaccard similarity between the two
        """
        n_tests = 100
        expected_error = 1.0 / 10  # Expected error is O(1/sqrt(dim))
        mh = MinHashSignature(10 * 10)
        err = 0.0

        for _ in xrange(n_tests):
            # Create random sets and their signatures
            sets = (randset(), randset())
            sigs = map(mh.get_signature, sets)

            # Calculate true Jaccard similarity, and sim of signatures
            jsim = jaccard_similarity(*sets)
            ssim = sigsim(*sigs, dim=100)

            # Accumulate error
            err += abs(jsim - ssim)

        # Over n_tests large, we should be within upper bound of expected error
        avg_err = err / n_tests
        self.assertGreaterEqual(
            expected_error,
            avg_err,
            msg="Accuracy test failed. (avg error: %f)" % avg_err)


if __name__ == '__main__':
    unittest.main()
