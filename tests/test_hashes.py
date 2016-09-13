import unittest
import random
from clustering_metrics.hashes import XORHashFamily, MultiplyHashFamily


class TestHashes(unittest.TestCase):

    def test_xor_family(self):
        """Calling XOR family hasher twice results in the same value
        """
        num_hashes = random.randint(1, 10)
        num_buckets = random.randint(1, 10)
        value = random.randint(0, 2 ** 32 - 1)
        xh = XORHashFamily(num_hashes, num_buckets)
        self.assertEqual(value, xh.hashn(xh.hashn(value).next()).next())

    def test_multiply_family(self):
        """When using one bucket, all resulting vaules are identical
        """
        mh = MultiplyHashFamily(3, 1)
        value = random.randint(0, 2 ** 32 - 1)
        results = list(mh.hashn(value))
        self.assertEqual(1, len(set(results)))
