#cython: infer_types=True

import sys
from struct import unpack
from itertools import izip
from hashlib import md5


cdef extern from * nogil:
    ctypedef unsigned char uint8_t
    ctypedef unsigned long int uint32_t
    ctypedef unsigned long long int uint64_t


ctypedef uint8_t uint8
ctypedef uint32_t uint32
ctypedef uint64_t uint64


cdef class PHashCombiner(object):
    """Use polynomial hashing to reduce a vector of hashes
    """

    cdef list _coeffs
    cdef _mask

    def __cinit__(self, size, prime=31, bits=64):
        # TODO: cdef prime and i to int64 to get speedup
        self._coeffs = [prime ** i for i in xrange(size)]
        self._mask = (1 << bits) - 1

    def combine(self, hashes):
        """Combine a list of integer hashes
        """
        # TODO: cdef h and c to int64 to get speedup
        ab = sum(h * c for h, c in izip(hashes, self._coeffs))
        return ab & self._mask


cpdef inline uint64 hash_combine_boost_64(uint64 seed, uint64 v):
    """Hash two 64-bit integers together
    Uses boost::hash_combine algorithm
    """
    return seed ^ (v + 0x9e3779b9ULL + (seed << 6ULL) + (seed >> 2ULL))


cpdef inline hash_combine_boost(seed, v):
    """Hash two 64-bit integers together
    Uses boost::hash_combine algorithm
    """
    return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2))


cpdef inline uint64 hash_combine_murmur_64(uint64 seed, uint64 v):
    """Hash two 64-bit integers together
    Uses a Murmur-inspired hash function
    """
    cdef uint64 a = (seed ^ v) * 0x9ddfea08eb382d69ULL
    a ^= (a >> 47ULL)
    cdef uint64 b = (v ^ a) * 0x9ddfea08eb382d69ULL
    b ^= (b >> 47ULL)
    b *= 0x9ddfea08eb382d69ULL
    return b


cpdef inline hash_combine_murmur(seed, v):
    """Hash two 64-bit integers together
    Uses a Murmur-inspired hash function
    """
    a = (seed ^ v) * 0x9ddfea08eb382d69
    a ^= (a >> 47)
    b = (v ^ a) * 0x9ddfea08eb382d69
    b ^= (b >> 47)
    b *= 0x9ddfea08eb382d69
    return b


cpdef inline hashable(value):
    if not isinstance(value, str):
        return repr(value)
    return value


cpdef uint64 hash_md5_64(x, uint64 seed=0):
    """Return value is 128 bits
    """
    cdef uint64 a
    cdef uint64 b
    a, b = unpack('<QQ', md5(hashable(x)).digest())
    return hash_combine_boost_64(seed, hash_combine_boost_64(a, b))


cpdef hash_md5_128(x, seed=0):
    """Return value is 128 bits
    """
    ab = hash_combine_boost(seed, long(md5(hashable(x)).hexdigest(), 16))
    return ab & ((1 << 128) - 1)


cpdef uint64 hash_builtin_64(x, uint64 seed=0):
    """A better hash function based on Python's built-in hash()

    Note: the built-in hash() is a terrible hash function. For some examples,
    see http://michaelnielsen.org/blog/consistent-hashing/
    """
    cdef uint64 a = hash(x) & 0xffffffffULL
    cdef uint64 b = hash(repr(x)) & 0xffffffffULL
    return hash_combine_boost_64(seed, 0x100000000ULL * a + b)


cpdef hash_builtin_128(x, seed=0):
    """A better hash function based on Python's built-in hash()

    Note: vanilla hash() is a terrible hash function.
    """
    a = hash_builtin_64(x, seed)
    b = hash_builtin_64(repr(x), seed)
    return (1 << 64) * a + b


cdef class VarlenHash(object):
    """Create a hash function of arbitrary output length
    """
    cdef _mask

    def __cinit__(self, bits=64):
        """
        :param bits: number of bits in hashed values
        :type bits: int,long

        """
        self._mask = (1 << bits) - 1

    def __call__(self, value, seed=0):
        """A variable-length version of Python's builtin hash
        """
        if isinstance(value, unicode):
            value = value.encode("utf-8")
        elif isinstance(value, str):
            pass
        else:
            value = repr(value)
        length_of_v = len(value)
        if length_of_v > 0:
            item = ord(value[0]) << 7
            mask = self._mask
            for char in value:
                item = ((item * 1000003) ^ ord(char)) & mask
            item ^= length_of_v
            if item == -1:
                item = -2
            return hash_combine_boost(item, seed)
        else:
            return 0


cpdef long2int(num):
    """Lossily map a long type to the range of int

    :param num: input long variable
    :type num: long
    :return: input mapped to int range
    :rtype: int

    """

    smi1 = sys.maxint + 1
    return int(num % (smi1 + smi1) - smi1)
