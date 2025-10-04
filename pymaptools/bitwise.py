__author__ = 'escherba'

"""
Various bitwise operations
"""
import ctypes
import struct
from itertools import izip
from functools import partial

PyLong_AsByteArray = ctypes.pythonapi._PyLong_AsByteArray
PyLong_AsByteArray.argtypes = [ctypes.py_object,
                               ctypes.c_char_p,
                               ctypes.c_size_t,
                               ctypes.c_int,
                               ctypes.c_int]


def bitlist(num):
    """Unpack number into a list, size-independent

    :param num: Some number
    :type num: int
    :returns: list of bits
    :rtype: list

    ::

        >>> import sys
        >>> import random
        >>> num = random.randint(0, sys.maxint)
        >>> bitlist(num) == bitlist(-num)
        True
        >>> bitlist(0)
        []
        >>> bitlist(31)
        [1, 1, 1, 1, 1]
    """
    vec = []
    if num < 0:
        num = -num
    while num > 0:
        vec.append(1 if num & 1 else 0)
        num >>= 1
    return vec


def bitstring(num):
    """Unpack number into a string, size-independent

    :param num: Some number
    :type num: int
    :returns: string of bits
    :rtype: str
    """
    return '{0:b}'.format(num)[::-1]


def bitstring_padded(size, num):
    """Unpack number into a string, size-independent

    :param num: Some number
    :type num: int
    :returns: string of bits
    :rtype: str
    """
    res = '{0:b}'.format(num)[::-1]
    return res.ljust(size, '0')


def from_bitlist(iterable):
    """Undo bitlist"""
    return sum(1 << i for i, b in enumerate(iterable) if b)


def from_bitstring(iterable):
    """Undo bitstring
    """
    return sum(1 << i for i, b in enumerate(iterable) if int(b))


def hamming_from_iter(vec1, vec2):
    """Return the Hamming distance between two lists of bits

    :param vec1: sequence 1
    :type vec1: list
    :param vec2: sequence 2
    :type vec2: list
    :returns: hamming distance between two lists of bits
    :rtype: int
    """

    len_delta = len(vec1) - len(vec2)
    if len_delta > 0:
        vec2 += [0] * len_delta
    else:
        vec1 += [0] * (-len_delta)
    return sum(ch1 != ch2 for ch1, ch2 in izip(vec1, vec2))


def hamming(num1, num2):
    """Return the Hamming distance between bits of two numbers

    :param num1: some number
    :type num1: long, int
    :param num2: some number
    :type num2: long, int
    :returns: hamming distance between two numbers
    :rtype: int
    """
    return bitlist(num1 ^ num2).count(1)


def packl_ctypes(bit_depth, num):
    """
    :param bit_depth: number of bits for encoding
    :type bit_depth: int
    :param num: input number to encode
    :type num: long
    :return: packed bitstring
    :rtype: str

    ::

        >>> num, denom = 37, 6
        >>> ((num + denom - 1) // denom) == 7
        True
        >>> num, denom = 36, 6
        >>> ((num + denom - 1) // denom) == 6
        True
    """
    bytes_needed = (bit_depth + 7) // 8
    buf = ctypes.create_string_buffer(bytes_needed + 1)
    PyLong_AsByteArray(long(num), buf, len(buf), 0, 1)
    return buf.raw[1:]


def create_bit_packer(bit_depth):
    """
    Returns a function that will pack integers into strings
    """
    if bit_depth == 64:
        fun = partial(struct.pack, '>Q')  # 64-bit sketch
    elif bit_depth == 32:
        fun = partial(struct.pack, '>L')  # 32-bit sketch
    elif bit_depth == 16:
        fun = partial(struct.pack, '>H')  # 16-bit sketch
    elif bit_depth == 8:
        fun = partial(struct.pack, '>B')  # 8-bit sketch
    else:
        fun = partial(packl_ctypes, bit_depth)
    return fun
