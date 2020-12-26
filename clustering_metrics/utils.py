# -*- coding: utf-8 -*-
import random
import operator
import string
from math import log
from operator import itemgetter
from pymaptools.iter import isiterable


PINF = float('inf')
NINF = float('-inf')
NAN = float('nan')


def _log(x, base=None):
    """Safe natural log
    """
    if x == 0.0:
        return NINF
    elif base is None:
        return log(x)
    else:
        return log(x, base)


def _div(numer, denom):
    """Safe division
    """
    if denom == 0.0:
        if numer == 0.0:
            return NAN
        elif numer > 0.0:
            return PINF
        else:
            return NINF
    return numer / float(denom)


def get_df_subset(df, fields):
    """Give a subset of a ``pandas.DataFrame`` instance
    """
    subset_fields = [field for field in set(fields) if field in df]
    return df[subset_fields]


def fill_with_last(lst, k):
    """
    extend a list to length k by duplicating last item

    >>> fill_with_last([1, 2, 3], 5)
    [1, 2, 3, 3, 3]
    """
    len_l = len(lst)
    if len_l < k:
        lst.extend([lst[-1]] * (k - len_l))
    return lst


def wrap_scalar(a):
    """If scalar, convert to tuple"""
    return a if isiterable(a) else (a,)


def tsorted(a):
    """Sort a tuple"""
    return tuple(sorted(a))


def getpropval(obj):
    """

    :return: a generator of properties and their values
    """
    return ((p, val) for p, val in ((p, getattr(obj, p)) for p in dir(obj))
            if not callable(val) and p[0] != '_')


def gapply(n, func, *args, **kwargs):
    """Apply a generating function n times to the argument list

    :param n: number of times to apply a function
    :type n: integer
    :param func: a function to apply
    :type func: instancemethod
    :rtype: collections.iterable
    """
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(n, func, *args, **kwargs):
    """Same as gapply, except returns a list

    :param n: number of times to apply a function
    :type n: integer
    :param func: a function to apply
    :type func: instancemethod
    :rtype: list
    """
    return list(gapply(n, func, *args, **kwargs))


def randset(value_range=(0, 10), sample_range=(5, 20)):
    """Return a random set of integers sampled

    :returns: a list of integers
    :rtype: tuple
    """
    n = random.choice(range(*sample_range))
    source = range(*value_range)
    return tuple(sorted(set(gapply(n, random.choice, source))))


def random_string(length, alphabet=string.ascii_letters):
    """Generate a random string

    :param length: length of the string
    :type length: int
    :param alphabet: alphabet to draw letters from
    :type alphabet: str
    :return: random string of specified length
    :rtype: str
    """
    return ''.join(str(random.choice(alphabet)) for _ in xrange(length))


def sigsim(x, y, dim):
    """Return the similarity of the two signatures

    :param x: signature 1
    :type x: object
    :param y: signature 2
    :type y: object
    :param dim: number of dimensions
    :type dim: int
    :returns: similarity between two signatures
    :rtype: float
    """
    return sum(map(operator.eq, x, y)) / float(dim)


def sort_by_length(els, reverse=True):
    """Given a list of els, sort its elements by len()
    in descending order. Returns a generator

    :param els: input list
    :type els: list
    :param reverse: Whether to reverse a list
    :type reverse: bool
    :rtype: collections.iterable
    """
    return map(itemgetter(0),
                sorted(((s, len(s)) for s in els),
                       key=operator.itemgetter(1), reverse=reverse))
