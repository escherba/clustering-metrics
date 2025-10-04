#cython: infer_types=True

import copy
from itertools import imap as _imap
from operator import eq as _eq
from collections import Callable
from pymaptools.iter import isiterable

from pymaptools._cyordereddict cimport OrderedDict

from cpython.object cimport PyObject_RichCompare, Py_EQ, Py_NE


SLICE_ALL = slice(None)


cdef class OrderedSet(set):
    """
    An OrderedSet is an object complying to MutableSet interface and which
    remembers its order, so that every entry has an index that can be looked
    up.

    This version is based on version written by Luminoso Technologies [11]_
    with the following changes: this class uses OrderedDict as storage and
    supports key removal. We drop support for indexing, and add support for
    fixed-size sets with maxlen parameter.

    With a small modification, this class can be made into an LRU cache.

    References
    ----------

    .. [11] `Github repository by LuminosoInsight
           <https://github.com/LuminosoInsight/ordered-set>`_
    """

    # Note: a pure Python version of this class originally inherited from
    # collections.MutableSet and collections.Sequence

    cdef object _map
    cdef object _maxlen

    def __init__(self, iterable=None, maxlen=None):
        self._map = OrderedDict()
        self._maxlen = maxlen
        if iterable is not None:
            self.__ior__(iterable)

    @classmethod
    def _from_iterable(self, it):
        return OrderedSet(it)

    # non-prefixed operators return a new object

    def __or__(self, other):
        s = OrderedSet(self)
        return s.__ior__(other)

    def __and__(self, other):
        s = OrderedSet(self)
        return s.__iand__(other)

    def __xor__(self, other):
        s = OrderedSet(self)
        return s.__ixor__(other)

    def __sub__(self, other):
        s = OrderedSet(self)
        return s.__isub__(other)

    # i-prefixed operators are on self object

    def __ior__(self, other):
        for key in other:
            self.add(key)
        return self

    def __iand__(self, other):
        kept = set()
        for key in other:
            if key in self:
                kept.add(key)
        for key in self:
            if key not in kept:
                self.remove(key)
        return self

    def __ixor__(self, other):
        added = set()
        removed = set()
        for key in other:
            if key in self:
                removed.add(key)
            else:
                added.add(key)
        for key in removed:
            self.remove(key)
        for key in added:
            self.add(key)
        return self

    def __isub__(self, other):
        removed = set()
        for key in other:
            if key in self:
                removed.add(key)
        for key in removed:
            self.remove(key)
        return self

    # other methods

    def clear(self):
        self._map.clear()

    def __len__(self):
        return len(self._map)

    def __getitem__(self, index):
        """
        Get the item at a given index.

        If `index` is a slice, you will get back that slice of items. If it's
        the slice [:], exactly the same object is returned. (If you want an
        independent copy of an OrderedSet, use `OrderedSet.copy()`.)

        If `index` is an iterable, you'll get the OrderedSet of items
        corresponding to those indices. This is similar to NumPy's
        "fancy indexing".
        """
        if index == SLICE_ALL:
            return self
        elif hasattr(index, '__index__') or isinstance(index, slice):
            result = self._map.keys()[index]
            if isinstance(result, list):
                return OrderedSet(result)
            else:
                return result
        elif isiterable(index):
            keys = self._map.keys()
            return OrderedSet([keys[i] for i in index])
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" % index)

    def copy(self):
        return OrderedSet(self)

    def __getstate__(self):
        if len(self) == 0:
            # The state can't be an empty list.
            # We need to return a truthy value, or else __setstate__ won't be run.
            #
            # This could have been done more gracefully by always putting the state
            # in a tuple, but this way is backwards- and forwards- compatible with
            # previous versions of OrderedSet.
            return (None,)
        else:
            return list(self)

    def __setstate__(self, state):
        if state == (None,):
            self.__init__([])
        else:
            self.__init__(state)

    def __contains__(self, key):
        return key in self._map

    def add(self, key):
        maxlen = self._maxlen
        mapping = self._map
        if key not in mapping:
            if maxlen is None or len(mapping) < maxlen:
                mapping[key] = 1
            else:
                mapping.popitem(last=False)
                mapping[key] = 1

    def update(self, iterable):
        for item in iterable:
            self.add(item)

    append = add

    def discard(self, key):
        # unlike ``remove``, ``discard`` does not raise KeyError
        # in case of missing key
        mapping  = self._map
        if key in mapping:
            self._map.__delitem__(key)

    def remove(self, key):
        self._map.__delitem__(key)

    def __iter__(self):
        return self._map.iterkeys()

    def __reversed__(self):
        return reversed(self._map.keys())

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def _eq__(self, other):
        if isinstance(other, OrderedSet):
            return set.__eq__(self, other) and all(_imap(_eq, self, other))
        return set.__eq__(self, other)

    def _ne__(self, other):
        return not self == other

    def __richcmp__(self, other, int op):
        if op == Py_EQ:
            return self._eq__(other)
        if op == Py_NE:
            return self._ne__(other)
        return PyObject_RichCompare(id(self), id(other), op)


cdef class DefaultOrderedDict(OrderedDict):
    """Ordered dict with default constructors

    Attribution: http://stackoverflow.com/a/6190500/562769
    """

    cdef object default_factory

    def __init__(self, default_factory=None, *args, **kwargs):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *args, **kwargs)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items(), memo=memo))
