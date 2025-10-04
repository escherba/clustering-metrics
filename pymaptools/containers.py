from functools import partial
from typing import Mapping
from collections import defaultdict, Counter
from pymaptools.iter import plen, iter_items, iter_keys, iter_vals
from pymaptools.utils import doc

OrderedDict = dict
OrderedSet = set
DefaultOrderedDict = defaultdict
OrderedCounter = partial(DefaultOrderedDict, int)

SLICE_ALL = slice(None, None, None)


class Struct(object):
    """An abstract class with namespace-like properties

    Note:

    ``collections.namedtuple`` is similar in concept but is too strict
    (requires all the keys every time you instantiate)

    ``bunch.Bunch`` object is very close to what we want, but it is not
    strict (it will not throw an error if we try to assign to it a
    a property it does not know about)::

        >>> class Duck(Struct):
        ...     readonly_attrs = frozenset(["description"])
        ...     readwrite_attrs = frozenset(["vocalization", "locomotion"])

        >>> duck = Duck(description="a medium-size bird")
        >>> duck.vocalization
        >>> duck.locomotion = "walk, swim, fly"
        >>> duck.vocalization = "quack"
        >>> duck.description = "an ostrich"
        Traceback (most recent call last):
        AttributeError: Attribute 'description' of Duck instance is read-only

        >>> duck.engine
        Traceback (most recent call last):
        AttributeError: 'Duck' object has no attribute 'engine'

        >>> duck.laden_speed = "40 mph"
        Traceback (most recent call last):
        AttributeError: Duck instance has no attribute 'laden_speed'

        >>> duck.to_dict()['vocalization']
        'quack'

        >>> another_duck = Duck.from_dict(duck.to_dict())
        >>> another_duck = Duck(engine='RD-180', **duck.to_dict())
        Traceback (most recent call last):
        AttributeError: Duck instance has no attribute 'engine'

    """
    readwrite_attrs = frozenset()
    readonly_attrs = frozenset()

    @classmethod
    def from_dict(cls, entries):
        return cls(**entries)

    def to_dict(self):
        return dict(self.__dict__)

    def __init__(self, **entries):
        self_dict = self.__dict__
        readwrite_attrs = self.readwrite_attrs
        readonly_attrs = self.readonly_attrs
        for name, value in entries.items():
            if (name in readwrite_attrs) or (name in readonly_attrs):
                self_dict[name] = value
            else:
                raise AttributeError(
                    "{} instance has no attribute '{}'".format(
                        self.__class__.__name__, name))

    def __setattr__(self, name, value):
        """Set an attribute

        Note that this also creates an attribute if one does not exist
        but is listed among the read-write attribute names
        """
        if name in self.readwrite_attrs:
            self.__dict__[name] = value
        elif name in self.readonly_attrs:
            raise AttributeError(
                "Attribute '{}' of {} instance is read-only".format(
                    name, self.__class__.__name__))
        else:
            raise AttributeError(
                "{} instance has no attribute '{}'".format(
                    self.__class__.__name__, name))

    def __getattr__(self, name):
        if name in self.readwrite_attrs or name in self.readonly_attrs:
            return None
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name))


class CrossTab(object):

    """Represents a RxC contingency table (cross-tabulation)

    A contingency table represents a cross-tabulation of either categorical
    (nominal) or ordinal variables [1]_. A contingency table is distinguished
    from the related concept of a 2D-histogram in that histogram's bins are
    drawn from a continuous distribution, while a contingency table can also
    represent discrete distributions.  Another related concept is correlation
    matrix, which is a special case of a contingency table where rows and
    columns have the same cardinality and represent two different measurements
    of the same vector of variables. To represent correlation matrices, use
    ``OrderedCrossTab`` variant.

    You can construct a dense ``CrossTab`` instance given either rows or
    columns (in a 2D-array format)::

        >>> t1 = CrossTab(rows=[(1, 5), (4, 6)])
        >>> t1.to_rows()
        [[1, 5], [4, 6]]
        >>> t1.grand_total
        16
        >>> t1[1, 0]
        4

        >>> t2 = CrossTab(cols=[(0, 1), (45, 1)])
        >>> t2.to_rows()
        [[0, 45], [1, 1]]

    You can use NumPy-like slice indexing to access entire rows/columns::

        >>> t1[0, :]
        [1, 5]
        >>> t2[:, 1]
        [45, 1]
        >>> t2[:, :]
        [[0, 45], [1, 1]]
        >>> t2[1:, :]
        [[1, 1]]

    Given mapping containers, the constructed instances of ``CrossTab``
    class will be sparse::

        >>> t3 = CrossTab(rows={'a': {'x': 2, 'y': 3}, 'b': {'x': 4}})
        >>> t3.shape
        (2, 2)
        >>> t3['a', 'x']
        2
        >>> t3['b', 'y']
        0

    Sparse CrossTabs also support slice indexing::

        >>> sorted(t3['a', :])
        [2, 3]
        >>> sorted(t3[:, 'x'])
        [2, 4]

    For missing rows or columns, a KeyError is raised::

        >>> t3['z', :]
        Traceback (most recent call last):
        KeyError: 'z'
        >>> t3[:, 'z']
        Traceback (most recent call last):
        KeyError: 'z'

    In the example above, the ``CrossTab`` instance doesn't store zero count
    for the ('b', 'y') key path, however it implicitly assumes that the count
    is zero because the 'y' column is present in ``t3.col_totals``.  Trying the
    same with a completely unknown column value 'z' results in a ``KeyError``
    exception (use standard ``get`` method if you don't want this behavior)::

        >>> t3['b', 'z']
        Traceback (most recent call last):
        KeyError: ('b', 'z')
        >>> t3.get(('b', 'z'), 0)
        0
        >>> t3.get(('a', 'x'), 0)
        2
        >>> t3['c', 'x']
        Traceback (most recent call last):
        KeyError: ('c', 'x')

    Iterating over the entire table doesn't return items corresponding to zero
    counts to gain efficiency in the case of very large and very sparse
    tables::

        >>> sorted(t3.keys())
        [('a', 'x'), ('a', 'y'), ('b', 'x')]
        >>> sorted(t3.values())
        [2, 3, 4]
        >>> sorted(t3.items())
        [(('a', 'x'), 2), (('a', 'y'), 3), (('b', 'x'), 4)]

    The length of a ``CrossTab`` is defined to be as the number of non-zero
    entries (this becomes important for efficient equality comparisons)::

        >>> list(map(len, [t1, t2, t3]))
        [4, 3, 3]
        >>> t1 != t2
        True
        >>> t2 == t2
        True
        >>> t2 == t3
        False
        >>> t3 == CrossTab(rows={'a': {'x': 2, 'y': 3}, 'b': {'x': 5}})
        False

    Note that the order of rows and columns in the "dict of dicts" case will
    not be guaranteed. To guarantee row and column order, either pass ordered
    mapping structures such as ``OrderedDict`` to the ``rows`` parameter, or,
    if relying on other constructors, use the ``OrderedCrossTab``
    specialization.

    See Also
    --------
    OrderedCrossTab, OrderedRowCrossTab, OrderedColCrossTab

    References
    ----------
    .. [1] `Wikipedia entry for Contingency Table
           <https://en.wikipedia.org/wiki/Contingency_table>`_
    """
    # TODO: use one of Scipy's sparse matrix representations instead of
    # a dict of dicts

    _col_type_1d = Counter
    _row_type_1d = Counter
    _col_type_2d = partial(defaultdict, _row_type_1d)
    _row_type_2d = partial(defaultdict, _col_type_1d)

    def __init__(self, rows=None, cols=None):
        super(CrossTab, self).__init__()
        self._rows = rows
        self._cols = cols
        self._row_totals = None
        self._col_totals = None
        self._grand_total = None

    @property
    def rows(self):
        """Table rows
        """
        _rows = self._rows
        if _rows is None:
            self._rows = _rows = self._row_type_2d()
            for cid, col in iter_items(self._cols):
                for rid, cell in iter_items(col):
                    _rows[rid][cid] = cell
        return _rows

    @property
    def cols(self):
        """Table columns
        """
        _cols = self._cols
        if _cols is None:
            self._cols = _cols = self._col_type_2d()
            for rid, row in iter_items(self._rows):
                for cid, cell in iter_items(row):
                    _cols[cid][rid] = cell
        return _cols

    @property
    def row_totals(self):
        """Row totals (right margin)

        ::

            >>> t = OrderedCrossTab.from_vals([1, 2, 3, 4, 5, 6], num_cols=2)
            >>> list(t.row_totals.values())
            [3, 7, 11]

        Ensure attribute caching::

            >>> t._row_totals[1]
            7
        """
        _row_totals = self._row_totals
        if _row_totals is None:
            self._row_totals = _row_totals = self._col_type_1d()
            for rid, row in iter_items(self.rows):
                _row_totals[rid] = sum(iter_vals(row))
        return _row_totals

    @property
    def col_totals(self):
        """Column totals (bottom margin)

        ::

            >>> t = OrderedCrossTab.from_vals([1, 2, 3, 4, 5, 6], num_cols=2)
            >>> list(t.col_totals.values())
            [9, 12]

        Ensure attribute caching::

            >>> t._col_totals[1]
            12
        """
        _col_totals = self._col_totals
        if _col_totals is None:
            self._col_totals = _col_totals = self._row_type_1d()
            for rid, col in iter_items(self.cols):
                _col_totals[rid] = sum(iter_vals(col))
        return _col_totals

    @property
    def grand_total(self):
        """Grand total of all counts
        """
        _grand_total = self._grand_total
        if _grand_total is None:
            self._grand_total = _grand_total = sum(self.iter_row_totals())
        return _grand_total

    @property
    def shape(self):
        return plen(self.iter_row_totals()), plen(self.iter_col_totals())

    def to_rows(self, rslice=SLICE_ALL, cslice=SLICE_ALL, default=0, rpad=False, cpad=False):
        """

        Example with dense matrices::

            >>> CrossTab(rows=[(1, 5), (4, 6)]).to_rows()
            [[1, 5], [4, 6]]
            >>> CrossTab(cols=[(0, 0), (45, 0)]).to_rows()
            [[0, 45], [0, 0]]

        Example with a sparse matrix::

            >>> a = [0, 2, 1, 1, 0, 3, 1, 3, 0, 1]
            >>> b = [0, 1, 1, 1, 0, 2, 1, 2, 3, 1]
            >>> t = OrderedCrossTab.from_labels(a, b)
            >>> t.to_rows()
            [[2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 4, 0], [0, 0, 0, 2]]

        Example with padding::

            >>> rows = [[10, 3, 8, 11], [9, 9, 3, 10], [9, 7, 7, 14]]
            >>> t = CrossTab(rows=rows)
            >>> t.to_rows(cpad=True)
            [[10, 3, 8, 11], [9, 9, 3, 10], [9, 7, 7, 14], [0, 0, 0, 0]]

            >>> rows = [[7, 9, 6], [10, 7, 15], [11, 11, 7], [9, 4, 4]]
            >>> t = CrossTab(rows=rows)
            >>> t.to_rows(rpad=True)
            [[7, 9, 6, 0], [10, 7, 15, 0], [11, 11, 7, 0], [9, 4, 4, 0]]

        """

        R = len(self.row_totals)
        C = len(self.col_totals)
        rpadding = [default] * ((R - C) if rpad else 0)
        cpadding = [{}] * ((C - R) if cpad else 0)

        all_rows = list(iter_vals(self.rows))
        all_rows.extend(cpadding)

        cols = list(iter_keys(self.col_totals))[cslice]

        result = []

        for row in all_rows[rslice]:
            output_row = [row.get(col, default) for col in cols] \
                if isinstance(row, Mapping) \
                else list(row)
            output_row.extend(rpadding)
            result.append(output_row)

        return result

    def to_labels(self):
        """Returns a tuple ([a], [b]). Inverse of ``from_labels``
        """
        ltrue = []
        lpred = []
        for (ri, ci), count in self.items():
            for _ in range(count):
                ltrue.append(ri)
                lpred.append(ci)
        return ltrue, lpred

    @classmethod
    def from_labels(cls, labels_true, labels_pred):
        """Instantiate from two arrays of observations (labels)
        """
        rows = cls._row_type_2d()
        for c, k in zip(labels_true, labels_pred):
            rows[c][k] += 1
        return cls(rows=rows)

    @classmethod
    def from_vals(cls, iterable, num_cols):
        """Instantiate from a reshaped iterable of values

        ::

            >>> t = CrossTab.from_vals([1, 2, 3, 4, 5, 6], num_cols=2)
            >>> t[1, 1]
            4
        """
        row_idx = 0
        rows = cls._row_type_2d()
        for idx, cell in enumerate(iterable):
            col_idx = idx % num_cols
            rows[row_idx][col_idx] = cell
            if col_idx == num_cols - 1:
                row_idx += 1
        return cls(rows=rows)

    def to_partitions(self):
        """Inverse to ``from_partitions`` constructor

        ::

            >>> p1 = [[5, 6, 7, 8], [9, 10, 11], [0, 1, 2, 3, 4]]
            >>> p2 = [[0, 1, 5, 6, 9], [2, 3, 7], [8, 10, 11], [4]]
            >>> t = OrderedCrossTab.from_partitions(p1, p2)
            >>> t.to_partitions()
            ([[5, 6, 7, 8], [9, 10, 11], [0, 1, 2, 3, 4]], [[0, 1, 5, 6, 9], [2, 3, 7], [8, 10, 11], [4]])
        """
        point = 0
        ptrue = defaultdict(list)
        ppred = defaultdict(list)
        for (ri, ci), count in self.items():
            for _ in range(count):
                ptrue[ri].append(point)
                ppred[ci].append(point)
                point += 1
        return list(ptrue.values()), list(ppred.values())

    @classmethod
    def from_partitions(cls, partitions1, partitions2):
        """Instantiate from two partitions

        A partition of N is a set of disjoint clusters s.t. every point in N
        belongs to one and only one cluster, and every cluster in N consists of
        at least one point.

        ::

            >>> p1 = [(1, 2, 3), (4, 5, 6)]
            >>> p2 = [(1, 2), (3, 4, 5), (6,)]
            >>> t = OrderedCrossTab.from_partitions(p1, p2)
            >>> t.to_labels()
            ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2])

        """
        ltrue, lpred = partitions_to_labels(partitions1, partitions2)
        return cls.from_labels(ltrue, lpred)

    def to_clusters(self):
        """Return a coded representation of clusters

        In the representations, clusters are lists, classes are integers

        ::

            >>> p1 = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
            >>> p2 = [[2, 4, 6, 8, 10], [3, 9, 12], [1, 5, 7], [11]]
            >>> t = OrderedCrossTab.from_partitions(p1, p2)
            >>> t.to_clusters()
            [[0, 0, 1, 2, 2], [0, 2, 2], [0, 1, 1], [2]]
        """
        ltrue, lpred = self.to_labels()
        return labels_to_clusters(ltrue, lpred)

    @classmethod
    def from_clusters(cls, clusters):
        """Instantiate from class-coded clustering representation

        ::

            >>> clusters = [[2, 2, 0, 0, 1], [2, 2, 0], [0, 1, 1], [2]]
            >>> t = OrderedCrossTab.from_clusters(clusters)
            >>> t.to_clusters()
            [[2, 2, 0, 0, 1], [2, 2, 0], [0, 1, 1], [2]]
        """
        ltrue = []
        lpred = []
        for k, class_labels in iter_items(clusters):
            for c in class_labels:
                ltrue.append(c)
                lpred.append(k)
        return cls.from_labels(ltrue, lpred)

    # Mapping methods

    def __contains__(self, item):
        ri, ci = item
        # a row may not contain all columns, but self.col_totals
        # always does
        return ri in self.row_totals and ci in self.col_totals

    def _column_slice(self, ci, cslice=SLICE_ALL):
        if ci not in self.col_totals:
            raise KeyError(ci)
        else:
            col = self.cols[ci]
            if isinstance(col, Mapping):
                result = []
                for ri in self.row_totals:
                    # calling ``get`` does not invoke defaultfactory
                    result.append(col.get(ri, 0))
                return result[cslice]
            else:
                return list(col[cslice])

    def _row_slice(self, ri, rslice=SLICE_ALL):
        if ri not in self.row_totals:
            raise KeyError(ri)
        else:
            row = self.rows[ri]
            if isinstance(row, Mapping):
                result = []
                for ci in self.col_totals:
                    # calling ``get`` does not invoke defaultfactory
                    result.append(row.get(ci, 0))
                return result[rslice]
            else:
                return list(row[rslice])

    def __getitem__(self, key):
        ri, ci = key
        if isinstance(ri, slice):
            if isinstance(ci, slice):
                return self.to_rows(ri, ci)
            else:
                # return values from entire column
                return self._column_slice(ci, ri)
        elif isinstance(ci, slice):
            # return values from entire row
            return self._row_slice(ri, ci)
        elif key in self:
            row = self.rows[ri]
            if isinstance(row, Mapping):
                return row.get(ci, 0)
            else:
                return row[ci]
        else:
            raise KeyError(key)

    @doc(dict.get)
    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for k, v in other.items():
            try:
                this_val = self[k]
            except KeyError:
                return False
            if v != this_val:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return plen(self.values())

    def keys(self):
        for ri, row in iter_items(self.rows):
            for ci in iter_keys(row):
                yield ri, ci

    __iter__ = keys

    def values(self):
        for row in self.iter_rows():
            for cell in row:
                yield cell

    def items(self):
        for ri, row in iter_items(self.rows):
            for ci, cell in iter_items(row):
                yield (ri, ci), cell

    # Other
    def iter_all(self):
        """Like `items()` but goes over all cells

        ::

            >>> t1 = OrderedCrossTab(rows=[(0, 5), (4, 6)])
            >>> sorted(t1.iter_all())
            [((0, 0), 0), ((0, 1), 5), ((1, 0), 4), ((1, 1), 6)]

            >>> t3 = CrossTab(rows={'a': {'x': 2, 'y': 3}, 'b': {'x': 4}})
            >>> sorted(t3.iter_all())
            [(('a', 'x'), 2), (('a', 'y'), 3), (('b', 'x'), 4), (('b', 'y'), 0)]

        """
        col_totals = self.col_totals
        for ri, row in iter_items(self.rows):
            for ci in iter_keys(col_totals):
                if isinstance(row, Mapping):
                    cell = row.get(ci, 0)
                else:
                    cell = row[ci]
                yield (ri, ci), cell

    def iter_all_with_margins(self):
        """Iterate over all cells while emitting row and column margins

        ::

            >>> t1 = OrderedCrossTab(rows=[(0, 5), (4, 6)])
            >>> sorted(t1.iter_all_with_margins())
            [(5, 4, 0), (5, 11, 5), (10, 4, 4), (10, 11, 6)]

            >>> t3 = CrossTab(rows={'a': {'x': 2, 'y': 3}, 'b': {'x': 4}})
            >>> sorted(t3.iter_all_with_margins())
            [(4, 3, 0), (4, 6, 4), (5, 3, 3), (5, 6, 2)]

            >>> sorted(t3.iter_vals_with_margins())
            [(4, 6, 4), (5, 3, 3), (5, 6, 2)]

        """
        col_totals = self.col_totals
        row_totals = self.row_totals
        for ri, row in iter_items(self.rows):
            rm = row_totals[ri]
            for ci in iter_keys(col_totals):
                if isinstance(row, Mapping):
                    cell = row.get(ci, 0)
                else:
                    cell = row[ci]
                cm = col_totals[ci]
                yield rm, cm, cell

    def iter_vals_with_margins(self):
        """Similar to `values()` except prepend row and column margins

        ::

            >>> t1 = OrderedCrossTab(rows=[(1, 5), (4, 6)])
            >>> sorted(t1.iter_vals_with_margins())
            [(6, 5, 1), (6, 11, 5), (10, 5, 4), (10, 11, 6)]

        """
        col_totals = self.col_totals
        row_totals = self.row_totals
        for ri, row in iter_items(self.rows):
            rm = row_totals[ri]
            for ci, cell in iter_items(row):
                cm = col_totals[ci]
                yield rm, cm, cell

    def iter_cols(self):
        """Iterate over table values column-wise

        ::

            >>> t1 = OrderedCrossTab(rows=[(1, 5), (4, 6)])
            >>> sorted(list(col) for col in t1.iter_cols())
            [[1, 4], [5, 6]]
        """
        for col in iter_vals(self.cols):
            yield iter_vals(col)

    def iter_rows(self):
        """Iterate over values in the right margin
        """
        for row in iter_vals(self.rows):
            yield iter_vals(row)

    def iter_col_totals(self):
        """Iterate over values in the bottom margin

        ::

            >>> t1 = OrderedCrossTab(rows=[(1, 5), (4, 6)])
            >>> sorted(t1.iter_col_totals())
            [5, 11]
        """
        return iter_vals(self.col_totals)

    def iter_row_totals(self):
        return iter_vals(self.row_totals)


class OrderedRowCrossTab(CrossTab):

    """Specialization of ``CrossTab`` for ordinal row variable

    See Also
    --------
    CrossTab
    """
    _col_type_1d = Counter
    _row_type_1d = OrderedCounter
    _col_type_2d = partial(defaultdict, _row_type_1d)
    _row_type_2d = partial(DefaultOrderedDict, _col_type_1d)


class OrderedColCrossTab(CrossTab):

    """Specialization of ``CrossTab`` for ordinal column variable

    See Also
    --------
    CrossTab
    """
    _col_type_1d = OrderedCounter
    _row_type_1d = Counter
    _col_type_2d = partial(DefaultOrderedDict, _row_type_1d)
    _row_type_2d = partial(defaultdict, _col_type_1d)


class OrderedCrossTab(CrossTab):

    """Specialization of ``CrossTab`` for ordinal variables

    See Also
    --------
    CrossTab
    """
    _col_type_1d = OrderedCounter
    _row_type_1d = OrderedCounter
    _col_type_2d = partial(DefaultOrderedDict, _row_type_1d)
    _row_type_2d = partial(DefaultOrderedDict, _col_type_1d)


def partitions_to_labels(p1, p2):

    """Jointly encode a pair of partitions as arrays of labels

        A partition of N is a set of disjoint clusters s.t. every point in N
        belongs to one and only one cluster, and every cluster in N consists of
        at least one point.

        A valid partition pair::

            >>> Y1 = [(1, 2, 3), (4, 5, 6)]
            >>> Y2 = [(1, 2), (3, 4, 5), (6,)]
            >>> partitions_to_labels(Y1, Y2)
            ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2])

        Four different examples of invalid partition pairs::

            >>> Y1 = [(1, 2, 3), (4, 5, 6, 3)]
            >>> Y2 = [(1, 2), (3, 4, 5), (6,)]
            >>> partitions_to_labels(Y1, Y2)
            Traceback (most recent call last):
            ValueError: Element '3' is in more than one cluster in p1

            >>> Y1 = [(1, 2, 3), (4, 5, 6)]
            >>> Y2 = [(1, 2), (3, 4, 5), (6, 3)]
            >>> partitions_to_labels(Y1, Y2)
            Traceback (most recent call last):
            ValueError: Element '3' is in more than one cluster in p2

            >>> Y1 = [(1, 2, 3), (4, 5, 6)]
            >>> Y2 = [(1, 2), (3, 4, 5), (6, 30)]
            >>> partitions_to_labels(Y1, Y2)
            Traceback (most recent call last):
            ValueError: Element '30' of p2 is not in p1

            >>> Y1 = [(1, 2, 3), (4, 5, 6, 30)]
            >>> Y2 = [(1, 2), (3, 4, 5), (6,)]
            >>> partitions_to_labels(Y1, Y2)
            Traceback (most recent call last):
            ValueError: 1 element(s) of p1 not in p2
    """

    a = []
    b = []

    els_to_cids_1 = {}

    for cid1, els in iter_items(p1):
        for el in els:
            if el in els_to_cids_1:
                raise ValueError("Element '%s' is in more than one cluster in p1" % el)
            else:
                els_to_cids_1[el] = cid1

    seen_p2 = set()
    for cid2, els in iter_items(p2):
        for el in els:
            if el in seen_p2:
                raise ValueError("Element '%s' is in more than one cluster in p2" % el)
            try:
                cid1 = els_to_cids_1[el]
            except KeyError:
                raise ValueError("Element '%s' of p2 is not in p1" % el)
            else:
                del els_to_cids_1[el]
            a.append(cid1)
            b.append(cid2)
            seen_p2.add(el)

    if els_to_cids_1:
        raise ValueError("%d element(s) of p1 not in p2" % len(els_to_cids_1))

    return a, b


def labels_to_clusters(labels_true, labels_pred):
    """Convert pair of label arrays to clusters of true labels

    Exact inverse of ``clusters_to_labels``::

        >>> pair = ([1, 1, 1, 1, 1, 0, 0],
        ...         [0, 0, 0, 1, 1, 2, 3])
        >>> labels_to_clusters(*pair)
        [[1, 1, 1], [1, 1], [0], [0]]

    """
    result = defaultdict(list)
    for label_true, label_pred in zip(labels_true, labels_pred):
        result[label_pred].append(label_true)
    return list(result.values())


def clusters_to_labels(iterable):
    """Convert clusters of true labels to pair of label arrays

    Exact inverse of ``labels_to_clusters``::

        >>> clusters = [[1, 1, 1], [1, 1], [0], [0]]
        >>> clusters_to_labels(clusters)
        ([1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 2, 3])

    """
    labels_true = []
    labels_pred = []
    for cluster_idx, cluster in enumerate(iterable):
        for label_true in cluster:
            labels_true.append(label_true)
            labels_pred.append(cluster_idx)
    return labels_true, labels_pred
