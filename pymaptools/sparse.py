import numpy as np
from functools import partial
from itertools import izip
from tqdm import tqdm
from scipy.sparse import coo_matrix
from pymaptools.containers import DefaultOrderedDict
from collections import defaultdict


def dd2coo(dd, dtype=np.float32):
    """
    Convert a dict of dicts to COO-type sparse matrix
    """
    # first create maps from key values to rows and columns

    # first level
    row_list = dd.keys()
    col_list = list(set(k for row in dd.itervalues() for k in row.iterkeys()))
    row_map = {k: idx for idx, k in enumerate(row_list)}
    col_map = {k: idx for idx, k in enumerate(col_list)}

    values = []
    row_indices = []
    col_indices = []
    for row_key, row in dd.iteritems():
        row_idx = row_map[row_key]
        for col_key, val in row.iteritems():
            values.append(val)
            row_indices.append(row_idx)
            col_indices.append(col_map[col_key])

    return row_list, col_list, coo_matrix((values, (row_indices, col_indices)), dtype=dtype)


def csr2dd(mat, transpose=False, show_progress=False):
    """Convert a CSR sparse matrix to dict-of-dicts format
    """
    dd = defaultdict(partial(dict))
    for i, j, v in iter_csr(mat, transpose=transpose, show_progress=show_progress):
        dd[i][j] = v
    return dd


class CooBuilder(object):

    """A simple demonstration of dd2coo usage
    """
    def __init__(self, dtype):

        self._dtype = dtype
        self._dict = DefaultOrderedDict(partial(DefaultOrderedDict, dtype))

    def add(self, x, y, val):

        self._dict[x][y] += val

    def assign(self, x, y, val):

        self._dict[x][y] = val

    def get_coo(self, transpose=False):

        rows, cols, mat = dd2coo(self._dict, dtype=self._dtype)
        if transpose:
            return cols, rows, mat.T
        else:
            return rows, cols, mat


def iter_csr(mat, transpose=False, show_progress=False):
    """Iterate over CSR matrix in memory-efficient way
    """
    indices = mat.nonzero()
    data = mat.data
    assert indices[0].shape[0] == indices[1].shape[0] == data.shape[0]
    li, ri = (1, 0) if transpose else (0, 1)
    iterator = izip(indices[li], indices[ri], data)
    if show_progress:
        iterator = tqdm(iterator, total=mat.nnz)
    return iterator
