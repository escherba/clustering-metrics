"""
Augments/fixes NumPy. All code in this module is from Scikit-Learn
"""

import numpy as np


__all__ = ['bincount', 'isclose']


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)


if np_version < (1, 8, 1):
    def array_equal(a1, a2):
        # copy-paste from numpy 1.8.1
        try:
            a1, a2 = np.asarray(a1), np.asarray(a2)
        except:
            return False
        if a1.shape != a2.shape:
            return False
        return bool(np.asarray(a1 == a2).all())
else:
    from numpy import array_equal


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)


if np_version < (1, 6, 2):
    # Allow bincount to accept empty arrays
    # https://github.com/numpy/numpy/commit/40f0844846a9d7665616b142407a3d74cb65a040
    def bincount(x, weights=None, minlength=None):
        if len(x) > 0:
            return np.bincount(x, weights, minlength)
        else:
            if minlength is None:
                minlength = 0
            minlength = np.asscalar(np.asarray(minlength, dtype=np.intp))
            return np.zeros(minlength, dtype=np.intp)

else:
    from numpy import bincount


try:
    from numpy import isclose
except ImportError:
    def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
        """
        Returns a boolean array where two arrays are element-wise equal within
        a tolerance.
        This function was added to numpy v1.7.0, and the version you are
        running has been backported from numpy v1.8.1. See its documentation
        for more details.
        """
        def within_tol(x, y, atol, rtol):
            with np.errstate(invalid='ignore'):
                result = np.less_equal(abs(x - y), atol + rtol * abs(y))
            if np.isscalar(a) and np.isscalar(b):
                result = bool(result)
            return result

        x = np.array(a, copy=False, subok=True, ndmin=1)
        y = np.array(b, copy=False, subok=True, ndmin=1)
        xfin = np.isfinite(x)
        yfin = np.isfinite(y)
        if all(xfin) and all(yfin):
            return within_tol(x, y, atol, rtol)
        else:
            finite = xfin & yfin
            cond = np.zeros_like(finite, subok=True)
            # Since we're using boolean indexing, x & y must be the same shape.
            # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
            # lib.stride_tricks, though, so we can't import it here.
            x = x * np.ones_like(cond)
            y = y * np.ones_like(cond)
            # Avoid subtraction with infinite/nan values...
            cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
            # Check for equality of infinite values...
            cond[~finite] = (x[~finite] == y[~finite])
            if equal_nan:
                # Make NaN == NaN
                cond[np.isnan(x) & np.isnan(y)] = True
            return cond
