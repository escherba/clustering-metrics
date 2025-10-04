import collections
import uuid
import sys
from copy import deepcopy
from contextlib import contextmanager


def doc(s):
    """Decorator to extract docstrings from existing methods

    Example usage:

    .. code-block:: python

        @doc(dict.__getitem__)
        def __getitem__(self, item):
            ...
    """
    if hasattr(s, '__call__'):
        s = s.__doc__

    def f(g):
        g.__doc__ = s
        return g
    return f


class SetComparisonMixin(object):
    """Mixin to ``unittest.TestCase`` to provide subset comparisons
    """
    def assertSetContainsSubset(self, expected, actual, msg=None):
        """Checks whether actual is a superset of expected."""
        self.assertListEqual(
            [],
            sorted(set(expected) - set(actual)),
            msg=msg
        )

    def assertSetDoesNotContainSubset(self, not_expected, actual, msg=None):
        """Expected should not be a part of actual."""
        self.assertListEqual(
            sorted(set(not_expected)),
            sorted(set(not_expected) - set(actual)),
            msg=msg
        )


def uuid1_to_posix(uuid1):
    """Convert a UUID1 timestamp to a standard POSIX timestamp

    ::

        >>> uuid1_to_posix("d64736cf-5bfa-11e4-a292-542696da2c01")
        1414209362.290043
    """
    uuid1 = uuid.UUID(uuid1)
    if uuid1.version != 1:
        raise ValueError('only applies to UUID type 1')
    return (uuid1.time - 0x01b21dd213814000) / 1e7


def deepupdate(dest, source):
    """Recursively update one dict with contents of another

    :param dest: mapping being updated
    :type dest: dict
    :param source: mapping to update with
    :type source: collections.Mapping
    :return: updated mapping
    :rtype: dict
    """
    for key, value in source.iteritems():
        dest[key] = deepupdate(dest.get(key, {}), value) \
            if isinstance(value, collections.Mapping) \
            else value
    return dest


def override(parent, child):
    """Inherit child from parent and return a new object
    """
    return deepupdate(deepcopy(parent), child)


@contextmanager
def empty_context(*args, **kwargs):
    """Generic empty context wrapper
    """
    yield None


@contextmanager
def joint_context(*args):
    """Generic empty context wrapper

    Allows constructions like:

    .. code-block:: python

        with joint_context(open("foo.txt", "r")) as fhandle:
            ...
        with joint_context(open("foo.txt", "r"), open("bar.txt", "r")) as (fh1, fh2):
            ...
    """
    try:
        yield args[0] if len(args) == 1 else args
    finally:
        for arg in args:
            if arg is not sys.stdout:
                arg.close()
