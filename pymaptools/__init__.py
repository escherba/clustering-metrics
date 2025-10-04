__version__ = '0.2.35'

from collections import defaultdict
from functools import partial, reduce
from pymaptools.func import compose


def amap(f, g):
    """Lift a scalar transformer f into an iterable

    :param f: transformer of type a -> b where a, b are scalars
    :type f: function
    :param g: mapper of type L x (returns an iterator)
    :type g: function
    :return: an iterable over domain of g with f mapped over range of g
    :rtype: function

    >>> def foo(x):
    ...     for i in range(x):
    ...         yield i
    >>> bar = amap(lambda x: x + 10, foo)
    >>> list(bar(5))
    [10, 11, 12, 13, 14]

    """
    return compose(partial(map, f), g)


def all_equal(xs):
    """check that all elements of a list are equal
    :param xs: a sequence of elements
    :type xs: list, str
    :rtype: bool

    >>> all_equal("aaa")
    True
    >>> all_equal("abc")
    False
    >>> all_equal(["a", "b"])
    False
    >>> all_equal([1] * 10)
    True
    """
    return not xs or xs.count(xs[0]) == len(xs)


def uniq(tokens):
    """Analog of UNIX command uniq

    :param tokens: an iterable of hashable tokens
    :type tokens: collections.Iterable
    :return: a generator of unique tokens
    :rtype: generator

    >>> list(uniq([1, 1, 2, 3, 2, 4]))
    [1, 2, 3, 4]
    """
    seen = set()
    for token in tokens:
        if token not in seen:
            yield token
            seen.add(token)


def uniq_replace(tokens, placeholder=None):
    """Same as uniq except replace duplicate tokens with placeholder

    :param tokens: an iterable of hashable tokens
    :type tokens: collections.Iterable
    :param placeholder: some object
    :type placeholder: object
    :return: a generator of unique tokens
    :rtype: generator

    >>> list(uniq_replace([1, 1, 2, 3, 2, 4], 'x'))
    [1, 'x', 2, 3, 'x', 4]
    """
    seen = set()
    for token in tokens:
        if token in seen:
            yield placeholder
        else:
            yield token
            seen.add(token)


def nested_get(root, keys, strict=False):
    """Get value from a nested dict using keys (a list of keys)
    :param root: root dictionary
    :type root: dict
    :param keys: a list of keys
    :type keys: list
    :param strict: whether to throw KeyError on missing key
    :type strict: bool

    >>> example = {"a": {"b": 1, "c": 42}, "d": None}
    >>> nested_get(example, ["a", "c"])
    42
    >>> example = {}
    >>> nested_get(example, ["abc"], strict=False) is None
    True
    """
    try:
        result = reduce(dict.__getitem__, keys, root)
    except (KeyError, TypeError):
        if strict:
            raise
        else:
            result = None
    return result


def nested_set(root, keys, value, strict=False):
    """Set value in a nested dict using a keys
    :param root: root dictionary
    :type root: dict
    :param keys: a list of keys
    :type keys: list
    :param value: a value to set
    :type value: obj
    :param strict: whether to throw KeyError on intermediate levels
    :type create: bool

    >>> example = {"a": {"b": 1, "c": 42}, "d": None}
    >>> nested_set(example, ["a", "c"], None)
    >>> example['a']['c']
    >>> example = {}
    >>> nested_set(example, ["a", "b", "c"], 56)
    >>> example['a']['b']['c']
    56
    >>> nested_set(example, [], 42)
    42
    """
    if len(keys) == 0:
        return value
    if strict:
        nested_get(root, keys[:-1], strict=strict)[keys[-1]] = value
    else:
        curr_dict = root
        for key in keys[:-1]:
            if key in curr_dict:
                curr_dict = curr_dict[key]
            else:
                new_dict = dict()
                curr_dict[key] = new_dict
                curr_dict = new_dict
        curr_dict[keys[-1]] = value


def nested_type(levels=0, constructor=dict):
    """Constructor for nested dict types

    Recursive implementation given here

    >>> ndtype = nested_type(2)
    >>> d = ndtype()
    >>> d[1][2][3] = 4
    >>> d[1][2][3]
    4
    """
    if levels == 0:
        return constructor
    else:
        return partial(defaultdict, nested_type(levels - 1, constructor=constructor))


def excise(lst, idx):
    """Return a new list with a particular element index removed

    >>> lst = range(0, 10)
    >>> excise(lst, 0)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return list(lst[:idx]) + list(lst[idx + 1:])
