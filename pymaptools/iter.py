"""
Many definitions here are from https://docs.python.org/2/library/itertools.html
"""
from typing import Mapping, Iterator
import operator
from collections import abc, deque, defaultdict
from itertools import islice, chain, starmap, count, \
    repeat, groupby, cycle, tee, combinations
from pymaptools.func import identity, compose


def ifilterfalse(predicate, iterable):
    # ifilterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


def izip_longest(*args, **kwds):
    # izip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-
    fillvalue = kwds.get('fillvalue')
    def sentinel(counter = ([fillvalue]*(len(args)-1)).pop):
        yield counter()         # yields the fillvalue, or raises IndexError
    fillers = repeat(fillvalue)
    iters = [chain(it, sentinel(), fillers) for it in args]
    try:
        for tup in zip(*iters):
            yield tup
    except IndexError:
        pass


def get_indices(header, fields):
    return tuple(header.index(f) for f in fields)


def as_tuple(possible_tuple):
    return possible_tuple if isiterable(possible_tuple) else (possible_tuple,)


def field_getter(header, fields):
    return identity if not fields else \
        compose(as_tuple, operator.itemgetter(*get_indices(header, fields)))


def plen(iterable, predicate=bool):
    """Count values in an iterable/iterator matching predicate

    ::

        >>> g = (x for x in [0, 1, 2, 3])
        >>> plen(g)
        3
        >>> plen([0, 1, 2, 3])
        3
    """
    return sum(predicate(el) for el in iterable)


def ilen(iterable):
    """Length of an iterable/iterator

    Why doesn't this exist in itertools...

    ::

        >>> g = (x for x in [1, 2, 3])
        >>> ilen(g)
        3
        >>> ilen([1, 2, 3])
        3
    """
    if isinstance(iterable, Iterator):
        return sum(1 for _ in iterable)
    else:
        return len(iterable)


def iter_items(iterable):
    """Map item iterator that treats arrays as maps
    :type iterable: collections.Iterable
    :rtype: collections.Iterator

    ::

        >>> d = {"a": 10, "b": 20}
        >>> list(iter_items(d)) == d.items()
        True
        >>> l = ["a", "b", "c"]
        >>> list(iter_items(l))
        [(0, 'a'), (1, 'b'), (2, 'c')]
    """
    if isinstance(iterable, Mapping):
        return iterable.items()
    else:
        return enumerate(iterable)


def iter_vals(iterable):
    """Map value iterator that treats arrays as maps
    :type iterable: collections.Iterable
    :rtype: collections.Iterator

    ::

        >>> d = {"a": 10, "b": 20}
        >>> list(iter_vals(d)) == d.values()
        True
        >>> l = ["a", "b", "c"]
        >>> list(iter_vals(l))
        ['a', 'b', 'c']
        >>> list(iter_vals(iter(l)))
        ['a', 'b', 'c']
    """
    if isinstance(iterable, Mapping):
        return iterable.values()
    elif isinstance(iterable, Iterator):
        return iterable
    else:
        return iter(iterable)


def iter_keys(iterable):
    """Map key iterator that treats arrays as maps
    :type iterable: collections.Iterable
    :rtype: collections.Iterator

    ::

        >>> d = {"a": 10, "b": 20}
        >>> list(iter_keys(d)) == d.keys()
        True
        >>> l = ["a", "b", "c"]
        >>> list(iter_keys(l))
        [0, 1, 2]
        >>> list(iter_keys(iter(l)))
        [0, 1, 2]
    """
    if isinstance(iterable, Mapping):
        return iterable.keys()
    elif isinstance(iterable, Iterator):
        return (idx for idx, _ in enumerate(iterable))
    else:
        return range(len(iterable))


def iter2map(iterable):
    """Generalize maps to array types (index serves as key)

    ::

        >>> sorted(iter2map([10, 20, 30]).items())
        [(0, 10), (1, 20), (2, 30)]
        >>> sorted(iter2map(iter2map([10, 20, 30])).items())
        [(0, 10), (1, 20), (2, 30)]
    """
    if isinstance(iterable, Mapping):
        return iterable
    else:
        return dict(enumerate(iterable))


def izip_with_cycles(*args):
    """Wrapper around izip that persists scalar variables

    This is useful when you want to be very flexible with the type of
    arguments your method can accept. For example, in plotting, this allows
    to be flexible color= parameter by letting it be an array when handling
    series of data inputs or a scalar string when handling single data input.

    ::

        >>> list(izip_with_cycles(["Series_A", "Series_B"], ['red', 'blue']))
        [('Series_A', 'red'), ('Series_B', 'blue')]
        >>> list(izip_with_cycles(["Series_A", "Series_B"], 'red'))
        [('Series_A', 'red'), ('Series_B', 'red')]
        >>> list(izip_with_cycles("abc", "def"))
        [('abc', 'def')]
    """
    iargs = []
    have_iterables = False
    for arg in args:
        if isiterable(arg):
            have_iterables = True
            break
    if have_iterables:
        for arg in args:
            if isinstance(arg, Iterator):
                iargs.append(arg)
            elif isiterable(arg):
                iargs.append(iter(arg))
            else:
                iargs.append(cycle([arg]))
    else:
        # do not cycle if no iterables found (otherwise won't terminate)
        for arg in args:
            iargs.append([arg])
    return zip(*iargs)


def aggregate_tuples(iterable):
    """Aggregate a list or iterable of tuples on the first key

    Note: this works as an iterator (in sequence) and does not backtrack.
    In order to achieve complete aggregation, the input iterable must be
    presorted on the first key.

    ::

        >>> tuples = [(1, "b"), (1, "a"), (1, "c"), (3, "d")]
        >>> list(aggregate_tuples(tuples))
        [(1, ['b', 'a', 'c']), (3, ['d'])]
        >>> list(aggregate_tuples([]))
        []
    """
    if not isinstance(iterable, Iterator):
        iterable = iter(iterable)
    try:
        fst_, snd_ = next(iterable)
    except StopIteration:
        return
    bucket = [snd_]
    for fst, snd in iterable:
        if fst == fst_:
            bucket.append(snd)
        else:
            yield fst_, bucket
            fst_ = fst
            bucket = [snd]
    yield fst_, bucket


def intersperse(delimiter, seq):
    """Intersperse a sequence with a delimiter

    :param delimiter: scalar
    :type delimiter: object
    :param seq: some iterable sequence
    :type seq: collections.Iterable
    :returns: sequence interspersed with a delimiter
    :returns: collections.Iterable

    ::

        >>> list(intersperse(" ", "abc"))
        ['a', ' ', 'b', ' ', 'c']
    """
    return islice(chain.from_iterable(zip(repeat(delimiter), seq)), 1, None)


def isiterable(obj):
    """
    Are we being asked to look up a list of things, instead of a single thing?
    We check for the ``__iter__`` attribute so that this can cover types that
    don't have to be known by this module, such as NumPy arrays. We consider
    stirngs to be atomic values, not iterables.

    We don't need to check for the Python 2 ``unicode`` type, because it
    doesn't have an ``__iter__`` attribute anyway, but for the sake of
    completeness, we call isinstance on ``basestring`` type.

    This method was originally written by Luminoso Technologies [10]_.

    References
    ----------

    .. [10] `Github repository by LuminosoInsight
           <https://github.com/LuminosoInsight/ordered-set>`_
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, basestring)


def ismonotonic(oper, iterable):
    """Check whether values in iterable are monotonic according to operator

    http://en.wikipedia.org/wiki/Monotonic_function

    ::

        >>> ismonotonic(operator.le, [1, 2, 3, 3])
        True
        >>> ismonotonic(operator.le, ["a", "b", "aa"])
        False
        >>> ismonotonic(operator.ge, [4, 2])
        True
        >>> ismonotonic(operator.ge, [2, 4])
        False
    """
    if not isinstance(iterable, abc.Mapping):
        iterable = list(iterable)  # probably a generator
    return all(oper(x, y) for x, y in zip(iterable, iterable[1:]))


def pyramid_slices(lst):
    """Treat input list as a hierarchical path and return all subpaths

    ::

        >>> list(pyramid_slices([1, 2, 3]))
        [[1], [1, 2], [1, 2, 3]]
    """
    for i in range(len(lst)):
        yield lst[:i + 1]


def shinglify(iterable, span, skip=0):
    """Extract shingles from an iterable

    :param iterable: Iterable
    :type iterable: collections.Iterable
    :param span: shingle span
    :type span: int
    :param skip: How many words to skip
    :type skip: int
    :returns: sequence of tuples (shingles)
    :rtype: generator

    ::

        >>> shingles = list(shinglify("abracadabra", 5, skip=1))
        >>> len(shingles)
        7
        >>> ('d', 'b', 'a') in shingles
        True

    Must return a single shingle when span > len(tokens)::

        >>> list(shinglify("abc", 4))
        [('a', 'b', 'c')]

    Must return an empty list when span=0::

        >>> list(shinglify("abc", 0))
        []

    Must return the last pair::

        >>> list(shinglify("abcde", 4, skip=1))
        [('a', 'c'), ('b', 'd'), ('c', 'e')]

    Must also skip tokens when span > len(tokens)::

        >>> list(shinglify("abc", 4, skip=1))
        [('a', 'c')]

    For very short iterables, returns a short tuple:

        >>> list(shinglify("a", 3))
        [('a',)]

    """
    if not isinstance(iterable, abc.Mapping):
        iterable = list(iterable)  # probably a generator
    if len(iterable) >= span:
        return zip(*nskip(skip, (iterable[i:] for i in range(span))))
    else:
        return iter([tuple(nskip(skip, iterable))])


def inverse_kvals(mapping):
    """Inverse a sparse-value dense-key dict form where values are assumed to be lists

    Resulting dict is sparse-key, dense-value

    ::

        >>> {k: v for k, v in inverse_kvals({"a": [1, 2], "b": [3]})}
        {1: 'a', 2: 'a', 3: 'b'}
    """
    for key, vals in iter_items(mapping):
        for val in vals:
            yield val, key


def inverse_kvals_collect(mapping):
    """Like inverse_kvals except collects values
    """
    result = defaultdict(set)
    for k, vs in iter_items(mapping):
        for v in vs:
            result[v].add(k)
    return result


def nskip(n, iterable):
    """Skip some elements from an iterable

    :param n: How many items to skip
    :type n: int
    :param iterable: Iterable
    :type iterable: collections.Iterable
    :returns: sequence with skipped items
    :rtype: generator

    ::

        >>> list(nskip(2, "abcdefg"))
        ['a', 'd', 'g']
        >>> list(nskip(5, "abc"))
        ['a']
    """
    n_1 = n + 1
    return (v for i, v in enumerate(iterable) if not i % n_1)


def ntuples(n, iterable):
    """Tuplify an iterable

    :param n: Tuple size
    :type n: int
    :param iterable: Iterable
    :type iterable: collections.Iterable
    :returns: iterable of tuples
    :rtype: itertools.izip

    ::

        >>> list(ntuples(2, "abcde"))
        [('a', 'b'), ('c', 'd')]
        >>> list(ntuples(2, "abcd"))
        [('a', 'b'), ('c', 'd')]
    """
    if not isinstance(iterable, abc.Mapping):
        iterable = list(iterable)  # probably a generator
    return zip(*[iterable[i::n] for i in range(n)])


def take(n, iterable):
    """Return first n items of the iterable as a list

    ::

        >>> take(2, [1, 2, 3])
        [1, 2]
    """
    return list(islice(iterable, n))


def tabulate(function, start=0):
    """Return function(0), function(1), ...

    ::

        >>> foo = tabulate(lambda x: x + 5, 10)
        >>> next(foo)
        15
    """
    return map(function, count(start))


def consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely

    ::

        >>> myiter = count(5)
        >>> consume(myiter, 4)
        >>> next(myiter)
        9
        >>> myiter = iter([1, 2, 3])
        >>> consume(myiter, None)
        >>> list(myiter)
        []
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    """Returns the nth item or a default value

    ::

        >>> nth(count(5), 6)
        11
    """
    return next(islice(iterable, n, None), default)


def quantify(iterable, pred=bool):
    """Sum predicate output over a sequence

    Example with counting odd numbers::

        >>> quantify([1, 2, 3, 4, 5], pred=lambda x: x % 2)
        3
    """
    return sum(map(pred, iterable))


def padnone(iterable):
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.

    ::

        >>> take(4, padnone([1,2,3]))
        [1, 2, 3, None]
    """
    return chain(iterable, repeat(None))


def ncycles(iterable, n):
    """Returns the sequence elements n times

    ::

        >>> list(ncycles([1,2,3], 2))
        [1, 2, 3, 1, 2, 3]
    """
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    """Return a dot product of two vectors

    ::

        >>> dotproduct([1, 2, 3], [2, 3, 4])
        20
    """
    return sum(map(operator.mul, vec1, vec2))


def symmetric_diff(s1, s2):
    """Symmetric (outer) difference between sets
    """
    if not isinstance(s1, set):
        s1 = set(s1)
    if not isinstance(s2, set):
        s2 = set(s2)
    ab = s1 - s2
    ba = s2 - s1
    return (ab | ba)


def prod_dict(arr, inverse=False, identity=False):
    """Create a dict from an array of pairs of iterables s.t. the every key
    on the left side maps to every value on the right side.

    For flexibility, treat scalar values as iterables of size one.
    """
    result = defaultdict(list)
    for ks, vs in arr:
        if not isiterable(vs):
            vs = [vs]
        if not isiterable(ks):
            ks = [ks]
        fkeys, fvals = (vs, ks) if inverse else (ks, vs)
        for fkey in fkeys:
            result[fkey].extend(([fkey] + fvals) if identity else fvals)
    return result


def prodmap(d, xs):
    """Like flatmap except applies prod_dict result to array
    """
    return flatten(d.get(x, [x]) for x in xs)


def flatten(iterable):
    """Flatten one level of nesting

    :param iterable: a list of lists
    :type iterable: collections.Iterable

    ::

        >>> list(flatten([[1,2,3],[3,4,5]]))
        [1, 2, 3, 3, 4, 5]
    """
    return chain.from_iterable(iterable)


def flatmap(func, iterable):
    """Like flatMap

    ::

        >>> arr = [[1, 2], [2, 3]]
        >>> f = lambda xs: (x ** 2 for x in xs)
        >>> list(flatmap(f, arr))
        [1, 4, 4, 9]
    """
    return flatten(func(item) for item in iterable)


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example: ``repeatfunc(random.random)``

    Another example::

        >>> s = set([1, 2, 3])
        >>> list(repeatfunc(s.pop, 2))
        [1, 2]
        >>> list(repeatfunc(s.pop))
        Traceback (most recent call last):
        KeyError: 'pop from an empty set'
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def pairwise(iterable):
    """

    ::

        >>> list(pairwise([1,2,3]))
        [(1, 2), (2, 3)]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks

    ::

        >>> list(grouper('ABCDEFG', 3, 'x'))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def roundrobin(*iterables):
    """
    Recipe credited to George Sakkis

    ::

        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    pending = len(iterables)
    funs = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for fun in funs:
                yield fun()
        except StopIteration:
            pending -= 1
            funs = cycle(islice(funs, pending))


def powerset(iterable):
    """ Return a power set of an iterable

    ::

        >>> list(powerset([1,2,3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = iterable if isinstance(iterable, list) else list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable, key=None):
    """ List unique elements, preserving order. Remember all elements ever seen

    ::

        >>> list(unique_everseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D']
        >>> list(unique_everseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """List unique elements, preserving order.
    Remember only the element just seen

    ::

        >>> list(unique_justseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D', 'A', 'B']
        >>> list(unique_justseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'A', 'D']
    """
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like __builtin__.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:

    .. code-block:: python

        bsddbiter = iter_except(db.next, bsddb.error, db.first)
        heapiter = iter_except(functools.partial(heappop, h), IndexError)
        dictiter = iter_except(d.popitem, KeyError)
        dequeiter = iter_except(d.popleft, IndexError)
        queueiter = iter_except(q.get_nowait, Queue.Empty)
        setiter = iter_except(s.pop, KeyError)

    ::

        >>> s = set([1, 2, 3])
        >>> t = set([4, 5, 6])
        >>> list(iter_except(t.pop, KeyError, first=s.pop))
        [1, 4, 5, 6]
        >>> len(s)
        2
        >>> len(t)
        0
    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def nonempty(xs):
    """Filter to non-empty members

    Can be easily accomplished with filter, however this operation is common
    enough that a defined function can be more clear sometimes.
    """
    return (x for x in xs if len(x) > 0)


def first_nonempty(iterable):
    """Return first value from iterable not equal to None

    after http://stackoverflow.com/a/18533669/597371

    ::

        >>> first_nonempty([None, None, 78, None, 89])
        78
        >>> first_nonempty([]) is None
        True
    """
    try:
        return next(item for item in iterable if item is not None)
    except StopIteration:
        pass


def tee_lookahead(t, i):
    """Inspect the i-th upcoming value from a tee object
       while leaving the tee object at its current position.

    Raise an IndexError if the underlying iterator doesn't
    have enough values.

    ::

        >>> tee_obj = tee([10, 20, 30, 40])[0]
        >>> tee_lookahead(tee_obj, 2)
        30
        >>> list(tee_obj)
        [10, 20, 30, 40]
        >>> tee_lookahead(tee([])[0], 1)
        Traceback (most recent call last):
        IndexError: 1
    """
    for value in islice(t.__copy__(), i, None):
        return value
    raise IndexError(i)
