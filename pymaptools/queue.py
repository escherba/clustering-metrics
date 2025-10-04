from collections import Iterable
from heapq import heappush, heapreplace, nsmallest, nlargest, heappop


class Heap(Iterable):
    """A simple object-oriented interface for Python's heap queue

    Allows one to easily maintain fixed-sized heaps

    ::

        >>> h = Heap(maxlen=2)
        >>> h.add(4, "woof")
        >>> h.add(3, "meow")
        >>> h.add(10, "moo")
        (3, 'meow')
        >>> list(h)
        ['woof', 'moo']
        >>> list(reversed(h))
        ['moo', 'woof']
    """

    def __init__(self, maxlen=None):
        """
        :param maxlen: maximum number of items in heap (no limit if None)
        :type maxlen: int
        """
        self._heap = []
        self._maxlen = maxlen

    def add(self, priority, item):
        """Place an item on the heap"""
        if self._maxlen is None or len(self._heap) < self._maxlen:
            return heappush(self._heap, (priority, item))
        elif self._maxlen > 0:
            return heapreplace(self._heap, (priority, item))

    append = add

    def __len__(self):
        """Return number of elements in the heap
        :rtype: int
        """
        return len(self._heap)

    def smallest(self, n=1):
        """same as heapq.nsmallest"""
        return nsmallest(n, self._heap)

    def largest(self, n=1):
        """same as heapq.nlargest"""
        return nlargest(n, self._heap)

    def __iter__(self):
        """iterate from smallest to largest"""
        return (v for _, v in nsmallest(len(self._heap), self._heap))

    def __reversed__(self):
        """iterate from largest to smallest"""
        return (v for _, v in nlargest(len(self._heap), self._heap))


class RangeQueue(object):
    """Rank objects according to consistently-spaced index

    Sometimes an ordered stream of items becomes disordered (for example, due
    to several workers operating on it). In that case, one may still decide to
    return items in order (i.e. not wait until all items are processed).

    The idea is that the workers hopefully will not disorganize the list *too
    much* (i.e.  will not introduce large distances between former neighboring
    items). If so, one may take adavantage of this partial order and try to
    return all items in-order by caching n items (where n is hopefully small)
    until all n items form a complete "run" without any lacking items in the
    middle and thus can be retrieved at once.

    ::

        >>> queue = RangeQueue()
        >>> queue.push(1, "a")
        >>> list(queue.retrieve())
        []
        >>> queue.push(0, "b")
        >>> queue.push(2, "c")
        >>> list(queue.retrieve())
        ['b', 'a', 'c']
    """
    def __init__(self, start=0, step=1):
        self._heap = []
        self._prev_idx = start - step
        self._step = step

    def push(self, idx, value):
        """
        :param idx: index of the item
        :type idx: int
        :param value: item value
        """
        heappush(self._heap, (idx, value))

    def retrieve(self):
        """
        :rtype: generator
        """
        heap = self._heap
        step = self._step
        while heap and nsmallest(1, heap)[0][0] == self._prev_idx + step:
            self._prev_idx, value = heappop(heap)
            yield value
