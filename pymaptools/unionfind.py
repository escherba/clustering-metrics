"""
``UnionFind`` is a method for creating, maintainig, and retrieving disjoint
clusters from a graph. An example:

.. code-block:: python

    >>> uf = UnionFind()
    >>> uf.union(0, 1)
    >>> uf.union(2, 3)
    >>> uf.union(3, 0)
    >>> uf.union(4, 5)
    >>> uf.sets()
    [[0, 1, 2, 3], [4, 5]]

"""

from collections import defaultdict


class UnionFind(object):
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

    ::

        >>> uf = UnionFind()
        >>> uf.union(0, 1)
        >>> uf.union(2, 3)
        >>> uf.union(3, 0)
        >>> uf.union(4, 5)
        >>> uf.sets()
        [[0, 1, 2, 3], [4, 5]]

    Source code based on [1]_ and [2]_, with modifications.

    References
    ----------

    .. [1] `Source code by D. Eppstein
        <http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py>`_
    .. [2] `Source code by J. Carlson
        <http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912>`_

    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, obj):
        """Find and return the representative of the set containing obj.
        :rtype: object
        """
        # check for previously unknown obj
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj

        # find path of objects leading to the root
        path = [obj]
        last_root = obj
        root = self.parents[obj]
        while root != last_root:
            path.append(root)
            last_root = root
            root = self.parents[root]

        # compress the path
        for ancestor in path:
            self.parents[ancestor] = root

        # return the representative
        return root

    def __iter__(self):
        """Iterate through all items ever found or union-ed by this structure

        :rtype: collections.iterable
        """
        return iter(self.parents)

    def union(self, *objs):
        """Find the sets containing the objects and merge them all."""
        weights = self.weights
        parents = self.parents
        found_roots = map(self.__getitem__, objs)
        heaviest_root = max(found_roots, key=weights.__getitem__)
        for root in found_roots:
            if root != heaviest_root:
                weights[heaviest_root] += weights[root]
                parents[root] = heaviest_root

    def sets(self):
        """Return a list of each disjoint set
        :rtype: list
        """
        result = defaultdict(list)
        for element in self.parents.iterkeys():
            result[self[element]].append(element)
        return result.values()

    def num_neighbors(self, obj):
        """Return the number of objects in the cluster
        containing given object
        """
        return self.weights[self[obj]] - 1
