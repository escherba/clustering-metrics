"""
This module provides basic set arithmetic on graphs where graphs are
represented as collections of edges, and a few basic algorithms for graph
analysis, including an implementation of MBEA algorithm for fast finding of
maximal bicliques in bipartite graphs described in [1]_.


.. code-block:: python

    >>> from pymaptools.graph import Bigraph
    >>> g = Bigraph()
    >>> g.add_clique(([1, 2, 3], [-1, -2, -3]))
    >>> h = Bigraph(g)
    >>> g.add_clique(([4], [-4, -5]))
    >>> g.add_clique(([5], [-5, -5]))
    >>> g.add_edge(4, -1)
    >>> h.add_edge(2, 100, weight=14)
    >>> h.add_edge(5, -5, weight=10)
    >>> j = g & h
    >>> components = j.find_connected_components()
    >>> curr = components.next()
    >>> (sorted(curr.U), sorted(curr.V))
    ([1, 2, 3], [-3, -2, -1])
    >>> curr = components.next()
    >>> (sorted(curr.U), sorted(curr.V))
    ([5], [-5])


In addition to standard operations, this module is designed with the common use
case in mind when edges are assigned integer weights. One can do things like


.. code-block:: python

    >>> from pymaptools.graph import Bigraph
    >>> b = Bigraph()
    >>> b.add_edge("a", "b", 4)
    >>> b.add_edge("b", "c", 1)
    >>> b.get_weight()
    5


References
----------

.. [1] `Zhang, Y., Chesler, E. J. & Langston, M. A. "On finding bicliques in
        bipartite graphs: a novel algorithm with application to the
        integration of diverse biological data types." HICSS 0, 473+ (2008).
        <http://dx.doi.org/10.1109/HICSS.2008.507>`_
"""

import operator
from copy import deepcopy
from collections import defaultdict
from itertools import chain, product
from StringIO import StringIO
from contextlib import closing
from pymaptools.io import SimplePicklableMixin


class SkipEdge(Exception):
    """Raise to skip adding edge during misc graph operations"""
    pass


def rename_store_weight(cls):
    '''
    Class decorator to allow Bigraph objects to use
    unsorted version of _store_edge() in place of
    the sorted one while still allowing Bigraph descendants
    to override store_weight_sorted() on their own
    '''
    cls.store_weight_sorted = cls.store_weight
    return cls


@rename_store_weight
class Bigraph(SimplePicklableMixin):
    """Undirected bipartite graph G = (U & V, E)

    Note that U->V mapping is unnecessary if the only goal is to find bicliques.

    No sorting in Bigraph -- classes that inherit from Bigraph will have to
    define the store_weight_sorted() method on their own.

    Weights by default are assumed to be integers, and the default
    instances serve as edge counters.

    Example usage::

        >>> g = Bigraph()
        >>> g.add_clique(([1, 2, 3], [-1, -2, -3]))
        >>> h = Bigraph(g)
        >>> g.add_clique(([4], [-4, -5]))
        >>> g.add_clique(([5], [-5, -6]))
        >>> g.add_edge(4, -1)
        >>> h.add_edge(2, 100, weight=14)
        >>> h.add_edge(5, -5, weight=10)
        >>> j = g & h
        >>> components = j.find_connected_components()
        >>> curr = components.next()
        >>> (sorted(curr.U), sorted(curr.V))
        ([1, 2, 3], [-3, -2, -1])
        >>> curr = components.next()
        >>> (sorted(curr.U), sorted(curr.V))
        ([5], [-5])
    """
    def __init__(self, base=None, weight_type=int, min_edge_weight=None):
        self.weight_type = weight_type
        if base is None:
            # new empty object
            self.U2V = defaultdict(set)  # left to rigt mapping dict
            self.V2U = defaultdict(set)  # right to left mapping dict
            self.edges = defaultdict(weight_type)
        else:
            if not isinstance(base, self.__class__):
                raise TypeError("Base object has incorrect type")
            if weight_type is not base.edges.default_factory:
                raise ValueError("Cannot convert %s factory to %s",
                                 base.edges.default_factory, weight_type)
            if min_edge_weight is None:
                # simple copy of base object
                self.edges = deepcopy(base.edges)
                self.U2V = defaultdict(set, deepcopy(base.U2V))
                self.V2U = defaultdict(set, deepcopy(base.V2U))
            else:
                # filter out edges with weight below requested
                self.U2V = defaultdict(set)
                self.V2U = defaultdict(set)
                self.edges = defaultdict(weight_type)
                for edge, weight in base.iter_edge_weights():
                    if weight >= min_edge_weight:
                        u, v = edge
                        self.add_edge(u, v, weight=deepcopy(weight))

    @classmethod
    def from_components(cls, components):
        """Constructs a graph from a series of components
        """
        return reduce(operator.or_, components, cls())

    @classmethod
    def from_edgelist(cls, edgelist):
        """Construct a graph from a list of tuples or triples

        Tuples represent edges u - v
        Triples represent edges u - v plus weight component
        """
        g = cls()
        for edge in edgelist:
            g.add_edge(*edge)
        return g

    def iter_edges(self):
        return self.edges.iterkeys()

    def iter_edge_weights(self):
        return self.edges.iteritems()

    def rename_nodes(self, unode_renamer=None, vnode_renamer=None):
        """Factory method that produces another graph just like current one
        except with renamed nodes (can be used for reducing a graph)
        """
        new_graph = self.__class__()
        for (u, v), weight in self.iter_edge_weights():
            try:
                u = unode_renamer(u)
                v = vnode_renamer(v)
            except SkipEdge:
                continue
            new_graph.add_edge(u, v, weight)
        return new_graph

    def get_weight(self):
        return sum(self.edges.itervalues())

    def get_vnode_weight(self, node):
        neighbors = self.V2U[node]
        node_weight = self.weight_type()
        for neighbor in neighbors:
            edge = self.make_edge(neighbor, node)
            node_weight += self.edges[edge]
        return node_weight

    def get_unode_weight(self, node):
        neighbors = self.U2V[node]
        node_weight = self.weight_type()
        for neighbor in neighbors:
            edge = self.make_edge(node, neighbor)
            node_weight += self.edges[edge]
        return node_weight

    def __and__(self, other):
        '''Get intersection of edges of two graphs

        This operation chooses the minimum edge weight and is commutative
        '''
        g = self.__class__()  # create another instance of this class
        dict1, dict2 = self.edges, other.edges
        edge_intersection = set(dict1.keys()) & set(dict2.keys())
        # compile edges into dicts and store weights
        g_map_edge = g.map_edge
        g_store_weight = g.store_weight
        this_zero = self.weight_type()
        other_zero = other.weight_type()
        for e in edge_intersection:
            g_map_edge(e)
            val = min(dict1.get(e, this_zero), dict2.get(e, other_zero))
            g_store_weight(e, val)
        return g

    def __or__(self, other):
        '''Get union of edges of two graphs

        This operation involves summing edge weights and is commutative
        '''
        g = self.__class__()
        dict1, dict2 = self.edges, other.edges
        edge_union = set(dict1.keys()) | set(dict2.keys())
        # compile edges into dicts and store weights
        g_map_edge = g.map_edge
        g_store_weight = g.store_weight
        this_zero = self.weight_type()
        other_zero = other.weight_type()
        for e in edge_union:
            g_map_edge(e)
            val = dict1.get(e, this_zero) + dict2.get(e, other_zero)
            g_store_weight(e, val)
        return g

    def __sub__(self, other):
        '''Get difference of edges of two graphs (noncommutative)
        '''
        g = self.__class__()  # create another instance of this class
        dict1, dict2 = self.edges, other.edges
        edge_difference = set(dict1.keys()) - set(dict2.keys())
        # hash edges into dicts and store weights
        g_map_edge = g.map_edge
        g_store_weight = g.store_weight
        for e in edge_difference:
            g_map_edge(e)
            g_store_weight(e, dict1[e])
        return g

    def get_dot(self, name="bipartite graph", bipartite=True, unode_decorator=None,
                vnode_decorator=None, edge_decorator=None, **kwargs):
        """Get a Graphviz representation
        """
        import pygraphviz as pgv
        if unode_decorator is None:
            unode_decorator = lambda g, u: (u, {})
        if vnode_decorator is None:
            vnode_decorator = lambda g, v: (v, {})
        if edge_decorator is None:
            edge_decorator = lambda g, u, v, weight: ((u, v), {})
        g = pgv.AGraph(name=name, **kwargs)
        cluster_prefix = 'cluster_' if bipartite else ''
        sU = g.subgraph(name=cluster_prefix + "U", style="dotted")
        for node in self.U:
            node_name, attrs = unode_decorator(self, node)
            sU.add_node(node_name, **attrs)
        sV = g.subgraph(name=cluster_prefix + "V", style="dotted")
        for node in self.V:
            node_name, attrs = vnode_decorator(self, node)
            sV.add_node(node_name, **attrs)
        for edge, weight in self.iter_edge_weights():
            unode, vnode = edge
            edge, attrs = edge_decorator(self, unode, vnode, weight)
            g.add_edge(*edge, **attrs)
        return g

    def map_edge(self, edge):
        u, v = edge
        if u is None or v is None:
            raise ValueError("An edge must connect two nodes")
        self.U2V[u].add(v)
        self.V2U[v].add(u)

    def store_weight(self, edge, weight):
        self.edges[edge] += weight

    def add_clique(self, clique, weight=1):
        '''Adds a complete bipartite subgraph (a 2-clique)

        :param clique: a clique descriptor (tuple of U and V vertices)
        '''
        unodes, vnodes = clique
        for u, v in product(unodes, vnodes):
            self.add_edge(u, v, weight=weight)

    def add_edge(self, u, v, weight=1):
        '''Add a single edge (plus two vertices if necessary)

        This is a special case of add_clique(), only with
        scalar parameters. For reading data from files or adjacency
        matrices.
        '''
        edge = (u, v)
        self.map_edge(edge)
        # using "sorted" version of adding an edge -- needed
        # only for subclasses which redefine this method
        self.store_weight_sorted(edge, weight)

    @staticmethod
    def make_edge(u, v):
        return (u, v)

    def __len__(self):
        '''Number of edges in an undirected graph
        '''
        return len(self.edges)

    def __eq__(self, other):
        return self.edges == other.edges

    @property
    def U(self):
        '''a set of all "left" nodes
        '''
        return self.U2V.keys()

    @property
    def V(self):
        '''a set of all "right" nodes
        '''
        return self.V2U.keys()

    def get_density(self):
        '''Return number of existing edges divided by the number of all possible edges
        '''
        nU, nV = len(self.U), len(self.V)
        nU2V = len(self.edges)
        if nU > 0 and nV > 0:
            denominator = nU * nV
            assert nU2V <= denominator
            density = float(nU2V) / denominator
            return density
        else:
            return None

    def find_connected_components(self):
        """Return all connected components as a list of sets
        """
        stack = []
        # create a modifiable copy of the set of all vertices
        d = (
            self.U2V,
            self.V2U
        )
        remaining = set(chain(
            ((0, u) for u in self.U),
            ((1, v) for v in self.V)
        ))
        make_tuple_for = (
            (lambda x, y: (x, y)),
            (lambda x, y: (y, x))
        )
        while remaining:
            component = self.__class__()
            # pick a vertex at random and add it to the stack
            stack.append(remaining.pop())
            while stack:
                # pick an element from the stack and add it to the current component
                idx, node = stack.pop()
                # expand stack to all unvisited neighbors of the element
                # `~idx + 2` flips 0 to 1 and 1 to 0
                neighbor_part_id = ~idx + 2
                make_tuple = make_tuple_for[idx]
                for neighbor in d[idx][node]:
                    edge = make_tuple(node, neighbor)
                    edge_weight = self.edges[edge]
                    component.add_edge(edge[0], edge[1], edge_weight)
                    neighbor_tuple = (neighbor_part_id, neighbor)
                    try:
                        remaining.remove(neighbor_tuple)
                    except KeyError:
                        # vertex does not exist or has already been visited;
                        # continue with the loop
                        pass
                    else:
                        stack.append(neighbor_tuple)
            # stack is empty: done with one component
            yield component

    def find_cliques(self, L=None, P=None):
        '''Find cliques (maximally connected components)

        Enumerate all maximal bicliques in an undirected bipartite graph.
        Adapted from [1]_.

        Terminology of variables and parameters:

        = ===============================================================
        L set of vertices in U that are common neighbors of vertices in R
        R set of vertices in V belonging to the current biclique
        P set of vertices in V that can be added to R
        Q set of vertices in V that have been previously added to R
        = ===============================================================

        References
        ----------

        .. [1] `Zhang, Y., Chesler, E. J., & Langston, M. A. (2008, July). On
               Finding bicliques in bipartite graphs: a novel algorithm with
               application to the integration of diverse biological data types. In
               HICSS (p. 473). IEEE.
               <http://dx.doi.org/10.1109/HICSS.2008.507>`_
        '''
        v2u = self.V2U
        L = set(self.U) if L is None else set(L)
        P = list(self.V) if P is None else list(P)
        stack = [(L, set(), P, set())]
        while stack:
            L, R, P, Q = stack.pop()
            while P:
                x = P.pop()
                # extend biclique
                R_prime = R | {x}
                L_prime = v2u[x] & L
                # create new sets
                P_prime = []
                Q_prime = set()
                # check maximality
                is_maximal = True
                for v in Q:
                    # checks whether L_prime is a subset of all adjacent nodes
                    # of v in Q
                    Nv = v2u[v] & L_prime
                    if len(Nv) == len(L_prime):
                        is_maximal = False
                        break
                    elif Nv:
                        # some vertices in L_prime are not adjacent to v:
                        # keep vertices adjacent to some vertex in L_prime
                        Q_prime.add(v)
                if is_maximal:
                    for v in P:
                        # get the neighbors of v in L_prime
                        Nv = v2u[v] & L_prime
                        if len(Nv) == len(L_prime):
                            R_prime.add(v)
                        elif Nv:
                            # some vertices in L_prime are not adjacent to v:
                            # keep vertices adjacent to some vertex in L_prime
                            P_prime.append(v)
                    yield (L_prime, R_prime)  # report maximal biclique
                    if P_prime:
                        stack.append((L_prime, R_prime, P_prime, Q_prime))
                # move x to former candidate set
                Q.add(x)


class Graph(Bigraph):
    """
    undirected graph G = (V, E).
    """
    def __init__(self, base=None, weight_type=int):
        super(Graph, self).__init__()
        self.weight_type = weight_type
        if base is None:
            # creating from scratch
            self.U2V = self.V2U = defaultdict(set)  # right to left mapping dict
            self.edges = defaultdict(weight_type)
        elif isinstance(base, self.__class__):
            # deriving from class instance
            self.U2V = self.V2U = deepcopy(base.V2U)
            self.edges = deepcopy(base.edges)
        elif issubclass(self.__class__, base.__class__):
            # deriving from Bigraph instance
            self.V2U = deepcopy(base.V2U)
            self.V2U.update(deepcopy(base.U2V))
            self.U2V = self.V2U
            edge_map = defaultdict(weight_type)
            for (node1, node2), weight in base.edges.iteritems():
                edge_map[self.make_edge(node1, node2)] = deepcopy(weight)
            self.edges = edge_map
        else:
            raise TypeError("Base object has incorrect type")

    def rename_nodes(self, vnode_renamer=None, **kwargs):
        """Factory method that produces another graph just like current one
        except with renamed nodes (can be used for reducing a graph)
        """
        new_graph = self.__class__()
        for (u, v), weight in self.iter_edge_weights():
            u = vnode_renamer(u)
            v = vnode_renamer(v)
            new_graph.add_edge(u, v, weight)
        return new_graph

    def store_weight_sorted(self, edge, weight):
        # Storing weights is antisymmetric -- order the tuple
        self.edges[self.make_edge(*edge)] += weight

    @property
    def U(self):
        raise NotImplementedError("Set U is only for a bipartite graph")

    def add_clique(self, clique, weight=1):
        '''Adds a complete bipartite subgraph (a 2-clique)

        :param clique: a clique descriptor (a set of vertices)
        '''
        for u, v in product(clique, clique):
            self.add_edge(u, v, weight=weight)

    def get_density(self):
        '''Return the number of existing edges divided by the number of all possible edges
        '''
        nV = len(self.V)
        nV2V = 2 * len(self.edges)
        if nV > 1:
            denominator = nV * (nV - 1)
            assert nV2V <= denominator
            density = float(nV2V) / denominator
            return density
        else:
            return None

    @staticmethod
    def make_edge(u, v):
        return tuple(sorted((u, v)))

    def find_connected_components(self):
        """Return all connected components as a list of sets
        """
        stack = []
        # create a modifiable copy of the set of all vertices
        remaining = set(self.V)
        v2u = self.V2U
        while remaining:
            # pick a vertex at random and add it to the stack
            component = self.__class__()
            stack.append(remaining.pop())
            while stack:
                # pick an element from the stack and add it to the current component
                v = stack.pop()
                # expand stack to all unvisited neighbors of the element
                for u in v2u[v]:
                    edge_weight = self.edges[(u, v)]
                    component.add_edge(u, v, edge_weight)
                    try:
                        remaining.remove(u)
                    except KeyError:
                        # vertex does not exist or has already
                        # been visited; continue with the loop
                        pass
                    else:
                        stack.append(u)
            # stack is empty: done with one component
            yield component

    def get_dot(self, name="graph", edge_decorator=None, vnode_decorator=None,
                **kwargs):
        import pygraphviz as pgv
        if edge_decorator is None:
            edge_decorator = lambda g, u, v, weight: ((u, v), {})
        if vnode_decorator is None:
            vnode_decorator = lambda g, v: (v, {})
        g = pgv.AGraph(name=name, **kwargs)
        for node in self.V:
            node_name, attrs = vnode_decorator(self, node)
            g.add_node(node_name, **attrs)
        for edge, weight in self.iter_edge_weights():
            unode, vnode = edge
            edge, attrs = edge_decorator(self, unode, vnode, weight)
            g.add_edge(*edge, **attrs)
        return g

    def find_cliques(self, nodes=None, min_clique_size=3):
        '''Return maximal cliques in a graph

        Implements Bron-Kerbosch algorithm [1]_.

        .. codeauthor:: Oleksii Kuchaiev
        .. codeauthor:: Eugene Scherba

        References
        ----------

        .. [1] `Wikipedia entry for Bron-Kerbosch algorithm
               <https://en.wikipedia.org/wiki/Bron-Kerbosch_algorithm>`_
        '''
        # subset to search
        search_space = set(self.V) if nodes is None else set(nodes)
        disc_num = len(search_space)
        stack = [(set(), search_space, set(), None, disc_num)]
        v2u = self.V2U
        while stack:
            (c_compsub, c_candidates, c_not, c_nd, c_disc_num) = stack.pop()
            if not c_candidates and not c_not and len(c_compsub) >= min_clique_size:
                yield c_compsub
                continue
            for u in list(c_candidates):
                Nu = v2u[u]                             # all neighbors of node u
                if c_nd is None or c_nd not in Nu:
                    c_candidates.remove(u)
                    new_compsub = set(c_compsub)
                    new_compsub.add(u)
                    new_candidates = c_candidates & Nu  # candidates that are neighbors of node u
                    new_not = c_not & Nu                # already seen neighbors of node u
                    if c_nd is None:
                        stack.append((new_compsub, new_candidates, new_not, c_nd, c_disc_num))
                    elif c_nd in new_not:
                        new_disc_num = c_disc_num - 1
                        if new_disc_num > 0:
                            stack.append((new_compsub, new_candidates, new_not, c_nd, new_disc_num))
                    else:
                        new_disc_num = disc_num
                        new_nd = c_nd
                        for cand_nd in new_not:
                            cand_disc_num = len(new_candidates - v2u[cand_nd])
                            if cand_disc_num < new_disc_num:
                                new_disc_num = cand_disc_num
                                new_nd = cand_nd
                        stack.append((new_compsub, new_candidates, new_not, new_nd, new_disc_num))
                    c_not.add(u)
                    # find the number of candidates that are not adjacent to u
                    new_disc_num = len(c_candidates - Nu)
                    if 0 < new_disc_num < c_disc_num:
                        stack.append((c_compsub, c_candidates, c_not, u, new_disc_num))
                    else:
                        stack.append((c_compsub, c_candidates, c_not, c_nd, c_disc_num))


def describe_graph(g, graph_name=None):
    with closing(StringIO()) as sio:
        if graph_name is not None:
            print >>sio, graph_name
        print >>sio, "Edges (%d):\n\t%s\n" % (len(g.edges), g.edges)
        print >>sio, "V2U mapping (%d):\n\t%s\n" % (len(g.V2U), g.V2U)
        print >>sio, "U2V mapping (%d):\n\t%s\n" % (len(g.U2V), g.U2V)
        print >>sio, "Nodes (%d):\n\t%s\n" % (len(g.V), g.V)
        print >>sio, "Connected components:"
        for comp in g.find_connected_components():
            print >>sio, "\tComponent density: %.3f" % comp.get_density()
            print >>sio, "\tMaximal cliques:"
            for jdx, clique in enumerate(g.find_cliques(comp), start=1):
                print >>sio, "\t\t%d: %s" % (jdx, str(clique))
        return sio.getvalue()
