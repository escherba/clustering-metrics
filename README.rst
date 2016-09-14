Clustering Metrics
==================

A Python implementation of various metrics (primarily external) used for clustering evaluation. The documentation is `available online here <https://escherba.github.io/clustering-metrics/>`_.

Motivation
----------

After creating an in-memory representation of a clustering or a partition, many common metrics can be calculated
very cheaply. The efficiency of the computation depends primarily on the in-memory representation of clustering.
Sparse representations are pefect for this purpose and allow us to calculate many metrics more efficiently than
packages like Scikit-Learn.

Installation
------------

At the moment, the package is not on PyPI. To install it, use ``pip`` like so:

.. code-block:: bash

   pip install git+https://github.com/escherba/pymaptools#egg=pymaptools-0.2.30
   pip install git+https://github.com/escherba/clustering-metrics#egg=clustering_metrics-0.0.1

Usage
-----

Clusters can be represented in different ways. One way is to enumerate all items in the cluster with integer labels:

.. code-block:: python

   >>> ground_truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
   >>> predicted = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2]

Note that ``ground_truth`` and ``predicted`` must have the same length. We can then produce various metrics
as follows:

.. code-block:: python

   >>> from clustering_metrics.metrics import ClusteringMetrics
   >>> cm = ClusteringMetrics.from_labels(ground_truth, predicted)
   >>> cm.adjusted_rand_index()
   0.242914979757085


Another way to represent clusters is using partition-style encoding. Here, each clustering is represented
as a set of partitions:

.. code-block:: python

   >>> ground_truth = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
   >>> predicted = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10, 11, 12}, {13, 14, 15, 16}]
   >>> cm = ClusteringMetrics.from_partitions(ground_truth, predicted)
   >>> cm.split_join_distance(normalize=False)
   4

Development
-----------

For development and testing, this package sets up a Python virtualenv under ``./env/``
relative to the source tree root.

.. code-block:: bash

   git clone https://github.com/escherba/clustering-metrics.git
   cd clustering-metrics
   make test


License
-------

This package is under a BSD 3-clause license.
