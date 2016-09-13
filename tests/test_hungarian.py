# Author: Brian M. Clapper, G. Varoquaux, Lars Buitinck
# License: BSD

import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from clustering_metrics.hungarian import linear_sum_assignment
from clustering_metrics.entropy import assignment_cost
from nose.tools import assert_equal, assert_almost_equal


def test_linear_sum_assignment():
    for cost_matrix, expected_cost in [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         [150, 400, 300]),

        # Rectangular variant
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         [150, 2, 300]),

        # Square
        ([[10, 10, 8],
          [9, 8, 1],
          [9, 7, 4]],
         [10, 1, 7]),

        # Rectangular variant
        ([[10, 10, 8, 11],
          [9, 8, 1, 1],
          [9, 7, 4, 10]],
         [10, 1, 4]),

        # n == 2, m == 0 matrix
        ([[], []],
         []),
        ]:
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert_array_equal(row_ind, np.sort(row_ind))
        assert_array_equal(expected_cost, cost_matrix[row_ind, col_ind])

        cost_matrix = cost_matrix.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assert_array_equal(row_ind, np.sort(row_ind))
        assert_array_equal(np.sort(expected_cost),
                           np.sort(cost_matrix[row_ind, col_ind]))


def test_assignment_score():
    for cost_matrix, expected_cost in [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         [150, 400, 300]),

        # Rectangular variant
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         [150, 2, 300]),

        # Square
        ([[10, 10, 8],
          [9, 8, 1],
          [9, 7, 4]],
         [10, 1, 7]),

        # Rectangular variant
        ([[10, 10, 8, 11],
          [9, 8, 1, 1],
          [9, 7, 4, 10]],
         [10, 1, 4]),

        # n == 2, m == 0 matrix
        ([[], []],
         []),
        ]:
        cost_matrix = np.array(cost_matrix)
        cost_matrix_T = cost_matrix.T

        expected_sum = np.sum(expected_cost)

        score = assignment_cost(cost_matrix)
        score_T = assignment_cost(cost_matrix_T)

        assert_equal(score, expected_sum)
        assert_equal(score_T, expected_sum)

        score_dbl = assignment_cost(np.asarray(cost_matrix, dtype=float))
        score_T_dbl = assignment_cost(np.asarray(cost_matrix_T, dtype=float))

        assert_almost_equal(score_dbl, float(expected_sum))
        assert_almost_equal(score_T_dbl, float(expected_sum))


def test_linear_sum_assignment_input_validation():
    assert_raises(ValueError, linear_sum_assignment, [1, 2, 3])

    C = [[1, 2, 3], [4, 5, 6]]
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(np.asarray(C)))
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(np.matrix(C)))
