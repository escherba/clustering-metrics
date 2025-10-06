def assert_true(expression, msg=None):
    """helper for migrating from nosetests"""
    assert expression, msg


def assert_equal(left, right, msg=None):
    """helper for migrating from nosetests"""
    assert left == right, msg



def assert_greater(left, right, msg=None):
    """helper for migrating from nosetests"""
    assert left > right, msg
