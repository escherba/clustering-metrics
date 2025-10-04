import functools


def identity(x):
    return x


def compose(*functions):
    """Function composition
    """
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, identity)
