"""
the get_class_attrs and get_object_attrs methods below
are based on http://stackoverflow.com/a/10313703/597371
"""
from pymaptools import uniq


def get_class_attrs(klass):
    """Get attributes of a class
    """
    ret = dir(klass)
    if hasattr(klass, '__bases__'):
        for base in klass.__bases__:
            ret.extend(get_class_attrs(base))
    return ret


def get_object_attrs(obj):
    """Get attributes of an object
    """
    ret = dir(obj)
    if hasattr(obj, '__class__'):
        ret.append('__class__')
        ret.extend(get_class_attrs(obj.__class__))
    return list(uniq(ret))


def hasmethod(obj, method):
    """check whether object has a method
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


def iter_methods(obj, names=None, private=False):
    """Return all methods from list of names supported by object

    ::

        >>> from pymaptools.iter import first_nonempty
        >>> class Foo(object):
        ...     def bar(self):
        ...         return "bar called"
        >>> foo = Foo()
        >>> method = first_nonempty(iter_methods(foo, ['missing', 'bar']))
        >>> hasmethod(foo, method.__name__)
        True
        >>> method()
        'bar called'
    """
    if names is None:
        names = dir(obj)
    if not private:
        names = [name for name in names if not name.startswith('_')]
    for method_name in names:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                yield method


def iter_method_names(obj, names=None, private=False):
    """Return all methods from list of names supported by object

    ::

        >>> from pymaptools.iter import first_nonempty
        >>> class Foo(object):
        ...     def bar(self):
        ...         return "bar called"
        >>> foo = Foo()
        >>> method = first_nonempty(iter_method_names(foo, ['missing', 'bar']))
        >>> hasmethod(foo, method)
        True
    """
    if names is None:
        names = dir(obj)
    if not private:
        names = [name for name in names if not name.startswith('_')]
    for method_name in names:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                yield method_name
