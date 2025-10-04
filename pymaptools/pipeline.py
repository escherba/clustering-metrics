"""
``Pipe`` is a basic pipeline for processing data in sequence. You create pipes
by composing ``Step`` instances (or any callables). ``Pipe`` makes extensive
use of generators to make processing memory-efficient. A basic example:

.. code-block:: python

    >>> def deserialize(obj):
    ...     yield int(obj)
    ...
    >>> def square(obj):
    ...     yield obj * obj
    ...
    >>> class Sum(Step):
    ...     def __init__(self):
    ...         self.sum = 0
    ...     def __call__(self, obj):
    ...         self.sum += obj
    ...
    >>> sumsq = Pipe([deserialize, square, Sum()])
    >>> sumsq.run(["1", "2", "3"])
    >>> sumsq.steps[-1].sum
    14

A more complex example:

.. code-block:: python

    import json
    import sys
    from pymaptools.pipeline import Pipe, Step

    def deserialize(obj):
        # demonstrate use of plain functions as callables
        # demonstrate multiple outputs
        #
        try:
            array = json.loads(obj)["x"]
            for num in array:
                yield int(num)
        except:
            print "failed to deserialize `{}`".format(obj)

    def filter_even(obj):
        # demonstrate that values can be dropped
        if obj % 2 == 0:
            yield obj

    class Add(Step):
        # demonstrate use of state
        #
        def __init__(self, value):
            self.value = value

        def __call__(self, obj):
            yield obj + self.value

    class MultiplyBy(Step):
        def __init__(self, value):
            self.value = value

        def __call__(self, obj):
            yield obj * self.value

    class Output(Step):
        # demonstrate that we can use IO
        #
        def __init__(self, handle):
            self.handle = handle

        def __call__(self, obj):
            self.handle.write(str(obj) + "\\n")


    # process a sequence of possible JSON strings
    input_seq = ['{"x":[0,-6,4]}', '{"x":[12]}', '{"x":[34]}', '{"x":[-9]}',
                "Ceci n'est pas une pipe", '{"x":[4]}']
    pipe = Pipe([
        deserialize,
        filter_even,
        Add(10),
        MultiplyBy(2),
        Output(sys.stdout)
    ])
    pipe.run(input_seq)

When run, the output of the above is:

.. code-block:: python

    20
    8
    28
    44
    88
    failed to deserialize `Ceci n\'est pas une pipe`
    28

"""

import abc
import contextlib
import itertools as it
from functools import partial


class Step(object):
    """
    base class for steps (for use with ``Pipe``)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, obj):
        """Process one piece of content"""

    def __enter__(self):
        """Enter context"""
        return self

    def __exit__(self, obj_type, obj_value, traceback):
        """Exit context (tear down)"""
        self.on_exit()

    def on_exit(self):
        """callback when pipeline is done or error occured"""
        pass


class StepWrapper(Step):
    """A wrapper for callables and objects not derived from ``Step``
    """
    def __init__(self, fun):
        if not callable(fun):
            raise ValueError("Cannot wrap non-callable object")
        self._callable = fun

    def __call__(self, obj):
        return self._callable(obj)


class Pipe(object):

    """Applies a series of steps to an iterator.

    When given an array of step objects, composes them into a pipe.

    ::

        >>> def deserialize(obj):
        ...     yield int(obj)
        ...
        >>> def square(obj):
        ...     yield obj * obj
        ...
        >>> class Sum(Step):
        ...     def __init__(self):
        ...         self.sum = 0
        ...     def __call__(self, obj):
        ...         self.sum += obj
        ...
        >>> sumsq = Pipe([deserialize, square, Sum()])
        >>> sumsq.run(["1", "2", "3"])
        >>> sumsq.steps[-1].sum
        14

    Parameters
    ----------

    steps : collecitons.Iterable
        A sequence of ``Step`` instances or callables
    """

    def __init__(self, steps):
        steps_with_exit = []
        for step in steps:
            if not isinstance(step, Step):
                step = StepWrapper(step)
            steps_with_exit.append(step)
        self.steps = steps_with_exit

    @staticmethod
    def apply_step(step, obj):
        """Apply step and return an empty list if result is not iterable
        """
        result = step(obj)
        return result if hasattr(result, '__iter__') else []

    def apply_steps(self, obj):
        """Runs all the steps on a single object
        """
        results = [obj]
        for step in self.steps:
            apply_step = partial(self.apply_step, step)
            # note: some exceptions may not be caught if imap is used
            # instead of map here below:
            results = list(it.chain(*map(apply_step, results)))
        for result in results:
            yield result

    def run(self, input_iter):
        """Runs all the steps on input iterator
        """
        with contextlib.nested(*self.steps) as entered_steps:
            for i, step in enumerate(entered_steps):
                self.steps[i] = step
            for obj in input_iter:
                for _ in self.apply_steps(obj):
                    pass
