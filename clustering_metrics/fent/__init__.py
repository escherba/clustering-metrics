"""

Experiment with F2PY


Useful links:

* http://www.engr.ucsb.edu/~shell/che210d/f2py.pdf
* http://aero-comlab.stanford.edu/jmartins/doc/python9-slides.pdf
* http://docs.scipy.org/doc/numpy/user/c-info.python-as-glue.html
* http://docs.scipy.org/doc/numpy/f2py/
* http://www.ucs.cam.ac.uk/docs/course-notes/unix-courses/archived/archived-python-courses/pythonfortran/files/f2py.pdf


Steps to compile::

    f2py -h _fent.pyf -m _fent _fent.f90 --overwrite-signature
    python setup.py build_ext --inplace

Numpy v0.10.1 has a bug where you have to edit the output of f2py
above and modify the PYF signature file adding ``intent(out)`` as
needed.
"""

from ._fent import minmaxr, emi_from_margins

__all__ = ["minmaxr", "emi_from_margins"]
