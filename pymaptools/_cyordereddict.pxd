cdef class OrderedDict(dict):
    cdef dict __map
    cdef list __root
