# cython: profile=True
"""Cython interface to BuDDy.


Reference
=========
    Jorn Lind-Nielsen
    "BuDDy: Binary Decision Diagram package"
    IT-University of Copenhagen (ITU)
    v2.4, 2002
    http://buddy.sourceforge.net
"""
import logging
import pprint
import sys
from libcpp cimport bool
from libc.stdio cimport FILE, fdopen, fopen
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from 'bdd.h':
    # BDD
    ctypedef int BDD
    # renaming pair
    cdef struct s_bddPair:
        pass
    ctypedef s_bddPair bddPair
    cdef int bdd_init(int BDDsize, int cachesize)
    cdef int bdd_isrunning()
    cdef void bdd_done()
    cdef int bdd_setacheratio(int r)
    # variable creation
    cdef BDD bdd_ithvar(int var)
    cdef BDD bdd_nithvar(int var)
    cdef int bdd_var2level(int var)
    cdef int bdd_level2var(int level)
    cdef int bdd_setvarnum(int num)
    cdef int bdd_extvarnum(int num)
    cdef int bdd_varnum()
    # variable manipulation
    cdef int bdd_var(BDD r)
    cdef BDD bdd_makeset(int *varset, int varnum)
    cdef int bdd_scanset(BDD a, int **varset, int *varnum)
    cdef BDD bdd_ibuildcube(int value, int width, int *var)
    cdef BDD bdd_buildcube(int value, int width, BDD *var)
    # BDD elements
    cdef BDD bdd_true()
    cdef BDD bdd_false()
    cdef BDD bdd_low(BDD r)
    cdef BDD bdd_high(BDD r)
    cdef BDD bdd_support(BDD r)
    cdef BDD bdd_satone(BDD r)  # cube
    cdef BDD bdd_fullsatone(BDD r)  # minterm
    cdef double bdd_satcount(BDD r)
    cdef int bdd_BDDcount(BDD r)
    # refs
    cdef BDD bdd_addref(BDD r)
    cdef BDD bdd_delref(BDD r)
    cdef void bdd_gbc()
    # basic Boolean operators
    cdef bdd_ite(BDD u, BDD v, BDD w)
    cdef bdd_apply(BDD u, BDD w, int op)
    cdef BDD bdd_not(BDD u)
    cdef BDD bdd_and(BDD u, BDD v)
    cdef BDD bdd_or(BDD u, BDD v)
    cdef BDD bdd_xor(BDD u, BDD v)
    cdef BDD bdd_imp(BDD u, BDD v)
    cdef BDD bdd_biimp(BDD u, BDD v)
    # composition operators
    cdef BDD bdd_restrict(BDD r, BDD var)
    cdef BDD bdd_constrain(BDD f, BDD c)
    cdef BDD bdd_compose(BDD f, BDD g, BDD v)
    cdef bdd_simplify(BDD f, BDD d)
    # quantification
    cdef BDD bdd_exist(BDD r, BDD var)
    cdef BDD bdd_forall(BDD r, BDD var)
    cdef BDD bdd_appex(BDD u, BDD v, int op, BDD var)
    cdef BDD bdd_appall(BDD u, BDD v, int op, BDD var)
    # renaming
    cdef BDD bdd_replace(BDD r, bddPair *pair)
    cdef int bdd_setpair(bddPair *pair, int oldvar, int newvar)
    cdef int bdd_setpairs(bddPair *pair, int *oldvar,
                          int *newvar, int size)
    cdef void bdd_resetpair(bddPair *pair)
    cdef void bdd_freepair(bddPair *p)
    # manager config
    cdef int bdd_setmaxBDDnum(int size)
    cdef int bdd_setmaxincrease(int size)
    cdef int bdd_setminfreeBDDs(int mf)
    cdef int bdd_getBDDnum()
    cdef int bdd_getallocnum()  # both dead and active
    # reordering
    cdef void bdd_reorder(int method)
    cdef int bdd_autoreorder(int method)
    cdef int bdd_autoreorder_times(int method, int num)
    cdef void bdd_enable_reorder()
    cdef void bdd_disable_reorder()
    cdef int bdd_reorder_gain()
    # I/O
    cdef int bdd_save(FILE *ofile, BDD r)
    cdef int bdd_load(FILE *ifile, BDD r)
    # info
    cdef int bdd_reorder_verbose(int value)
    cdef void bdd_printorder()
    cdef void bdd_fprintorder(FILE *ofile)
    # cdef void bdd_stats(bddStat *stat)
    # cdef void bdd_cachestats(bddCacheStat *s)
    cdef void bdd_fprintstat(FILE *f)
    cdef void bdd_printstat()
APPLY_MAP = {
    'and': 0, 'xor': 1, 'or': 2, 'nand': 3, 'nor': 4,
    'imp': 5, 'biimp': 6, 'diff': 7, 'less': 8, 'invimp': 9}
BDD_REORDER_WIN2 = 1
BDD_REORDER_SIFT = 13


logger = logging.getLogger(__name__)


cdef class BuddyBDD(object):
    """Wrapper of BuDDy.

    Interface similar to `dd.bdd.BDD`.
    There is only a single global shared BDD,
    so use only one instance.
    """

    cpdef public object var_to_index

    def __cinit__(self):
        self.var_to_index = dict()
        if bdd_isrunning():
            return
        n = 10**6
        cache = 10**4
        n_vars = 250
        bdd_init(n, cache)
        bdd_setvarnum(n_vars)

    def __dealloc__(self):
        bdd_done()

    def __str__(self):
        bdd_printstat()
        return ''

    cdef incref(self, int u):
        bdd_addref(u)

    cdef decref(self, int u):
        bdd_delref(u)

    property False:

        def __get__(self):
            return self._bool(False)

    property True:

        def __get__(self):
            return self._bool(True)

    cdef _bool(self, b):
        if b:
            r = bdd_true()
        else:
            r = bdd_false()
        return Function(r)

    cpdef add_var(self, var):
        """Return index for variable `var`.

        @type var: `str`
        @rtype: `int`
        """
        j = self.var_to_index.get(var)
        if j is not None:
            return j
        j = len(self.var_to_index)
        self.var_to_index[var] = j
        return j

    cpdef var(self, var):
        """Return BDD for variable `var`.

        @type var: `str`
        @rtype: `Function`
        """
        assert var in self.var_to_index, (
            var, self.var_to_index)
        j = self.var_to_index[var]
        cdef int r = bdd_ithvar(j)
        assert r != self.False, 'failed'
        return Function(r)

    cpdef level(self, var):
        """Return level of variable `var`.

        @type var: `str`
        """
        j = self.add_var(var)
        level = bdd_var2level(j)
        return level

    cpdef at_level(self, level):
        level = bdd_level2var(level)
        index_to_var = {
            v: k for k, v in self.var_to_index.iteritems()}
        j = index_to_var[level]
        return j

    cpdef apply(self, op, u, v=None):
        """Return as `Function` the result of applying `op`."""
        # unary
        if op in ('!', 'not'):
            assert v is None
            r = bdd_not(u.BDD)
        else:
            assert v is not None
        # binary
        if op in ('&', 'and'):
            r = bdd_and(u.BDD, v.BDD)
        elif op in ('|', 'or'):
            r = bdd_or(u.BDD, v.BDD)
        elif op in ('^', 'xor'):
            r = bdd_xor(u.BDD, v.BDD)
        return Function(r)

    cpdef quantify(self, u, qvars, forall=False):
        cube = self.cube(qvars)
        if forall:
            r = bdd_forall(u, cube)
        else:
            r = bdd_exist(u, cube)
        return Function(r)

    cdef cube(self, dvars):
        """Return a positive unate cube for `dvars`."""
        n = len(dvars)
        cdef int *x
        x = <int *> PyMem_Malloc(n * sizeof(int *))
        for i, var in enumerate(dvars):
            j = self.add_var(var)
            x[i] = j
        try:
            r = bdd_makeset(x, n)
        finally:
            PyMem_Free(x)
        return Function(r)


cpdef and_abstract(u, v, qvars, bdd):
    cube = bdd.cube(qvars)
    op = APPLY_MAP['and']
    r = bdd_appall(u, v, op, cube)
    return Function(r)


cpdef or_abstract(u, v, qvars, bdd):
    cube = bdd.cube(qvars)
    op = APPLY_MAP['or']
    r = bdd_appex(u, v, op, bdd)
    return Function(r)


def rename(u, dvars, bdd):
    n = len(dvars)
    cdef int *oldvars
    cdef int *newvars
    oldvars = <int *> PyMem_Malloc(n * sizeof(int *))
    newvars = <int *> PyMem_Malloc(n * sizeof(int *))
    for i, (a, b) in enumerate(dvars.iteritems()):
        ja = bdd.add_var(a)
        jb = bdd.add_var(b)
        oldvars[i] = ja
        oldvars[i] = jb
    cdef bddPair *pair = NULL
    try:
        bdd_setpairs(pair, oldvars, newvars, n)
        r = bdd_replace(u, pair)
    finally:
        PyMem_Free(oldvars)
        PyMem_Free(newvars)
    return Function(r)


cdef class Function(object):

    cdef object __weakref__
    cpdef public int node

    def __cinit__(self, node):
        self.node = node
        bdd_addref(node)

    def __dealloc__(self):
        bdd_delref(self.node)

    def __str__(self):
        return 'Function({u})'.format(u=self.node)

    def __richcmp__(self, other, op):
        if other is None:
            eq = False
        else:
            eq = (self.node == other.node)
        if op == 2:
            return eq
        elif op == 3:
            return not eq
        else:
            raise Exception('Only `==` and `!=` defined.')

    def __invert__(self):
        return Function(bdd_not(self.node))

    def __and__(self, other):
        return Function(bdd_and(self.node, other.node))

    def __or__(self, other):
        return Function(bdd_or(self.node, other.node))
