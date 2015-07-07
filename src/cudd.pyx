# cython: profile=True
"""Cython interface to CUDD.


Reference
=========
    Fabio Somenzi
    "CUDD: CU Decision Diagram Package"
    University of Colorado at Boulder
    v2.5.1, 2015
    http://vlsi.colorado.edu/~fabio/
"""
import logging
import pprint
import sys
from libcpp cimport bool
from libc.stdio cimport FILE, fdopen, fopen
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from 'cudd.h':
    # node
    ctypedef unsigned int DdHalfWord
    cdef struct DdNode:
        DdHalfWord index
        DdHalfWord ref
    ctypedef DdNode DdNode
    # manager
    cdef struct DdManager:
        pass
    ctypedef DdManager DdManager
    cdef DdManager * Cudd_Init(
        unsigned int numVars,
        unsigned int numVarsZ,
        unsigned int numSlots,
        unsigned int cacheSize,
        unsigned long maxMemory)
    ctypedef enum Cudd_ReorderingType:
        pass
    # node elements
    cdef DdNode * Cudd_bddNewVar(DdManager *dd)
    cdef DdNode * Cudd_bddNewVarAtLevel(DdManager *dd, int level)
    cdef DdNode * Cudd_bddIthVar(DdManager *dd, int i)
    cdef DdNode * Cudd_ReadLogicZero(DdManager *dd)
    cdef DdNode * Cudd_ReadOne(DdManager *dd)
    cdef DdNode * Cudd_Regular(DdNode *u)
    cdef bool Cudd_IsConstant(DdNode *u)
    cdef DdNode * Cudd_T(DdNode *u)
    cdef DdNode * Cudd_E(DdNode *u)
    # basic Boolean operators
    cdef DdNode * Cudd_Not(DdNode *dd)
    cdef DdNode * Cudd_bddIte(DdManager *dd, DdNode *f,
                              DdNode *g, DdNode *h)
    cdef DdNode * Cudd_bddAnd(DdManager *dd,
                              DdNode *dd, DdNode *dd)
    cdef DdNode * Cudd_bddOr(DdManager *dd,
                             DdNode *dd, DdNode *dd)
    cdef DdNode * Cudd_bddXor(DdManager *dd,
                              DdNode *f, DdNode *g)
    cdef DdNode * Cudd_Support(DdManager *dd, DdNode *f)
    cdef DdNode * Cudd_bddComputeCube(
        DdManager *dd, DdNode **vars, int *phase, int n)
    cdef int Cudd_PrintMinterm(DdManager *dd, DdNode *f)
    # refs
    cdef void Cudd_Ref(DdNode *n)
    cdef void Cudd_RecursiveDeref(DdManager *table,
                                  DdNode *n)
    cdef void Cudd_Deref(DdNode *n)
    # checks
    cdef int Cudd_CheckZeroRef(DdManager *manager)
    cdef int Cudd_DebugCheck(DdManager *table)
    cdef void Cudd_Quit(DdManager *unique)
    cdef DdNode * Cudd_bddTransfer(
        DdManager *ddSource, DdManager *ddDestination, DdNode *f)
    # info
    cdef int Cudd_PrintInfo(DdManager *dd, FILE *fp)
    cdef int Cudd_ReadSize(DdManager *dd)
    cdef long Cudd_ReadNodeCount(DdManager *dd)
    cdef long Cudd_ReadPeakNodeCount(DdManager *dd)
    cdef int Cudd_ReadPeakLiveNodeCount(DdManager * dd)
    cdef unsigned long Cudd_ReadMemoryInUse(DdManager *dd)
    cdef unsigned int Cudd_ReadReorderings(DdManager *dd)
    cdef long Cudd_ReadReorderingTime(DdManager * dd)
    cdef int Cudd_ReadPerm(DdManager *dd, int i)
    cdef int Cudd_ReadInvPerm(DdManager *dd, int i)
    cdef int Cudd_DagSize(DdNode *node)
    # manager config
    cdef void Cudd_SetMaxCacheHard(DdManager *dd, unsigned int mc)
    cdef void Cudd_AutodynEnable(DdManager *unique,
                                 Cudd_ReorderingType method)
    cdef void Cudd_SetMaxGrowth(DdManager *dd, double mg)
    cdef void Cudd_SetMinHit(DdManager *dd, unsigned int hr)
    # quantification
    cdef DdNode * Cudd_bddExistAbstract(
        DdManager *manager, DdNode *f, DdNode *cube)
    cdef DdNode * Cudd_bddUnivAbstract(
        DdManager *manager, DdNode *f, DdNode *cube)
    cdef DdNode * Cudd_bddAndAbstract(
        DdManager *manager, DdNode *f, DdNode *g, DdNode *cube)
    cdef DdNode * Cudd_bddSwapVariables(
        DdManager *dd,
        DdNode *f, DdNode **x, DdNode **y, int n)
CUDD_UNIQUE_SLOTS = 256
CUDD_CACHE_SLOTS = 262144
CUDD_REORDER_GROUP_SIFT = 14
MAX_CACHE = <unsigned int> -1


cdef extern from 'dddmp.h':
    ctypedef enum Dddmp_VarInfoType:
        pass
    cdef int Dddmp_cuddBddStore(
        DdManager *ddMgr, char *ddname, DdNode *f,
        char **varnames, int *auxids, int mode,
        Dddmp_VarInfoType varinfo, char *fname, FILE *fp)
DDDMP_MODE_TEXT = <int>'A'
DDDMP_VARIDS = 0


logger = logging.getLogger(__name__)


cdef class BDD(object):
    """Wrapper of CUDD manager.

    Interface similar to `dd.bdd.BDD`.
    Variable names are strings.
    Attributes:

      - `vars`
    """

    cdef DdManager * manager
    cpdef public object vars
    cpdef public object _index_of_var
    cpdef public object _var_with_index

    def __cinit__(self, memory=None):
        """Initialize BDD manager.

        @param memory: maximum allowed memory, in GB.
        """
        if memory is None:
            memory = 2 * 1024**3
        mgr = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS,
                        CUDD_CACHE_SLOTS, memory)
        Cudd_SetMaxCacheHard(mgr, MAX_CACHE)
        Cudd_AutodynEnable(mgr, CUDD_REORDER_GROUP_SIFT)
        Cudd_SetMaxGrowth(mgr, 1.2)
        Cudd_SetMinHit(mgr, 1)
        self.manager = mgr
        self.vars = set()
        self._index_of_var = dict()  # map: str -> unique fixed int
        self._var_with_index = dict()

    def __dealloc__(self):
        n = Cudd_CheckZeroRef(self.manager)
        assert n == 0, (
            'Still {n} nodes '
            'referenced upon shutdown.').format(n=n)
        Cudd_Quit(self.manager)

    def __str__(self):
        n = Cudd_ReadNodeCount(self.manager)
        peak = Cudd_ReadPeakLiveNodeCount(self.manager)
        n_vars = Cudd_ReadSize(self.manager)
        reordering_time = Cudd_ReadReorderingTime(self.manager)
        reordering_time = reordering_time / 1000.0
        n_reorderings = Cudd_ReadReorderings(self.manager)
        mem = Cudd_ReadMemoryInUse(self.manager)
        mem = float(mem) / 10**6
        s = (
            'Binary decision diagram (CUDD wrapper) with:\n'
            '\t {n} live nodes now\n'
            '\t {peak} live nodes at peak\n'
            '\t {n_vars} BDD variables\n'
            '\t {mem:10.1f} MB in use\n'
            '\t {reorder_t:10.1f} sec spent reordering\n'
            '\t {n_reorder} reorderings\n').format(
                n=n, peak=peak, n_vars=n_vars,
                reorder_t=reordering_time,
                n_reorder=n_reorderings,
                mem=mem)
        return s

    cdef incref(self, DdNode * u):
        Cudd_Ref(u)

    cdef decref(self, DdNode * u, recursive=True):
        if recursive:
            Cudd_RecursiveDeref(self.manager, u)
        else:
            Cudd_Deref(u)

    property False:

        def __get__(self):
            return self._bool(False)

    property True:

        def __get__(self):
            return self._bool(True)

    def _bool(self, v):
        """Return terminal node for Boolean `v`."""
        cdef DdNode * r
        if v:
            r = Cudd_ReadOne(self.manager)
        else:
            r = Cudd_ReadLogicZero(self.manager)
        f = Function()
        f.init(self.manager, r)
        return f

    cpdef add_var(self, var, index=None):
        """Return index of variable named `var`.

        If a variable named `var` exists,
        the assert that it has `index`.
        Otherwise, create a variable named `var`
        with `index` (if given).

        If no reordering has yet occurred,
        then the returned index equals the level,
        provided `add_var` has been used so far.
        """
        # var already exists ?
        j = self._index_of_var.get(var)
        if j is not None:
            assert j == index, (j, index)
            return j
        # new var
        if index is None:
            j = len(self._index_of_var)
        else:
            j = index
        u = Cudd_bddIthVar(self.manager, j)
        assert u != NULL, 'failed to add var "{v}"'.format(v=var)
        self._add_var(var, j)
        return j

    cpdef insert_var(self, var, level):
        """Create a new variable named `var`, at `level`."""
        cdef DdNode * r
        r = Cudd_bddNewVarAtLevel(self.manager, level)
        assert r != NULL, 'failed to create var "{v}"'.format(v=var)
        j = r.index
        self._add_var(var, j)
        return j

    cdef _add_var(self, var, index):
        self.vars.add(var)
        self._index_of_var[var] = index
        self._var_with_index[index] = var
        assert (len(self._index_of_var) ==
            len(self._var_with_index))
        return index

    cpdef Function var(self, var):
        """Return node for variable named `var`."""
        assert var in self._index_of_var, (
            'undefined variable "{v}", '
            'known variables are:\n {d}').format(
                v=var, d=self._index_of_var)
        j = self._index_of_var[var]
        r = Cudd_bddIthVar(self.manager, j)
        f = Function()
        f.init(self.manager, r)
        return f

    def var_at_level(self, level):
        """Return name of variable at `level`."""
        j = Cudd_ReadInvPerm(self.manager, level)
        # no var there yet ?
        if j == -1:
            return None
        assert j in self._var_with_index, (j, self._var_with_index)
        var = self._var_with_index[j]
        return var

    def level_of_var(self, var):
        """Return level of variable named `var`."""
        assert var in self._index_of_var, (
            'undefined variable "{v}", '
            'known variables are:\n {d}').format(
                v=var, d=self._index_of_var)
        j = self._index_of_var[var]
        level = Cudd_ReadPerm(self.manager, j)
        return level

    cpdef Function apply(self, op, Function u, Function v=None):
        """Return as `Function` the result of applying `op`."""
        # TODO: add ite, also to slugsin syntax
        assert self.manager == u.manager
        cdef DdNode * r
        cdef DdManager * mgr
        mgr = u.manager
        # unary
        r = NULL
        if op in ('!', 'not'):
            assert v is None
            r = Cudd_Not(u.node)
        else:
            assert v is not None
            assert u.manager == v.manager
        # binary
        if op in ('&', 'and'):
            r = Cudd_bddAnd(mgr, u.node, v.node)
        elif op in ('|', 'or'):
            r = Cudd_bddOr(mgr, u.node, v.node)
        elif op in ('^', 'xor'):
            r = Cudd_bddXor(mgr, u.node, v.node)
        if r == NULL:
            raise Exception(
                'unknown operator: "{op}"'.format(op=op))
        f = Function()
        f.init(mgr, r)
        return f

    cpdef Function quantify(self, Function u,
                            qvars, forall=False):
        """Abstract variables `qvars` from node `u`."""
        cdef DdManager * mgr = u.manager
        cube = self.cube(mgr, qvars)
        # quantify
        if forall:
            r = Cudd_bddUnivAbstract(mgr, u.node, cube.node)
        else:
            r = Cudd_bddExistAbstract(mgr, u.node, cube.node)
        # wrap
        f = Function()
        f.init(mgr, r)
        return f

    cdef Function cube(self, DdManager * manager, dvars):
        """Return node for cube over `dvars`.

        Only positive unate cubes implemented for now.
        """
        n = len(dvars)
        # make cube
        cdef DdNode * cube
        cdef DdNode **x
        x = <DdNode **> PyMem_Malloc(n * sizeof(DdNode *))
        for i, var in enumerate(dvars):
            f = self.var(var)
            x[i] = f.node
        try:
            cube = Cudd_bddComputeCube(manager, x, NULL, n)
        finally:
            PyMem_Free(x)
        f = Function()
        f.init(manager, cube)
        return f

    cpdef assert_consistent(self):
        assert Cudd_DebugCheck(self.manager) == 0


cpdef Function and_abstract(Function u, Function v, qvars, BDD bdd):
    """Return `? qvars. u & v`."""
    assert u.manager == v.manager
    mgr = u.manager
    cube = bdd.cube(mgr, qvars)
    r = Cudd_bddAndAbstract(u.manager, u.node, v.node, cube.node)
    f = Function()
    f.init(mgr, r)
    return f


cpdef Function or_abstract(Function u, Function v, qvars, BDD bdd):
    """Return `! qvars. u | v`."""
    assert u.manager == v.manager
    mgr = u.manager
    cube = bdd.cube(mgr, qvars)
    cdef DdNode * r
    r = Cudd_bddAndAbstract(
        u.manager, Cudd_Not(u.node), Cudd_Not(v.node), cube.node)
    r = Cudd_Not(r)
    f = Function()
    f.init(mgr, r)
    return f


cpdef Function rename(Function u, bdd, dvars):
    """Return node `u` after renaming variables in `dvars`."""
    n = len(dvars)
    cdef DdNode **x = <DdNode **> PyMem_Malloc(n * sizeof(DdNode *))
    cdef DdNode **y = <DdNode **> PyMem_Malloc(n * sizeof(DdNode *))
    cdef DdNode * r
    cdef DdManager * mgr = u.manager
    cdef Function f
    for i, xvar in enumerate(dvars):
        yvar = dvars[xvar]
        f = bdd.var(xvar)
        x[i] = f.node
        f = bdd.var(yvar)
        y[i] = f.node
    try:
        r = Cudd_bddSwapVariables(
            mgr, u.node, x, y, n)
    finally:
        PyMem_Free(x)
        PyMem_Free(y)
    f = Function()
    f.init(mgr, r)
    return f


def support(Function f, bdd, as_str=True):
    """Return support of node `f`."""
    mgr = f.manager
    u = f.node
    supp = set()
    _support(mgr, u, supp)
    id_to_var = {v: k for k, v in bdd._index_of_var.iteritems()}
    if as_str:
        supp = map(id_to_var.get, supp)
    return supp


cdef _support(DdManager * mgr, DdNode * u, set supp):
    """Recurse to collect indices of support variables."""
    # TODO: use cache
    # terminal ?
    if Cudd_IsConstant(u):
        return
    # add var
    r = Cudd_Regular(u)
    supp.add(r.index)
    # recurse
    v = Cudd_E(u)
    w = Cudd_T(u)
    _support(mgr, v, supp)
    _support(mgr, w, supp)


cpdef transfer_bdd(Function u, BDD bdd):
    """Transfer the node `u` to `bdd`."""
    logger.debug('++ transfer bdd')
    assert u.manager != bdd.manager
    r = Cudd_bddTransfer(u.manager, bdd.manager, u.node)
    f = Function()
    f.init(bdd.manager, r)
    logger.debug('-- done transfering bdd')
    return f


cdef print_info(DdManager *mgr, f=None):
    cdef FILE *cf
    if f is None:
        f = sys.stdout
    cf = fdopen(f.fileno(), 'w')
    Cudd_PrintInfo(mgr, cf)


cdef dump(Function u, BDD bdd, fname):
    """Dump BDD as DDDMP file `fname`."""
    n = len(bdd._index_of_var)
    cdef FILE * f
    cdef char **names
    cdef int *indices
    names = <char **> PyMem_Malloc(n * sizeof(char *))
    indices = <int *> PyMem_Malloc(n * sizeof(int))
    for i, (var, idx) in enumerate(bdd._index_of_var.iteritems()):
        names[i] = var
        indices[i] = idx
    try:
        f = fopen(fname, 'w')
        Dddmp_cuddBddStore(
            bdd.manager, NULL, u.node,
            names, indices, DDDMP_MODE_TEXT,
            DDDMP_VARIDS, NULL, f)
    finally:
        PyMem_Free(names)
        PyMem_Free(indices)


cdef class Function(object):
    """Wrapper of `DdNode` from CUDD.

    Use as:

    ```
    bdd = BDD()
    cdef DdNode * u
    u = Cudd_ReadOne(bdd.manager)
    f = Function()
    f.init(bdd.manager, u)
    h = g | ~ f
    ```
    """

    cdef object __weakref__
    cpdef DdManager * manager
    cpdef DdNode * node

    cdef init(self, DdManager * mgr, DdNode * u):
        if u == NULL:
            raise Exception('`DdNode *u` is `NULL` pointer.')
        self.manager = mgr
        self.node = u
        Cudd_Ref(u)

    property index:

        def __get__(self):
            return self.node.index

    def __dealloc__(self):
        Cudd_RecursiveDeref(self.manager, self.node)

    def __str__(self):
        cdef DdNode * u
        u = Cudd_Regular(self.node)
        return (
            'Function(DdNode with: '
            'var_index={idx}, '
            'ref_count={ref})').format(
                idx=u.index,
                ref=u.ref)

    def __len__(self):
        return Cudd_DagSize(self.node)

    def __richcmp__(Function self, Function other, op):
        if other is None:
            eq = False
        else:
            # guard against mixing managers
            assert self.manager == other.manager
            eq = (self.node == other.node)
        if op == 2:
            return eq
        elif op == 3:
            return not eq
        else:
            raise Exception('Only __eq__ and __ne__ defined.')

    def __invert__(self):
        cdef DdNode * r
        r = Cudd_Not(self.node)
        f = Function()
        f.init(self.manager, r)
        return f

    def __and__(Function self, Function other):
        assert self.manager == other.manager
        r = Cudd_bddAnd(self.manager, self.node, other.node)
        f = Function()
        f.init(self.manager, r)
        return f

    def __or__(Function self, Function other):
        assert self.manager == other.manager
        r = Cudd_bddOr(self.manager, self.node, other.node)
        f = Function()
        f.init(self.manager, r)
        return f


'''
def main():
    cdef DdManager *mgr
    cdef DdNode *u
    cdef DdNode *v
    cdef DdNode *x0
    cdef DdNode *x1
    cdef DdNode *x2
    cdef DdNode *x3

    mgr = Cudd_Init(
        0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS,
        16 * 1024UL * 1024UL * 1024UL)
    u = Cudd_ReadOne(mgr)
    print('True: ', u.index)
    v = Cudd_ReadLogicZero(mgr)
    print('False: ', v.index)
    # Cudd_RecursiveDeref(mgr, u)
    # Cudd_RecursiveDeref(mgr, v)

    # add some variables, and store them in an array

    x0 = Cudd_bddNewVar(mgr)
    x1 = Cudd_bddNewVar(mgr)
    x2 = Cudd_bddNewVar(mgr)
    x3 = Cudd_bddNewVar(mgr)

    cdef DdNode * x[2]
    cdef DdNode * y[2]
    x[0] = x0
    x[1] = x1
    y[0] = x2
    y[1] = x3

    u = Cudd_bddAnd(mgr, x0, x2)
    Cudd_Ref(u)
    Cudd_bddSwapVariables(mgr, u, x, y, 2)

    # print_info(mgr)
    fu = Function()
    fu.init(mgr, u)
    mgr = fu.manager
    fv = Function()
    fv.init(mgr, v)
    f = _bdd_apply('and', fu, fv)
    print(f)
    del f, fu, fv
    Cudd_RecursiveDeref(mgr, u)

    bdd = BDD()
    bdd.manager = mgr
    bdd._index_of_var = dict(x=0, y=1)
    print(bdd._index_of_var)
    s = '& | 0 1 x'
    f = add_expr(s, bdd)
    print(f)
    u = Cudd_ReadOne(mgr)
    Cudd_Ref(u)
    print(u.index)
    Cudd_RecursiveDeref(mgr, u)
    del f

    print_info(mgr)

    n = Cudd_CheckZeroRef(mgr)
    assert n == 0, n
    Cudd_Quit(mgr)


def test():
    # main()
    cdef BDD bdd
    cdef Function u
    bdd = BDD()
    fname = 'slugs.txt'
    d = load_slugsin_file(fname)
    # pprint.pprint(d)
    dvars = add_variables(d, bdd)
    print(dvars)
    # pprint.pprint(dvars)
    u = add_expr(d['[SYS_INIT]'][0], bdd)
    suffix = "'"
    rename = {k: k + suffix for k in d['[OUTPUT]']}
    print(rename)
    cdef Function f
    supp = support(u, bdd)
    print('support before rename:', supp)
    f = _bdd_rename(u, bdd, rename)
    supp = support(f, bdd)
    print('support after rename:', supp)
    # f = Function()
    # f.init(bdd.manager, u.node)
    # qvars = {'ex_sys_0'}
    # r = bdd.quantify(f, qvars, forall=False)
    # print(support(r, bdd))
'''
