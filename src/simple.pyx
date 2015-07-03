"""Cython interface to CUDD."""
import logging
import math
import pprint
import sys
import time
from libcpp cimport bool
from libc.stdio cimport FILE, fdopen, fopen
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from omega.symbolic.bdd import Nodes as _Nodes
from omega.symbolic.bdd import Parser
from omega.symbolic.symbolic import Automaton


cdef extern from 'cudd.h':
    ctypedef unsigned int DdHalfWord
    cdef struct DdNode:
        DdHalfWord index
        DdHalfWord ref
    ctypedef DdNode DdNode
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
    cdef DdNode * Cudd_bddNewVar(DdManager *dd)
    cdef DdNode * Cudd_bddIthVar(DdManager *dd, int i)
    cdef DdNode * Cudd_ReadLogicZero(DdManager *dd)
    cdef DdNode * Cudd_ReadOne(DdManager *dd)
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

    cdef DdNode * Cudd_Regular(DdNode *u)
    cdef bool Cudd_IsConstant(DdNode *u)
    cdef DdNode * Cudd_T(DdNode *u)
    cdef DdNode * Cudd_E(DdNode *u)

    cdef void Cudd_Ref(DdNode *n)
    cdef void Cudd_RecursiveDeref(DdManager *table,
                                  DdNode *n)
    cdef void Cudd_Deref(DdNode *n)
    cdef int Cudd_CheckZeroRef(DdManager *manager)
    cdef int Cudd_DebugCheck(DdManager *table)
    cdef void Cudd_Quit(DdManager *unique)

    cdef int Cudd_PrintInfo(DdManager *dd, FILE *fp)
    cdef int Cudd_ReadSize(DdManager *dd)
    cdef long Cudd_ReadNodeCount(DdManager *dd)
    cdef long Cudd_ReadPeakNodeCount(DdManager *dd)
    cdef long Cudd_ReadReorderingTime(DdManager * dd)
    cdef int Cudd_DagSize(DdNode *node)

    cdef void Cudd_SetMaxCacheHard(DdManager *dd, unsigned int mc)
    cdef void Cudd_AutodynEnable(DdManager *unique,
                                 Cudd_ReorderingType method)
    cdef void Cudd_SetMaxGrowth(DdManager *dd, double mg)
    cdef void Cudd_SetMinHit(DdManager *dd, unsigned int hr)

    cdef DdNode * Cudd_bddExistAbstract(
        DdManager *manager, DdNode *f, DdNode *cube)
    cdef DdNode * Cudd_bddUnivAbstract(
        DdManager *manager, DdNode *f, DdNode *cube)
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


# TODO: use relational product
# TODO: group primed and unprimed vars
# TODO: use efficient rename for neighbors
# also: try with multiple managers,
# in a sequential program,
# to observe the effect of decoupling the variable order


cdef class BDD(object):
    """Wrapper of CUDD manager.

    Interface similar to `dd.bdd.BDD`.
    Attributes:

    - `var_to_index`: maps each variable to a unique fixed integer
    """

    cdef DdManager * manager
    cpdef public object var_to_index

    def __cinit__(self):
        mgr = Cudd_Init(
            0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS,
            2 * 1024UL * 1024UL * 1024UL)
        Cudd_SetMaxCacheHard(mgr, MAX_CACHE)
        Cudd_AutodynEnable(mgr, CUDD_REORDER_GROUP_SIFT)
        Cudd_SetMaxGrowth(mgr, 1.2)
        Cudd_SetMinHit(mgr, 1)
        self.manager = mgr
        self.var_to_index = dict()

    def __dealloc__(self):
        n = Cudd_CheckZeroRef(self.manager)
        assert n == 0, (
            'Still {n} nodes '
            'referenced upon shutdown.').format(n=n)
        Cudd_Quit(self.manager)

    def __str__(self):
        peak = Cudd_ReadPeakNodeCount(self.manager)
        n_vars = Cudd_ReadSize(self.manager)
        n = Cudd_ReadNodeCount(self.manager)
        s = (
            'Binary decision diagram (CUDD wrapper) with:\n'
            '\t {n} nodes now\n'
            '\t {peak} nodes at peak\n'
            '\t {n_vars} BDD variables\n').format(
                n=n, peak=peak, n_vars=n_vars)
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

    def add_var(self, var):
        """Add a new variable named `var`."""
        # var already exists ?
        j = self.var_to_index.get(var)
        if j is not None:
            return j
        # new var
        j = len(self.var_to_index)
        self.var_to_index[var] = j
        Cudd_bddIthVar(self.manager, j)
        return j

    cpdef Function var(self, var):
        """Return node for variable `var`."""
        assert var in self.var_to_index, (
            'undefined variable "{v}", '
            'known variables are:\n {d}').format(
                v=var, d=self.var_to_index)
        j = self.var_to_index[var]
        r = Cudd_bddIthVar(self.manager, j)
        f = Function()
        f.init(self.manager, r)
        return f

    def add_ast(self, t):
        if t.type == 'var':
            f = self.var(t.value)
        else:
            raise NotImplemented('yet')
        return f

    def apply(self, op, Function u, Function v=None):
        assert self.manager == u.manager
        return _bdd_apply(op, u, v)

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
        """Return node for cube over `dvars`."""
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


cdef class Function(object):
    """Wrapper of `DdNode` from CUDD.

    Use as:

    ```
    bdd = BDD()
    u = bdd.True
    f = Function()
    f.init(bdd.manager, u.node)
    f = f & ~ f
    """

    cdef object __weakref__
    cpdef DdManager * manager
    cpdef DdNode * node

    cdef init(self, DdManager * mgr, DdNode * u):
        self.manager = mgr
        self.node = u
        Cudd_Ref(u)

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


cdef print_info(DdManager *mgr, f=None):
    cdef FILE *cf
    if f is None:
        f = sys.stdout
    cf = fdopen(f.fileno(), 'w')
    Cudd_PrintInfo(mgr, cf)


cpdef Function _bdd_apply(op, Function u, Function v=None):
    """Return result of applying `op`, as `Function`."""
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


def support(Function f, bdd, as_str=True):
    """Return support of node `f`."""
    mgr = f.manager
    u = f.node
    supp = set()
    _support(mgr, u, supp)
    id_to_var = {v: k for k, v in bdd.var_to_index.iteritems()}
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


cpdef Function _bdd_rename(Function u, bdd, dvars):
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


class BDDNodes(_Nodes):
    """AST to flatten prefix syntax to CUDD."""

    class Operator(_Nodes.Operator):
        def flatten(self, bdd, *arg, **kw):
            operands = [
                u.flatten(bdd=bdd, *arg, **kw)
                for u in self.operands]
            u = bdd.apply(self.operator, *operands)
            return u

    class Var(_Nodes.Var):
        def flatten(self, bdd, *arg, **kw):
            u = bdd.add_ast(self)
            return u

    class Num(_Nodes.Num):
        def flatten(self, BDD bdd, *arg, **kw):
            assert self.value in ('0', '1'), self.value
            u = int(self.value)
            if u == 0:
                r = bdd.False
            else:
                r = bdd.True
            return r


parser = Parser(nodes=BDDNodes())


def add_expr(e, bdd):
    """Return `Function` from `str` expression `e`."""
    tree = parser.parse(e)
    u = tree.flatten(bdd=bdd)
    return u


def slugsin_parser(s):
    sections = {
        'INPUT', 'OUTPUT', 'ENV_INIT', 'SYS_INIT',
        'ENV_TRANS', 'SYS_TRANS',
        'ENV_LIVENESS', 'SYS_LIVENESS'}
    sections = {'[{s}]'.format(s=s) for s in sections}
    d = dict()
    store = None
    for line in s.splitlines():
        if not line or line.startswith('#'):
            continue
        if line in sections:
            store = list()
            d[line] = store
        else:
            assert store is not None
            logger.debug('storing in: ', str(store))
            store.append(line)
    return d


def load_slugsin_file(fname):
    """Return a `dict` that keyed by slugsin file section."""
    with open(fname, 'r') as f:
        s = f.read()
    d = slugsin_parser(s)
    # pprint.pprint(d)
    return d


def make_automaton(d, bdd):
    """Return `symbolic.Automaton` from slugsin spec.

    @type d: dict(str: list)
    """
    a = Automaton()
    a.bdd = bdd
    dvars, prime, partition = add_variables(d, bdd)
    a.vars = dvars
    a.prime = prime
    a.evars = partition['evars']
    a.epvars = partition['epvars']
    a.uvars = partition['uvars']
    a.upvars = partition['upvars']
    # formulae
    # TODO: conjoin in prefix syntax
    sections = (
        '[ENV_INIT]', '[SYS_INIT]',
        '[ENV_TRANS]', '[SYS_TRANS]',
        '[ENV_LIVENESS]', '[SYS_LIVENESS]')
    dnodes = {k: list() for k in sections}
    for section, nodes in dnodes.iteritems():
        if section not in d:
            continue
        for s in d[section]:
            u = add_expr(s, bdd)
            nodes.append(u)
    # no liveness ?
    c = dnodes['[ENV_LIVENESS]']
    if not c:
        c.append(bdd.True)
    c = dnodes['[SYS_LIVENESS]']
    if not c:
        c.append(bdd.True)
    # assign them
    a.init['env'] = dnodes['[ENV_INIT]']
    a.init['sys'] = dnodes['[SYS_INIT]']
    a.action['env'] = dnodes['[ENV_TRANS]']
    a.action['sys'] = dnodes['[SYS_TRANS]']
    a.win['env'] = dnodes['[ENV_LIVENESS]']
    a.win['sys'] = dnodes['[SYS_LIVENESS]']
    return a


def add_variables(d, bdd):
    """Add unprimed and primed copies for bits from slugsin file."""
    suffix = "'"
    dvars = dict()
    prime = dict()
    for k, v in d.iteritems():
        if k not in ('[INPUT]', '[OUTPUT]'):
            continue
        for var in v:
            # make primed var
            pvar = var + suffix
            prime[var] = pvar
            # add unprimed and primed copies
            j = bdd.add_var(var)
            dvars[var] = j
            j = bdd.add_var(pvar)
            dvars[pvar] = j
            # print('added new variable "{var}", index: {j}'.format(
            #    var=var, j=j))
    uvars = list(d['[INPUT]'])
    upvars = map(prime.__getitem__, uvars)
    evars = list(d['[OUTPUT]'])
    epvars = map(prime.__getitem__, evars)
    partition = dict(uvars=uvars, upvars=upvars,
                     evars=evars, epvars=epvars)
    return dvars, prime, partition


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
    bdd.var_to_index = dict(x=0, y=1)
    print(bdd.var_to_index)

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


cdef dump(Function u, BDD bdd, fname):
    """Dump BDD as DDDMP file `fname`."""
    n = len(bdd.var_to_index)
    cdef FILE * f
    cdef char **names
    cdef int *indices
    names = <char **> PyMem_Malloc(n * sizeof(char *))
    indices = <int *> PyMem_Malloc(n * sizeof(int))
    for i, (var, idx) in enumerate(bdd.var_to_index.iteritems()):
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


def compute_winning_set(aut):
    """Compute winning region, w/o memoizing iterates."""
    print('Compute winning region')
    bdd = aut.bdd
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    start_time = time.time()
    # todo: add counter variable
    z = bdd.True
    zold = None
    logger.debug('before z fixpoint')
    while z != zold:
        logger.debug('Start Z iteration')
        zold = z
        zp = _bdd_rename(z, bdd, aut.prime)
        yj = list()
        for goal in aut.win['sys']:
            logger.debug('Guarantee: {goal}'.format(goal=goal))
            live_trans = goal & zp
            y = bdd.False
            yold = None
            while y != yold:
                logger.debug('Start Y iteration')
                yold = y
                yp = _bdd_rename(y, bdd, aut.prime)
                live_trans = live_trans | yp
                good = y
                for excuse in aut.win['env']:
                    # logger.debug(
                    #     'Assumption: {excuse}'.format(excuse=excuse))
                    x = bdd.True
                    xold = None
                    while x != xold:
                        logger.debug('Start X iteration')
                        xold = x
                        logger.debug('rename')
                        xp = _bdd_rename(x, bdd, aut.prime)
                        # desired transitions
                        logger.debug('conjoin')
                        x = xp & ~ excuse
                        del xp
                        logger.debug('disjoin')
                        x = x | live_trans
                        logger.debug('conjoin with sys_action')
                        x = x & sys_action
                        logger.debug(
                             "rho_s & (live_trans | (! J_i^e & X'))")
                        # cox
                        logger.debug('exists')
                        x = bdd.quantify(x, aut.epvars, forall=False)
                        logger.debug('implication')
                        x = x | ~ env_action
                        logger.debug('forall')
                        x = bdd.quantify(x, aut.upvars, forall=True)
                    logger.debug('Disjoin X of this assumption')
                    good = good | x
                    del x, xold
                y = good
                del good
            logger.debug('Reached Y fixpoint')
            yj.append(y)
            del y, yold, live_trans
        del zp
        # conjoin
        # z = compute_as_binary_tree(lambda x, y: x & y, yj)
        # z_ = linear_operator_simple(lambda x, y: x & y, yj)
        # assert z == z_
        linear_operator(lambda x, y: x & y, yj)
        z = yj[0]
        logger.debug('zold = {zold}'.format(zold=zold))
        logger.debug('z = {z}'.format(z=z))
        bdd.assert_consistent()
        current_time = time.time()
        t = current_time - start_time
        logger.info('Completed Z iteration in: {t} sec'.format(t=t))
    end_time = time.time()
    t = end_time - start_time
    print(
        'Reached Z fixpoint:\n'
        '{u}\n'
        'in: {t:1.0f} sec'.format(
            u=z, t=t))
    return z


def memoize_iterates(z, aut):
    """Store iterates of X, Y, given Z fixpoint."""
    pass


def construct_streett_1_transducer(z, aut):
    """Return Street(1) I/O transducer."""
    # Compute iterates, now that we know the outer fixpoint
    bdd = aut.bdd
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    transducers = list()
    selector = aut.add_expr('strat_type')
    zp = _bdd_rename(z, bdd, aut.prime)
    for j, goal in enumerate(aut.win['sys']):
        live_trans = goal & zp
        covered = bdd.False
        transducer = bdd.False
        y = bdd.False
        yold = None
        while y != yold:
            yold = y
            yp = _bdd_rename(y, bdd, aut.prime)
            live_trans = live_trans | yp
            good = y
            for excuse in aut.win['env'][0]:
                x = bdd.True
                xold = None
                while x != xold:
                    xold = x
                    xp = _bdd_rename(x, bdd, aut.prime)
                    x = xp & ~ excuse
                    x = x | live_trans
                    x = x & sys_action
                    found_paths = x
                    x = bdd.quantify(x, aut.epvars, forall=False)
                    x = x | ~ env_action
                    x = bdd.quantify(x, aut.upvars, forall=True)
                good = good | x
                new = bdd.quantify(found_paths, aut.epvars, forall=False)
                new = new & ~ covered
                covered = covered | new
                transducer = transducer | (new & found_paths)
            y = good
        # is it more efficient to do this now, or later ?
        # problem is that it couples with more variables (the counters)
        counter = aut.add_expr('c = {j}'.format(j=j))
        transducer = transducer & counter & (goal | ~ selector)
        transducers.append(transducer)
    # disjoin the strategies for the individual goals
    transducer = compute_as_binary_tree(lambda x, y: x or y, transducers)
    return transducer


def recurse_binary(f, x):
    """Recursively traverse binary tree of computation."""
    n = len(x)
    assert n > 0
    if n == 1:
        return x.pop()
    k = math.floor(n % 2)
    m = 2**k
    left = x[:m]
    right = x[m:]
    del x[:]
    a = recurse_binary(f, left)
    b = recurse_binary(f, right)
    return f(a, b)


def compute_as_binary_tree(f, x):
    """Return result of applying operator `f`."""
    logger.debug('++ start binary tree')
    assert len(x) > 0
    # y = list(x)
    # del x[:]  # deref in caller
    # assert len(x) == 0
    # x = y
    while len(x) > 1:
        n = len(x)
        k = math.floor(n / 2.0)
        logger.debug('Binary at: {n}'.format(n=n))
        # consume the power of 2
        for i in xrange(k):
            j = 2 * i
            a = x[j]
            b = x[j + 1]
            x[i] = f(a, b)
        # has last element ?
        if len(x) % 2 == 1:
            x[k] = x[2 * k]
        # empty tail
        del x[k + 1:]
        assert len(x) == n - k, (len(x), n - k)
    assert len(x) == 1, len(x)
    logger.debug('-- done binary tree')
    return x[0]


def compute_as_binary_tree_simple(f, x):
    """Return result of applying operator `f`."""
    logger.debug('++ start binary tree')
    assert len(x) > 0
    # y = list(x)
    # del x[:]  # deref in caller
    # assert len(x) == 0
    # x = y
    while len(x) > 1:
        n = len(x)
        k = math.floor(n / 2.0)
        # consume the power of 2
        r = [f(a, b) for a, b in zip(x[::2], x[1::2])]
        # has last element ?
        if len(x) % 2 == 1:
            r.append(x[-1])
        # empty tail
        x = r
        assert len(x) == n - k, (len(x), n - k)
    assert len(x) == 1, len(x)
    logger.debug('-- done binary tree')
    return x[0]


def linear_operator(f, x):
    """Return result of applying linearly operator `f`."""
    logger.debug('++ start linear operator')
    assert len(x) > 0
    n = len(x)
    for i in xrange(1, n):
        x[0] = f(x[0], x.pop())
    assert len(x) == 1
    logger.debug('-- done linear operator')


def linear_operator_simple(f, x):
    """Return result of applying linearly operator `f`."""
    logger.debug('++ start simple linear operator')
    assert len(x) > 0
    u = x[0]
    for v in x[1:]:
        u = f(u, v)
    logger.debug('-- done simple linear operator')
    return u


def solve_game(fname):
    """Construct transducer for game in file `fname`."""
    d = load_slugsin_file(fname)
    bdd = BDD()
    aut = make_automaton(d, bdd)
    print(aut)
    print(bdd)
    # aut.action['sys'][0] = bdd.False
    z = compute_winning_set(aut)
    print(bdd)


fname = 'slugs.txt'
logger.addHandler(logging.StreamHandler())
logger.setLevel('DEBUG')

