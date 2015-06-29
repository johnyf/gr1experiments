from libc.stdio cimport FILE, fdopen
import sys
from omega.symbolic.bdd import Nodes as _Nodes
from omega.symbolic.bdd import Parser


cdef extern from "cudd.h":
    ctypedef unsigned int DdHalfWord
    cdef struct DdNode:
        DdHalfWord index
        DdHalfWord ref
    ctypedef DdNode DdNode
    cdef struct DdManager:
        pass
    ctypedef DdManager DdManager
    cdef DdManager * Cudd_Init (
    	unsigned int numVars,
    	unsigned int numVarsZ,
    	unsigned int numSlots,
    	unsigned int cacheSize,
    	unsigned long maxMemory)
    cdef DdNode * Cudd_bddNewVar(DdManager *dd)
    cdef DdNode * Cudd_bddIthVar(DdManager *dd, int i)
    cdef DdNode * Cudd_ReadLogicZero(DdManager *dd)
    cdef DdNode * Cudd_ReadOne(DdManager *dd)
    cdef DdNode * Cudd_Not(DdNode *dd)
    cdef DdNode * Cudd_bddIte(DdManager *dd,
                              DdNode *f, DdNode *g, DdNode *h)
    cdef DdNode * Cudd_bddAnd(DdManager *dd, DdNode *dd, DdNode *dd)
    cdef DdNode * Cudd_bddOr(DdManager *dd, DdNode *dd, DdNode *dd)
    cdef DdNode * Cudd_bddXor(DdManager *dd, DdNode *f, DdNode *g)

    cdef void Cudd_Ref(DdNode *n)
    cdef void Cudd_RecursiveDeref(DdManager *table, DdNode *n)
    cdef void Cudd_Deref(DdNode *n)
    cdef int Cudd_CheckZeroRef(DdManager *manager)
    cdef void Cudd_Quit(DdManager *unique)
    cdef int Cudd_PrintInfo(DdManager *dd, FILE *fp)
    cdef int Cudd_DagSize(DdNode *node)

    cdef DdNode * Cudd_bddExistAbstract(
        	DdManager *manager, DdNode *f, DdNode *cube)
    cdef DdNode * Cudd_bddUnivAbstract(
        	DdManager *manager, DdNode *f, DdNode *cube)
    cdef DdNode * Cudd_bddSwapVariables(
        	DdManager *dd, DdNode *f, DdNode **x, DdNode **y, int n)


CUDD_UNIQUE_SLOTS = 256
CUDD_CACHE_SLOTS = 262144


# TODO: provide two implementations, one using __weakref__


cdef class BDD(object):
    """Wrapper of CUDD manager.

    Same interface as `dd.bdd.BDD`.
    """

    cdef DdManager * manager
    cdef public var_to_index

    def __cinit__(self):
        self.var_to_index = dict()

    cdef incref(self, DdNode * u):
        Cudd_Ref(u)

    cdef decref(self, DdNode * u, recursive=True):
        if recursive:
            Cudd_RecursiveDeref(self.manager, u)
        else:
            Cudd_Deref(u)

    def False(self):
        return self._bool(False)

    def True(self):
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

    def add_ast(self, t):
        if t.type == 'var':
            assert t.value in self.var_to_index, (
                'undefined variable "{v}", '
                'known variables are:\n {d}').format(
                    v=t.value, d=self.var_to_index)
            j = self.var_to_index[t.value]
            #cdef DdNode * r
            r = Cudd_bddIthVar(self.manager, j)
            f = Function()
            f.init(self.manager, r)
            # what about adding a BDD method for proj functions ?
        else:
            raise NotImplemented('yet')
        return f

    def apply(self, op, Function u, Function v=None):
        assert self.manager == u.manager
        return _bdd_apply(op, u, v)


cdef class Function:
    """Wrapper of `DdNode` from CUDD."""

    cpdef DdManager * manager
    cpdef DdNode * node

    cdef init(self, DdManager * mgr, DdNode * u):
        self.manager = mgr
        self.node = u
        Cudd_Ref(u)

    def __dealloc__(self):
        Cudd_RecursiveDeref(self.manager, self.node)

    def __str__(self):
        return 'Function(DdNode at index {u})'.format(
            u=self.node.index)

    def __not__(self):
        cdef DdNode * r
        r = Cudd_Not(self.node)

    def __and__(Function self, Function other):
        r = Cudd_bddAnd(self.manager, self.node, other.node)
        f = Function()
        f.init(self.manager, r)
        return f

    def __or__(Function self, Function other):
        r = Cudd_bddOr(self.manager, self.node, other.node)
        f = Function()
        f.init(self.manager, r)
        return f


cpdef Function _bdd_apply(op, Function u, Function v=None):
    """Return result of applying `op`, as `Function`."""
    cdef DdNode * r
    cdef DdManager * mgr
    mgr = u.manager
    # unary
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
    else:
        raise Exception(
            'unknown operator: {op}'.format(op=op))
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
                r = bdd.False()
            else:
                r = bdd.True()
            return r


parser = Parser(nodes=BDDNodes())


def add_expr(e, bdd):
    tree = parser.parse(e)
    u = tree.flatten(bdd=bdd)
    return u


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
    Cudd_bddSwapVariables(mgr, u, x, y, 2)
    # print_info(mgr)

    fu = Function()
    fu.init(mgr, u)
    mgr = fu.manager
    fv = Function()
    fv.init(mgr, v)
    f = _bdd_apply('and', fu, fv)
    print(f)

    bdd = BDD()
    bdd.manager = mgr
    bdd.var_to_index = dict(x=0, y=1)
    print(bdd.var_to_index)

    s = '| 0 1'
    f = add_expr(s, bdd)
    print(f)
    u = Cudd_ReadOne(mgr)
    print(u.index)


cdef print_info(DdManager * mgr):
    cdef FILE * cfile
    cfile = fdopen(sys.stdout.fileno(), 'w')
    Cudd_PrintInfo(mgr, cfile)


print("hello world")
main()
