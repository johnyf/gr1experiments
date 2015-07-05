#!/usr/bin/env python
import argparse
import logging
import math
import time
import cudd
from omega.symbolic.bdd import Nodes as _Nodes
from omega.symbolic.bdd import Parser
from omega.symbolic.symbolic import Automaton


logger = logging.getLogger(__name__)


# TODO:
#
# record variable order
# record events (reordering, garbage collection)
# plot events in annotated timeline
# compare different histories of variable orders
# check that `config.json` is the same
# group primed and unprimed vars
# use efficient rename for neighbors
# use a CUDD map for repeated renaming


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
            u = bdd.add_var(self.value)
            return u

    class Num(_Nodes.Num):
        def flatten(self, bdd, *arg, **kw):
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
            logger.debug('storing line: {line}'.format(line=line))
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


# @profile
def compute_winning_set(aut, z=None):
    """Compute winning region, w/o memoizing iterates."""
    logger.info('++ Compute winning region')
    log = logging.getLogger('solver')
    bdd = aut.bdd
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    start_time = time.time()
    # todo: add counter variable
    if z is None:
        z = bdd.True
    zold = None
    log.debug('before z fixpoint')
    while z != zold:
        log.debug('Start Z iteration')
        zold = z
        # yj = list()
        for j, goal in enumerate(aut.win['sys']):
            log.info('Goal: {j}'.format(j=j))
            log.info(bdd)
            zp = cudd.rename(z, bdd, aut.prime)
            live_trans = goal & zp
            y = bdd.False
            yold = None
            while y != yold:
                log.debug('Start Y iteration')
                yold = y
                yp = cudd.rename(y, bdd, aut.prime)
                live_trans = live_trans | yp
                good = y
                for excuse in aut.win['env']:
                    # log.debug(
                    #     'Assumption: {excuse}'.format(excuse=excuse))
                    x = bdd.True
                    xold = None
                    while x != xold:
                        log.debug('Start X iteration')
                        xold = x
                        xp = cudd.rename(x, bdd, aut.prime)
                        # desired transitions
                        x = xp & ~ excuse
                        x = x | live_trans
                        old = False
                        if old:
                            x = x & sys_action
                            x = bdd.quantify(x, aut.epvars, forall=False)
                            x = x | ~ env_action
                            x = bdd.quantify(x, aut.upvars, forall=True)
                        else:
                            x = cudd.and_abstract(x, sys_action,
                                                  aut.epvars, bdd)
                            x = cudd.or_abstract(x, ~ env_action,
                                                 aut.upvars, bdd)
                    log.debug('Disjoin X of this assumption')
                    good = good | x
                    del x, xold
                y = good
                del good
            log.debug('Reached Y fixpoint')
            z = z & y
            # yj.append(y)
            del y, yold, live_trans
        del zp
        # conjoin
        # z = recurse_binary(lambda x, y: x & y, yj)
        # z_ = linear_operator_simple(lambda x, y: x & y, yj)
        # assert z == z_
        # z = linear_operator(lambda x, y: x & y, yj)
        bdd.assert_consistent()
        current_time = time.time()
        t = current_time - start_time
        log.info('Completed Z iteration at: {t} sec'.format(t=t))
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
    # copy vars
    bdd = aut.bdd
    other_bdd = cudd.BDD()
    for var in bdd.var_to_index:
        other_bdd.add_var(var)
    # Compute iterates, now that we know the outer fixpoint
    log = logging.getLogger('solver')
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    # bdd.add_var('strat_type')
    # selector = aut.add_expr('strat_type')
    store = list()
    all_new = list()
    zp = cudd.rename(z, bdd, aut.prime)
    for j, goal in enumerate(aut.win['sys']):
        log.info('Goal: {j}'.format(j=j))
        log.info(bdd)
        covered = bdd.False
        transducer = bdd.False
        live_trans = goal & zp
        y = bdd.False
        yold = None
        while y != yold:
            log.debug('Start Y iteration')
            yold = y
            yp = cudd.rename(y, bdd, aut.prime)
            live_trans = live_trans | yp
            good = y
            for excuse in aut.win['env']:
                x = bdd.True
                xold = None
                paths = None
                new = None
                while x != xold:
                    del paths, new
                    log.debug('Start X iteration')
                    xold = x
                    xp = cudd.rename(x, bdd, aut.prime)
                    x = xp & ~ excuse
                    paths = x | live_trans
                    new = cudd.and_abstract(paths, sys_action, aut.epvars, bdd)
                    x = cudd.or_abstract(new, ~ env_action, aut.upvars, bdd)
                good = good | x
                print('transfer')
                paths = cudd.transfer_bdd(paths, other_bdd)
                new = cudd.transfer_bdd(new, other_bdd)
                store.append((j, paths))
                all_new.append((j, new))
                print('the other bdd:')
                print(other_bdd)
            y = good
        # is it more efficient to do this now, or later ?
        # problem is that it couples with more variables (the counters)
    print('done, lets construct strategies now')
    transducers = list()
    for j, goal in enumerate(aut.win['sys']):
        log.info('Goal: {j}'.format(j=j))
        log.info(other_bdd)
        covered = other_bdd.False
        transducer = other_bdd.False
        for (k, paths), (k, new) in zip(store, all_new):
            if k != j:
                continue
            print('covering...')
            covered = covered | new
            new = new & ~ covered
            transducer = transducer | (new & paths)
        del paths, new
        log.info('appending transducer for this goal')
        # counter = aut.add_expr('c = {j}'.format(j=j))
        # transducer = transducer & counter & (goal | ~ selector)
        transducers.append(transducer)
        del covered
    log.info(other_bdd)
    log.info('clean intermediate results')
    del store[:]
    del all_new[:]
    log.info(other_bdd)
    # disjoin the strategies for the individual goals
    # transducer = linear_operator(lambda x, y: x | y, transducers)
    log.info('disjoin transducers')
    transducer = recurse_binary(lambda x, y: x | y, transducers)
    log.info(other_bdd)
    log.info(other_bdd)
    sys_action = cudd.transfer_bdd(sys_action, other_bdd)
    log.info('conjoin with sys action')
    transducer = transducer & sys_action
    log.info(other_bdd)
    log.info(transducer)
    del sys_action
    transducer = cudd.transfer_bdd(transducer, bdd)
    return transducer


def recurse_binary(f, x):
    """Recursively traverse binary tree of computation."""
    n = len(x)
    assert n > 0
    if n == 1:
        return x.pop()
    k = int(math.floor(n % 2))
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
        logger.debug('Binary at: {n}'.format(n=n))
        k = int(math.floor(n / 2.0))
        # consume the power of 2
        for i in xrange(k):
            j = 2 * i
            a = x[j]
            b = x[j + 1]
            x[i] = f(a, b)
        if len(x) % 2 == 1:
            # has last element ?
            x[k] = x[2 * k]
            # empty tail
            del x[k + 1:]
        else:
            del x[k:]
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
        k = int(math.floor(n / 2.0))
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
    return x.pop()


def linear_operator(f, x):
    """Return result of applying linearly operator `f`."""
    logger.debug('++ start linear operator')
    assert len(x) > 0
    n = len(x)
    for i in xrange(1, n):
        x[0] = f(x[0], x.pop())
    assert len(x) == 1, len(x)
    logger.debug('-- done linear operator')
    return x.pop()


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
    bdd = cudd.BDD()
    aut = make_automaton(d, bdd)
    # print(aut)
    print(bdd)
    # aut.action['sys'][0] = bdd.False
    z = compute_winning_set(aut)
    print(bdd)
    z = compute_winning_set(aut, z)
    # construct_streett_1_transducer(z, aut)
    del aut, z


def test_binary_operators():
    for n in xrange(1, 1500):
        a = range(n)
        f = plus
        x0 = compute_as_binary_tree(f, list(a))
        x1 = compute_as_binary_tree_simple(f, list(a))
        x2 = linear_operator(f, list(a))
        x3 = linear_operator_simple(f, list(a))
        x4 = recurse_binary(f, list(a))
        z = sum(a)
        assert x0 == z, (x0, z)
        assert x1 == z, (x1, z)
        assert x2 == z, (x2, z)
        assert x3 == z, (x3, z)
        assert x4 == z, (x4, z)
        print(z)


def plus(x, y):
    return x + y


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str, help='slugsin input file')
    args = p.parse_args()
    fname = args.file
    # fname = 'slugs_small.txt'
    logger = logging.getLogger('solver')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')
    solve_game(fname)
