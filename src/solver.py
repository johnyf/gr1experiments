#!/usr/bin/env python
import argparse
import logging
import math
import time
# import cudd
import cudd
from cudd import BDD
import natsort
from omega.symbolic.bdd import add_expr
from omega.symbolic.symbolic import Automaton


logger = logging.getLogger(__name__)
REORDERING_LOG = 'reorder'


# TODO:
#
# record events (reordering, garbage collection)
# plot events in annotated timeline
#
# check that `config.json` is the same
#
# group primed and unprimed vars
# use efficient rename for neighbors
# use a CUDD map for repeated renaming


def load_slugsin_file(fname):
    """Return `dict` keyed by slugsin file section."""
    with open(fname, 'r') as f:
        s = f.read()
    d = _parser_slugsin(s)
    return d


def _parser_slugsin(s):
    """Return sections of slugsin file, as `dict`."""
    sections = dict(
        INPUT='input',
        OUTPUT='output',
        ENV_INIT='env_init',
        SYS_INIT='sys_init',
        ENV_TRANS='env_action',
        SYS_TRANS='sys_action',
        ENV_LIVENESS='env_win',
        SYS_LIVENESS='sys_win')
    sections = {
        '[{k}]'.format(k=k): v
        for k, v in sections.iteritems()}
    d = dict()
    store = None
    for line in s.splitlines():
        if not line or line.startswith('#'):
            continue
        if line in sections:
            store = list()
            key = sections[line]
            d[key] = store
        else:
            assert store is not None
            store.append(line)
    return d


def make_automaton(d, bdd):
    """Return `symbolic.Automaton` from slugsin spec.

    @type d: dict(str: list)
    """
    a = Automaton()
    a.bdd = bdd
    dvars, prime, partition = _add_variables(d, bdd)
    reordering_log = logging.getLogger(REORDERING_LOG)
    s = var_order(bdd)
    reordering_log.debug(repr(s))
    # TODO: correct this
    a.vars = dvars
    a.prime = prime
    a.evars = partition['evars']
    a.epvars = partition['epvars']
    a.uvars = partition['uvars']
    a.upvars = partition['upvars']
    # formulae
    # TODO: conjoin in prefix syntax
    sections = (
        'env_init', 'env_action', 'env_win',
        'sys_init', 'sys_action', 'sys_win')
    dnodes = {k: list() for k in sections}
    # add in fixed order, to improve
    # effectiveness of reordering
    lengths = {k: section_len(d[k]) for k in sections}
    for section in sorted(lengths, key=lengths.__getitem__,
                          reverse=True):
        if section not in d:
            continue
        for s in d[section]:
            u = add_expr(s, bdd)
            dnodes[section].append(u)
    # no liveness ?
    c = dnodes['env_win']
    if not c:
        c.append(bdd.True)
    c = dnodes['sys_win']
    if not c:
        c.append(bdd.True)
    # assign them
    a.init['env'] = dnodes['env_init']
    a.init['sys'] = dnodes['sys_init']
    a.action['env'] = dnodes['env_action']
    a.action['sys'] = dnodes['sys_action']
    a.win['env'] = dnodes['env_win']
    a.win['sys'] = dnodes['sys_win']
    return a


def section_len(formulae):
    """Return sum of `len` of `str` in `formulae`."""
    return sum(len(s) for s in formulae)


def _add_variables(d, bdd):
    """Add unprimed and primed copies for bits from slugsin file."""
    suffix = "'"
    dvars = dict()
    prime = dict()
    for k, v in d.iteritems():
        if k not in ('input', 'output'):
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
    uvars = list(d['input'])
    upvars = map(prime.__getitem__, uvars)
    evars = list(d['output'])
    epvars = map(prime.__getitem__, evars)
    partition = dict(uvars=uvars, upvars=upvars,
                     evars=evars, epvars=epvars)
    return dvars, prime, partition


# @profile
def compute_winning_set(aut, z=None):
    """Compute winning region, w/o memoizing iterates."""
    logger.info('++ Compute winning region')
    log = logging.getLogger('solver')
    reordering_log = logging.getLogger(REORDERING_LOG)
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
        s = var_order(bdd)
        reordering_log.debug(repr(s))
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
                        s = var_order(bdd)
                        reordering_log.debug(repr(s))
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


def var_order(bdd):
    """Return `dict` that maps each variable to a level.

    @rtype: `dict(str: int)`
    """
    return {var: bdd.level_of_var(var) for var in bdd.vars}


def memoize_iterates(z, aut):
    """Store iterates of X, Y, given Z fixpoint."""
    pass


def construct_streett_1_transducer(z, aut):
    """Return Street(1) I/O transducer."""
    # copy vars
    bdd = aut.bdd
    other_bdd = BDD()
    for var in bdd._index_of_var:
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
                    new = cudd.and_abstract(paths, sys_action,
                                            aut.epvars, bdd)
                    x = cudd.or_abstract(new, ~ env_action,
                                         aut.upvars, bdd)
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
    transducer = _recurse_binary(lambda x, y: x | y, transducers)
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


def _recurse_binary(f, x):
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
    a = _recurse_binary(f, left)
    b = _recurse_binary(f, right)
    return f(a, b)


def _compute_as_binary_tree(f, x):
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


def _compute_as_binary_tree_simple(f, x):
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


def _linear_operator(f, x):
    """Return result of applying linearly operator `f`."""
    logger.debug('++ start linear operator')
    assert len(x) > 0
    n = len(x)
    for i in xrange(1, n):
        x[0] = f(x[0], x.pop())
    assert len(x) == 1, len(x)
    logger.debug('-- done linear operator')
    return x.pop()


def _linear_operator_simple(f, x):
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
    # aut.action['sys'][0] = bdd.False
    z = compute_winning_set(aut)
    construct_streett_1_transducer(z, aut)
    del aut, z


def test_binary_operators():
    for n in xrange(1, 1500):
        a = range(n)
        f = _plus
        x0 = _compute_as_binary_tree(f, list(a))
        x1 = _compute_as_binary_tree_simple(f, list(a))
        x2 = _linear_operator(f, list(a))
        x3 = _linear_operator_simple(f, list(a))
        x4 = _recurse_binary(f, list(a))
        z = sum(a)
        assert x0 == z, (x0, z)
        assert x1 == z, (x1, z)
        assert x2 == z, (x2, z)
        assert x3 == z, (x3, z)
        assert x4 == z, (x4, z)
        print(z)


def _plus(x, y):
    return x + y


def load_order_history(fname):
    with open(fname, 'r') as f:
            s = f.read()
    t = dict()
    for line in s.splitlines():
        d = eval(line)
        for k, v in d.iteritems():
            if k not in t:
                t[k] = list()
            t[k].append(v)
    return t


def main():
    fname = 'reordering_slugs_31.txt'
    other_fname = 'reordering_slugs_31_old.txt'
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str, help='slugsin input file')
    p.add_argument('--plot-order', action='store_true',
                   help='plot reordering of variales from log')
    args = p.parse_args()
    if args.plot_order:
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t = load_order_history(fname)
        other_t = load_order_history(other_fname)
        plt.hold('on')
        for i, k in enumerate(natsort.natsorted(t)):
            v = t[k]
            w = other_t.get(k)
            if w is None:
                print('Missing var "{var}"'.format(var=k))
                continue
            m = min(len(v), len(w))
            # ax.plot(range(len(v)), v, i)
            ax.plot(v[:m], w[:m], i)
        plt.savefig('reordering.pdf')
        plt.show()
        return
    input_fname = args.file
    # fname = 'slugs_small.txt'
    logger = logging.getLogger('solver')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')
    # reordering
    reordering_fname = 'reordering_{f}'.format(f=input_fname)
    log = logging.getLogger(REORDERING_LOG)
    h = logging.FileHandler(reordering_fname, 'w')
    log.addHandler(h)
    log.setLevel('DEBUG')
    solve_game(input_fname)


if __name__ == '__main__':
    main()
