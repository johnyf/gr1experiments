import argparse
import copy
import logging
import math
import time
from dd import autoref as _bdd
import natsort
from omega.logic import syntax
from omega.symbolic import symbolic


logger = logging.getLogger(__name__)
REORDERING_LOG = 'reorder'
SOLVER_LOG = 'solver'
COUNTER = '_jx_b'
SELECTOR = 'strat_type'


# TODO:
#
# record events (reordering, garbage collection)
# plot events in annotated timeline
#
# group primed and unprimed vars
# use efficient rename for neighbors
# use a CUDD map for repeated renaming
#
# init of counter and strategy_type
# allow passing a desired level for the first bit
#     of an integer


def solve_game(s):
    """Construct transducer for game in file `fname`.

    @param s: `str` in `slugs` syntax
    """
    d = parse_slugsin(s)
    bdd = _bdd.BDD()
    aut = make_automaton(d, bdd)
    z = compute_winning_set(aut)
    assert z != bdd.False, 'unrealizable'
    t = construct_streett_transducer(z, aut)
    fname = 'tugs_strategy.txt'
    t.bdd.dump(t.action['sys'][0], fname)
    logger.info(t)
    log = logging.getLogger(SOLVER_LOG)
    log.info(aut)
    del z


def log_reordering(fname):
    reordering_fname = 'reordering_{f}'.format(f=fname)
    log = logging.getLogger(REORDERING_LOG)
    h = logging.FileHandler(reordering_fname, 'w')
    log.addHandler(h)
    log.setLevel('ERROR')


def parse_slugsin(s):
    """Return `dict` keyed by `slugsin` file section."""
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
    # bits -- shouldn't produce safety or init formulae
    a = symbolic.Automaton()
    a.vars = _init_vars(d)
    a = symbolic._bitblast(a)
    # formulae
    sections = symbolic._make_section_map(a)
    for section, target in sections.iteritems():
        target.extend(d[section])
    a.conjoin('prefix')
    logger.debug(a)
    # compile
    a.bdd = bdd  # use `cudd.BDD`, but fill vars
    a = symbolic._bitvector_to_bdd(a)
    symbolic.fill_blanks(a, as_bdd=True)
    return a


def _init_vars(d):
    dvars = dict()
    players = dict(input='env', output='sys')
    for section in ('input', 'output'):
        if section not in d:
            continue
        owner = players[section]
        for bit in d[section]:
            dvars[bit] = dict(type='bool', owner=owner)
    return dvars


# @profile
def compute_winning_set(aut, z=None):
    """Compute winning region, w/o memoizing iterates."""
    USE_BINARY = True
    logger.info('++ Compute winning region')
    log = logging.getLogger(SOLVER_LOG)
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
        yj = list()
        for j, goal in enumerate(aut.win['sys']):
            log.info('Goal: {j}'.format(j=j))
            log.info(bdd)
            zp = _bdd.rename(z, bdd, aut.prime)
            live_trans = goal & zp
            y = bdd.False
            yold = None
            while y != yold:
                log.debug('Start Y iteration')
                yold = y
                yp = _bdd.rename(y, bdd, aut.prime)
                live_trans = live_trans | yp
                good = y
                for i, excuse in enumerate(aut.win['env']):
                    x = bdd.True
                    xold = None
                    while x != xold:
                        log.debug('Start X iteration')
                        xold = x
                        xp = _bdd.rename(x, bdd, aut.prime)
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
                            x = and_exists(x, sys_action,
                                           aut.epvars, bdd)
                            x = or_forall(x, ~ env_action,
                                          aut.upvars, bdd)
                        # log
                        try:
                            stats = bdd.statistics()
                            n_nodes = stats['n_nodes']
                            reordering_time = stats['reordering_time']
                            current_time = time.time()
                            t = current_time - start_time
                            log.info((
                                'time (ms): {t}, '
                                'reordering (ms): {reorder_time}, '
                                'sysj: {sysj}, '
                                'envi: {envi}, '
                                'nodes: all: {n_nodes}, '
                                'Z: {z}, '
                                'Y: {y}, '
                                'X: {x}\n').format(
                                    t=t,
                                    reorder_time=reordering_time,
                                    sysj=j,
                                    envi=i,
                                    n_nodes=n_nodes,
                                    z=len(z),
                                    y=len(y),
                                    x=len(x)))
                        except:
                            pass
                    log.debug('Reached X fixpoint')
                    del xold
                    good = good | x
                    del x
                y = good
                del good
            del yold, live_trans
            log.debug('Reached Y fixpoint')
            if USE_BINARY:
                yj.append(y)
            else:
                z = z & y
            del y, goal
        del zp
        # conjoin
        # if USE_BINARY:
        z = syntax.recurse_binary(conj, yj)
        # z = syntax._linear_operator_simple(conj, yj)
        bdd.assert_consistent()
        current_time = time.time()
        t = current_time - start_time
        log.info('Completed Z iteration in: {t} sec'.format(t=t))
    end_time = time.time()
    t = end_time - start_time
    log.info(
        'Reached Z fixpoint:\n'
        '{u}\n'
        'in: {t:1.0f} sec'.format(
            u=z, t=t))
    return z


# @profile
def construct_streett_transducer(z, aut):
    """Return Street(1) I/O transducer."""
    log = logging.getLogger(SOLVER_LOG)
    reordering_log = logging.getLogger(REORDERING_LOG)
    # copy vars
    bdd = aut.bdd
    other_bdd = _bdd.BDD()
    _bdd.copy_vars(bdd, other_bdd)
    # Compute iterates, now that we know the outer fixpoint
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    sys_action_2 = _bdd.copy_bdd(sys_action, bdd, other_bdd)
    env_action_2 = _bdd.copy_bdd(env_action, bdd, other_bdd)
    log.info('done copying actions')
    zp = _bdd.rename(z, bdd, aut.prime)
    # transducer automaton
    t = symbolic.Automaton()
    t.vars = copy.deepcopy(aut.vars)
    t.vars[SELECTOR] = dict(type='bool', owner='sys', level=0)
    n_goals = len(aut.win['sys'])
    t.vars[COUNTER] = dict(
        type='saturating', dom=(0, n_goals - 1),
        owner='sys', level=0)
    t = t.build(other_bdd, add=True)
    transducers = list()
    selector = t.add_expr(SELECTOR)
    start_time = time.time()
    for j, goal in enumerate(aut.win['sys']):
        log.info('Goal: {j}'.format(j=j))
        log.info(bdd)
        # for fixpoint
        live_trans = goal & zp
        y = bdd.False
        yold = None
        # for strategy construction
        covered = other_bdd.False
        transducer = other_bdd.False
        while y != yold:
            log.debug('Start Y iteration')
            yold = y
            yp = _bdd.rename(y, bdd, aut.prime)
            live_trans = live_trans | yp
            good = y
            for i, excuse in enumerate(aut.win['env']):
                x = bdd.True
                xold = None
                paths = None
                new = None
                while x != xold:
                    del paths, new
                    log.debug('Start X iteration')
                    xold = x
                    xp = _bdd.rename(x, bdd, aut.prime)
                    x = xp & ~ excuse
                    del xp
                    paths = x | live_trans
                    new = and_exists(paths, sys_action,
                                     aut.epvars, bdd)
                    x = or_forall(new, ~ env_action,
                                  aut.upvars, bdd)
                    # log
                    try:
                        stats = bdd.statistics()
                        n_nodes = stats['n_nodes']
                        reordering_time = stats['reordering_time']
                        current_time = time.time()
                        dtime = current_time - start_time
                        log.info((
                            'time (ms): {t}, '
                            'reordering (ms): {reorder_time}, '
                            'goal: {j}, '
                            'onion_ring: 0, '
                            'nodes: all: {n_nodes}, '
                            'strategy: {strategy}, '
                            'cases_covered: 0, '
                            'new_cases: 0\n').format(
                                t=dtime,
                                reorder_time=int(reordering_time),
                                j=j,
                                strategy=len(transducer),
                                n_nodes=n_nodes,
                                z=len(z),
                                y=len(y),
                                x=len(x)))
                    except:
                        pass
                del xold, excuse
                good = good | x
                del x
                # strategy construction
                # in `other_bdd`
                log.info('transfer')
                paths = _bdd.copy_bdd(paths, bdd, other_bdd)
                new = _bdd.copy_bdd(new, bdd, other_bdd)
                rim = new & ~ covered
                covered = covered | new
                del new
                rim = rim & paths
                del paths
                transducer = transducer | rim
                del rim
            y = good
            del good
        assert y == z, (y, z)
        del y, yold, covered
        log.info('other BDD:')
        log.info(other_bdd)
        # make transducer
        goal = _bdd.copy_bdd(goal, bdd, other_bdd)
        counter = t.add_expr('{c} = {j}'.format(c=COUNTER, j=j))
        u = goal | ~ selector
        del goal
        u = counter & u
        del counter
        transducer = transducer & u
        del u
        transducer = transducer & sys_action_2
        # check_winning_region(transducer, aut, t, bdd, other_bdd, z, j)
        transducers.append(transducer)
        # log
        s = var_order(other_bdd)
        reordering_log.debug(repr(s))
        del transducer
    del sys_action_2, zp
    log.info(other_bdd)
    # disjoin the strategies for the individual goals
    # transducer = linear_operator(lambda x, y: x | y, transducers)
    log.info('disjoin transducers')
    transducer = syntax.recurse_binary(disj, transducers)
    # transducer = syntax._linear_operator_simple(disj, transducers)
    n_remain = len(transducers)
    assert n_remain == 0, n_remain
    log.info('bdd:\n{b}'.format(b=bdd))
    log.info('other bdd:\n{b}'.format(b=other_bdd))
    # add counter limits
    # transducer = transducer & t.action['sys'][0]
    # env lost ?
    # transducer = transducer | ~ env_action_2
    print('size of final transducer:', len(transducer))
    t.action['sys'] = [transducer]
    log.debug(
        'time (ms): 0, '
        'reordering (ms): 0, '
        'goal: 0, '
        'nodes: all: 100, '
        'combined_strategy: 0\n')
    # self-check
    check_winning_region(transducer, aut, t, bdd, other_bdd, z, 0)
    del selector, env_action_2, transducer
    return t


def check_winning_region(transducer, aut, t, bdd, other_bdd, z, j):
    u = transducer
    u = symbolic.cofactor(transducer, COUNTER, j, other_bdd, t.vars)
    u = other_bdd.quantify(u, [SELECTOR], forall=False)
    u = other_bdd.quantify(u, t.epvars, forall=False)
    u = other_bdd.quantify(u, t.upvars, forall=True)
    z_ = _bdd.copy_bdd(z, bdd, other_bdd)
    print('u == z', u == z_)


def recurse_binary(f, x, bdds):
    """Recursively traverse binary tree of computation."""
    logger = logging.getLogger(SOLVER_LOG)
    logger.info('++ recurse binary')
    n = len(x)
    logger.debug('{n} items left to recurse'.format(n=n))
    assert n > 0
    if n == 1:
        assert len(x) == 1, x
        assert len(bdds) == 1, bdds
        return x.pop(), bdds.pop()
    k = int(math.floor(math.log(n, 2)))
    m = 2**k
    if m == n:
        m = int(n / 2.0)
    left = x[:m]
    right = x[m:]
    del x[:]
    a, bdd_a = recurse_binary(f, left, bdds[:m])
    b, bdd_b = recurse_binary(f, right, bdds[m:])
    new_bdd = _bdd.BDD()
    _bdd.copy_vars(bdds[0], new_bdd)
    cpa = _bdd.copy_bdd(a, bdd_a, new_bdd)
    cpb = _bdd.copy_bdd(b, bdd_b, new_bdd)
    # logger.info(bdds)
    logger.info('-- done recurse binary ({n} items)'.format(n=n))
    return f(cpa, cpb), new_bdd


def make_strategy(store, all_new, j, goal, aut):
    log = logging.getLogger(SOLVER_LOG)
    log.info('++ Make strategy for goal: {j}'.format(j=j))
    bdd = aut.bdd
    log.info(bdd)
    covered = bdd.False
    transducer = bdd.False
    while store:
        log.info('covering...')
        assert all_new
        paths = store.pop(0)
        new = all_new.pop(0)
        rim = new & ~ covered
        covered = covered | new
        del new
        rim = rim & paths
        del paths
        transducer = transducer | rim
        del rim
    assert not store, store
    assert not all_new, all_new
    counter = aut.add_expr('{c} = {j}'.format(c=COUNTER, j=j))
    selector = aut.add_expr(SELECTOR)
    transducer = transducer & counter & (goal | ~ selector)
    log.info('-- done making strategy for goal: {j}'.format(j=j))
    return transducer


def and_exists(u, v, qvars, bdd):
    try:
        return _bdd.and_exists(u, v, qvars, bdd)
    except:
        r = u & v
        return bdd.quantify(r, qvars, forall=False)


def or_forall(u, v, qvars, bdd):
    try:
        return _bdd.or_forall(u, v, qvars, bdd)
    except:
        r = u | v
        return bdd.quantify(r, qvars, forall=True)


def disj(x, y):
    return x | y


def conj(x, y):
    return x & y


def var_order(bdd):
    """Return `dict` that maps each variable to a level.

    @rtype: `dict(str: int)`
    """
    return {var: bdd.level_of_var(var) for var in bdd.vars}


def memoize_iterates(z, aut):
    """Compute winning set, while storing iterates."""
    pass


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


def log_var_order(bdd):
    reordering_log = logging.getLogger(REORDERING_LOG)
    s = var_order(bdd)
    reordering_log.debug(repr(s))


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
    command_line_wrapper(input_fname)


def command_line_wrapper():
    """Solve game in `slugsin` file `fname`."""
    p = argparse.ArgumentParser()
    p.add_argument('file', type=str, help='`slugsin` input')
    args = p.parse_args()
    fname = args.file
    with open(fname, 'r') as f:
        s = f.read()
    solve_game(s)


def test_indices_and_levels():
    bdd = _bdd.BDD()
    ja = bdd.add_var('a', index=3)
    jb = bdd.add_var('b', index=10)
    jc = bdd.add_var('c', index=0)
    print(ja, jb, jc)
    print('a level', bdd.level_of_var('a'))
    print('b level', bdd.level_of_var('b'))
    print('c level', bdd.level_of_var('c'))
    u = bdd.var('a') & bdd.var('b')
    print str(u)
    print bdd.var_at_level(10)
