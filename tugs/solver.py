import argparse
import copy
import datetime
import logging
import math
import os
import time
from dd import cudd as _bdd
import humanize
import natsort
from omega.logic import syntax
from omega.symbolic import symbolic
import psutil
import __builtin__
try:
    __builtin__.profile
except AttributeError:
    def profile(func):
        return func


log = logging.getLogger(__name__)
REORDERING_LOG = 'reorder'
COUNTER = '_jx_b'
SELECTOR = 'strat_type'
WINNING_SET_FILE = 'winning_set'
STRATEGY_FILE = 'tugs_strategy.dddmp'
USE_BINARY = True
GB = 2**30
MAX_MEMORY = 10 * GB
INIT_CACHE = 2**18

# TODO:
#
# 2. implement `slugs` algorithm in Cython
#
# 3. compare 2 to `slugs`
#
# 4. add a case study
#
# group primed and unprimed vars
# use efficient rename for neighbors
# use a CUDD map for repeated renaming
#
# init of counter and strategy_type
# allow passing a desired level for the first bit
#     of an integer


@profile
def solve_game(s, win_set_fname=None):
    """Construct transducer for game in file `fname`.

    @param s: `str` in `slugs` syntax
    """
    d = parse_slugsin(s)
    bdd = _bdd.BDD(
        memory_estimate=MAX_MEMORY,
        initial_cache_size=INIT_CACHE)
    bdd.configure(
        max_memory=MAX_MEMORY,
        max_growth=1.7)
    log.info(bdd.configure())
    aut = make_automaton(d, bdd)
    log_bdd(bdd)
    if win_set_fname is None:
        z = compute_winning_set(aut)
        dump_winning_set(z, bdd)
    else:
        z = _bdd.load(win_set_fname, bdd)
    log_bdd(bdd)
    assert z != bdd.false, 'unrealizable'
    t = construct_streett_transducer(z, aut)
    dump_strategy(t)
    del z


def dump_winning_set(z, bdd):
    log.debug('++ dump_winning_set')
    t0 = time.time()
    memory = 3 * GB
    b = _bdd.BDD(memory_estimate=memory)
    b.configure(max_memory=memory, reordering=False)
    _bdd.copy_vars(bdd, b)
    order = var_order(bdd)
    _bdd.reorder(b, order)
    u = _bdd.copy_bdd(z, bdd, b)
    _bdd.dump(u, WINNING_SET_FILE, b)
    del u
    t1 = time.time()
    dt = t1 - t0
    log.info(
        'Winning set dumped in {dt:1.2} sec'.format(
            dt=dt))
    log.debug('-- done dump_winning_set')


def dump_strategy(t):
    t0 = time.time()
    action = t.action['sys'][0]
    t.bdd.dump(action, STRATEGY_FILE)
    t1 = time.time()
    dt = t1 - t0
    log.info(
        'Strategy dumped in {dt:1.2} sec.'.format(dt=dt))


def log_reordering(fname):
    reordering_fname = 'reordering_{f}'.format(f=fname)
    log = logging.getLogger(REORDERING_LOG)
    h = logging.FileHandler(reordering_fname, 'w')
    log.addHandler(h)
    log.setLevel('ERROR')


def parse_slugsin(s):
    """Return `dict` keyed by `slugsin` file section."""
    log_event(parse_slugsin=True)
    sections = dict(
        INPUT='input',
        OUTPUT='output',
        ENV_INIT='env_init',
        SYS_INIT='sys_init',
        ENV_TRANS='env_action',
        SYS_TRANS='sys_action',
        ENV_LIVENESS='<>[]',
        SYS_LIVENESS='[]<>')
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
    log.info('-- done parse_slugsin')
    return d


def make_automaton(d, bdd):
    """Return `symbolic.Automaton` from slugsin spec.

    @type d: dict(str: list)
    """
    log_event(make_automaton=True)
    # bits -- shouldn't produce safety or init formulae
    a = symbolic.Automaton()
    a.vars = _init_vars(d)
    a = symbolic._bitblast(a)
    # formulae
    sections = symbolic._make_section_map(a)
    for section, target in sections.iteritems():
        target.extend(d[section])
    a.conjoin('prefix')
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


@profile
def compute_winning_set(aut, z=None):
    """Compute winning region, w/o memoizing iterates."""
    log_event(winning_set_start=True)
    # reordering_log = logging.getLogger(REORDERING_LOG)
    bdd = aut.bdd
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    if z is None:
        z = bdd.true
    zold = None
    log.debug('Before z fixpoint')
    while z != zold:
        log.debug('Start Z iteration')
        # s = var_order(bdd)
        # reordering_log.debug(repr(s))
        zold = z
        yj = list()
        for j, goal in enumerate(aut.win['[]<>']):
            log.info('Goal: {j}'.format(j=j))
            # log.info(bdd)
            zp = _bdd.rename(z, bdd, aut.prime)
            live_trans = goal & zp
            y = bdd.false
            yold = None
            while y != yold:
                log.debug('Start Y iteration')
                yold = y
                yp = _bdd.rename(y, bdd, aut.prime)
                live_trans = live_trans | yp
                good = y
                for i, excuse in enumerate(aut.win['<>[]']):
                    x = bdd.true
                    xold = None
                    while x != xold:
                        log.debug('Start X iteration')
                        xold = x
                        xp = _bdd.rename(x, bdd, aut.prime)
                        # desired transitions
                        x = xp & excuse
                        x = x | live_trans
                        # reordering_log.debug(repr(s))
                        '''
                        dvars = var_order(bdd)
                        epvars = aut.epvars
                        pickle_data = dict(
                            dvars=dvars,
                            epvars=epvars)
                        pickle_fname = 'bug_vars.pickle'
                        with open(pickle_fname, 'wb') as fid:
                            pickle.dump(pickle_data,
                                        fid, protocol=2)
                        bdd_fname = 'bug_bdd.txt'
                        bdd.dump(x, bdd_fname)
                        bdd_fname = 'sys_action.txt'
                        bdd.dump(sys_action, bdd_fname)
                        bdd.configure(
                            reordering=False,
                            garbage_collection=False)
                        log.info(bdd.configure())
                        stats = bdd.statistics()
                        log.info(stats)
                        '''
                        x = and_exists(x, sys_action,
                                       aut.epvars, bdd)
                        '''
                        bdd.configure(
                            reordering=True,
                            garbage_collection=True)
                        '''
                        x = or_forall(x, ~ env_action,
                                      aut.upvars, bdd)
                        log_loop(i, j, None, x, y, z)
                        log_bdd(bdd, '')
                    log.debug('Reached X fixpoint')
                    del xold
                    good = good | x
                    del x
                y = good
                del good
            log.debug('Reached Y fixpoint')
            del yold, live_trans
            if USE_BINARY:
                yj.append(y)
            else:
                z = z & y
            del y, goal
        del zp
        z = syntax.recurse_binary(conj, yj)
        # z = syntax._linear_operator_simple(conj, yj)
        # bdd.assert_consistent()
    log.info('Reached Z fixpoint')
    log_bdd(bdd, '')
    log_event(winning_set_end=True)
    return z


@profile
def construct_streett_transducer(z, aut):
    """Return Street(1) I/O transducer."""
    log_event(make_transducer_start=True)
    # reordering_log = logging.getLogger(REORDERING_LOG)
    # copy vars
    bdd = aut.bdd
    # one more manager
    b3 = _bdd.BDD(memory_estimate=MAX_MEMORY)
    b3.configure(max_memory=MAX_MEMORY)
    _bdd.copy_vars(bdd, b3)
    # copy var order
    # order = var_order(bdd)
    # _bdd.reorder(b3, order)
    # copy actions with reordering off
    env_action = aut.action['env'][0]
    sys_action = aut.action['sys'][0]
    sys_action_2 = _bdd.copy_bdd(sys_action, bdd, b3)
    env_action_2 = _bdd.copy_bdd(env_action, bdd, b3)
    # Compute iterates, now that we know the outer fixpoint
    log_bdd(b3, 'b3_')
    log.info('done copying actions')
    zp = _bdd.rename(z, bdd, aut.prime)
    # transducer automaton
    t = symbolic.Automaton()
    t.vars = copy.deepcopy(aut.vars)
    t.vars[SELECTOR] = dict(type='bool', owner='sys', level=0)
    n_goals = len(aut.win['[]<>'])
    t.vars[COUNTER] = dict(
        type='saturating', dom=(0, n_goals - 1),
        owner='sys', level=0)
    t = t.build(b3, add=True)
    transducers = list()
    selector = t.add_expr(SELECTOR)
    max_vars = 20
    b3.configure(max_vars=max_vars)
    for j, goal in enumerate(aut.win['[]<>']):
        log.info('Goal: {j}'.format(j=j))
        log_bdd(bdd, '')
        # for fixpoint
        live_trans = goal & zp
        y = bdd.false
        yold = None
        # for strategy construction
        covered = b3.false
        transducer = b3.false
        while y != yold:
            log.debug('Start Y iteration')
            yold = y
            yp = _bdd.rename(y, bdd, aut.prime)
            live_trans = live_trans | yp
            good = y
            for i, excuse in enumerate(aut.win['<>[]']):
                x = bdd.true
                xold = None
                paths = None
                new = None
                while x != xold:
                    del paths, new
                    log.debug('Start X iteration')
                    xold = x
                    xp = _bdd.rename(x, bdd, aut.prime)
                    x = xp & excuse
                    del xp
                    paths = x | live_trans
                    new = and_exists(paths, sys_action,
                                     aut.epvars, bdd)
                    x = or_forall(new, ~ env_action,
                                  aut.upvars, bdd)
                    log_loop(i, j, None, x, y, z)
                    log_bdd(bdd, '')
                log.debug('Reached X fixpoint')
                del xold, excuse
                good = good | x
                del x
                # strategy construction
                # in `b3`
                log.info('transfer `paths` to `b3`')
                paths = _bdd.copy_bdd(paths, bdd, b3)
                new = _bdd.copy_bdd(new, bdd, b3)
                log.info('done transferring')
                rim = new & ~ covered
                covered = covered | new
                del new
                rim = rim & paths
                del paths
                transducer = transducer | rim
                del rim
            y = good
            del good
        log.debug('Reached Y fixpoint (Y = Z)')
        assert y == z, (y, z)
        del y, yold, covered
        log_bdd(b3, 'b3_')
        # make transducer
        goal = _bdd.copy_bdd(goal, bdd, b3)
        e = '{c} = {j}'.format(c=COUNTER, j=j)
        counter = t.add_expr(e)
        u = goal | ~ selector
        del goal
        u = counter & u
        del counter
        transducer = transducer & u
        del u
        transducer = transducer & sys_action_2
        # check_winning_region(transducer, aut, t,
        #                      bdd, other_bdd, z, j)
        transducers.append(transducer)
        # s = var_order(other_bdd)
        # reordering_log.debug(repr(s))
        del transducer
    del sys_action_2, zp
    log_bdd(b3, 'b3_')
    log.info('disjoin transducers')
    transducer = syntax.recurse_binary(disj, transducers)
    log.info('done with disjunction')
    # transducer = syntax._linear_operator_simple(
    #   disj, transducers)
    n_remain = len(transducers)
    assert n_remain == 0, n_remain
    # add counter limits
    # transducer = transducer & t.action['sys'][0]
    # env lost ?
    # transducer = transducer | ~ env_action_2
    t.action['sys'] = [transducer]
    n_nodes = len(transducer)
    print('Transducer BDD: {n} nodes'.format(n=n_nodes))
    log_bdd(bdd, '')
    log_bdd(b3, 'b3_')
    log_event(make_transducer_end=True)
    # self-check
    # check_winning_region(transducer, aut, t, bdd,
    #                      other_bdd, z, 0)
    del selector, env_action_2, transducer
    return t


def log_event(**d):
    """Log `dict` `d` with timestamp."""
    t = time.time()
    dlog = dict(d)
    dlog['time'] = t
    log.info('')
    log.info(dlog)
    date = datetime.datetime.fromtimestamp(t)
    s = date.strftime('%Y-%m-%d %H:%M:%S')
    log.info(s)  # for direct reading by humans


def log_loop(i, j, transducer, x, y, z):
    if log.getEffectiveLevel() > logging.INFO:
        return
    if transducer is not None:
        transducer_nodes = len(transducer)
    else:
        transducer_nodes = None
    t = time.time()
    dlog = dict(
        time=t,
        goal=j,
        excuse=i,
        transducer_nodes=transducer_nodes,
        x_nodes=len(x),
        y_nodes=len(y),
        z_nodes=len(z))
    log.info(dlog)


def log_bdd(bdd, name=''):
    if log.getEffectiveLevel() > logging.INFO:
        return
    # `psutil` used as in `openpromela.slugsin`
    pid = os.getpid()
    proc = psutil.Process(pid)
    rss, vms = proc.memory_info()
    try:
        stats = bdd.statistics()
        reordering_time = float(stats['reordering_time'])
        n_reorderings = int(stats['n_reorderings'])
        peak_nodes = int(stats['peak_nodes'])
    except AttributeError:
        # using `autoref`
        reordering_time = None
        peak_nodes = None
    t = time.time()
    dlog = {
        'time': t,
        'rss': humanize.naturalsize(rss),
        'vms': humanize.naturalsize(vms),
        name + 'reordering_time': reordering_time,
        name + 'n_reorderings': n_reorderings,
        name + 'total_nodes': len(bdd),
        name + 'peak_nodes': peak_nodes}
    log.info(dlog)


def check_winning_region(transducer, aut, t, bdd,
                         other_bdd, z, j):
    u = transducer
    u = symbolic.cofactor(transducer, COUNTER, j,
                          other_bdd, t.vars)
    u = other_bdd.quantify(u, [SELECTOR], forall=False)
    u = other_bdd.quantify(u, t.epvars, forall=False)
    u = other_bdd.quantify(u, t.upvars, forall=True)
    z_ = _bdd.copy_bdd(z, bdd, other_bdd)
    print('u == z', u == z_)


def old_cox(x, env_action, sys_action, aut):
    bdd = aut.bdd
    x = x & sys_action
    x = bdd.quantify(x, aut.epvars, forall=False)
    x = x | ~ env_action
    x = bdd.quantify(x, aut.upvars, forall=True)
    return x


def recurse_binary(f, x, bdds):
    """Recursively traverse binary tree of computation."""
    log.debug('++ recurse binary')
    n = len(x)
    log.debug('{n} items left to recurse'.format(n=n))
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
    log.debug(
        '-- done recurse binary ({n} items)'.format(n=n))
    return f(cpa, cpb), new_bdd


def make_strategy(store, all_new, j, goal, aut):
    log.info('++ Make strategy for goal: {j}'.format(j=j))
    bdd = aut.bdd
    log.info(bdd)
    covered = bdd.false
    transducer = bdd.false
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
    log.info(
        '-- done making strategy for goal: {j}'.format(j=j))
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


def var_order(bdd):
    """Return `dict` that maps each variable to a level.

    @rtype: `dict(str: int)`
    """
    return {var: bdd.level_of_var(var) for var in bdd.vars}


def main():
    fname = 'reordering_slugs_31.txt'
    other_fname = 'reordering_slugs_31_old.txt'
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str,
                   help='slugsin input file')
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
    p.add_argument('--win_set', type=str,
                   help='winning set BDD as DDDMP file')
    p.add_argument('--debug', type=int, help='logging level')
    args = p.parse_args()
    log.setLevel(level=args.debug)
    h = logging.StreamHandler()
    log.addHandler(h)
    fname = args.file
    with open(fname, 'r') as f:
        s = f.read()
    win_set_fname = args.win_set
    solve_game(s, win_set_fname=win_set_fname)


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
