#!/usr/bin/env python
import argparse
import copy
import logging
import time
import cudd
from cudd import BDD
import natsort
from omega.logic import syntax
from omega.symbolic import symbolic


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
    # bits -- shouldn't produce safety or init formulae
    a = symbolic.Automaton()
    a.vars = _init_vars(d)
    a = symbolic._bitblast(a)
    # formulae
    sections = symbolic._make_section_map(a)
    for section, target in sections.iteritems():
        target.extend(d[section])
    a.conjoin('prefix')
    print(a)
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
        yj = list()
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
            # z = z & y
            yj.append(y)
            del y, yold, live_trans
        del zp
        # conjoin
        z = syntax.recurse_binary(lambda x, y: x & y, yj)
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


def construct_streett_transducer(z, aut):
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
    store = dict()
    all_new = dict()
    zp = cudd.rename(z, bdd, aut.prime)
    for j, goal in enumerate(aut.win['sys']):
        log.info('Goal: {j}'.format(j=j))
        store[j] = list()
        all_new[j] = list()
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
                store[j].append(paths)
                all_new[j].append(new)
            y = good
        # is it more efficient to do this now, or later ?
        # problem is that it couples with more variables (the counters)
    print('done, lets construct strategies now')
    # transducer automaton
    print('sys action has {n} nodes'.format(n=len(sys_action)))
    sys_action = cudd.transfer_bdd(sys_action, other_bdd)
    print('done transferring')
    t = symbolic.Automaton()
    t.vars = copy.deepcopy(aut.vars)
    t.vars['strat_type'] = dict(type='bool', owner='sys')
    n_goals = len(aut.win['sys'])
    t.vars['c'] = dict(type='saturating', dom=(0, n_goals - 1), owner='sys')
    t = t.build(other_bdd, add=True)
    selector = t.add_expr('strat_type')
    transducers = list()
    # construct strategies
    for j, goal in enumerate(aut.win['sys']):
        log.info('Goal: {j}'.format(j=j))
        log.info(other_bdd)
        covered = other_bdd.False
        transducer = other_bdd.False
        cur_store = store[j]
        cur_new = all_new[j]
        while cur_store:
            assert cur_new, cur_new
            paths = cur_store.pop(0)
            new = cur_new.pop(0)
            print('covering...')
            rim = new & ~ covered
            covered = covered | new
            del new
            transducer = transducer | (rim & paths)
            del rim, paths
        log.info('appending transducer for this goal')
        counter = t.add_expr('c = {j}'.format(j=j))
        goal = cudd.transfer_bdd(goal, other_bdd)
        transducer = transducer & counter & (goal | ~ selector)
        transducers.append(transducer)
        del covered
    log.info(other_bdd)
    log.info('clean intermediate results')
    for j in store:
        assert not store[j], (j, store[j])
        assert not all_new[j], (j, all_new[j])
    # insert the counters at the top of the order
    # semi-symbolic representation ?
    log.info(other_bdd)
    # disjoin the strategies for the individual goals
    # transducer = linear_operator(lambda x, y: x | y, transducers)
    log.info('disjoin transducers')
    transducer = syntax.recurse_binary(lambda x, y: x | y,
                                       transducers, other_bdd)
    print(len(transducer))
    print(other_bdd)
    print(bdd)
    n_remain = len(transducers)
    assert n_remain == 0, n_remain
    log.info(other_bdd)
    log.info('transfer bdd')
    log.info('conjoin with sys action')
    transducer = transducer & sys_action
    log.info(other_bdd)
    log.info(transducer)
    del sys_action
    # TODO: init of counter and strategy_type
    # conjoin with counter limits
    log.info('final conjunction')
    transducer = transducer & t.action['sys'][0]
    t.action['sys'] = [transducer]
    return t


def solve_game(fname):
    """Construct transducer for game in file `fname`."""
    d = load_slugsin_file(fname)
    bdd = BDD()
    aut = make_automaton(d, bdd)
    # aut.action['sys'][0] = bdd.False
    z = compute_winning_set(aut)
    t = construct_streett_transducer(z, aut)
    print(t)
    del aut, z, t


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
    # syntax
    log = logging.getLogger('omega.logic.syntax')
    log.addHandler(logging.StreamHandler())
    log.setLevel('DEBUG')
    solve_game(input_fname)


def test_indices_and_levels():
    bdd = cudd.BDD()
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


if __name__ == '__main__':
    main()
