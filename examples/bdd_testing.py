import logging
import pickle
import pprint
import time
from dd import cudd as _bdd


log = logging.getLogger(__name__)
REORDERING_LOG = 'reorder'
COUNTER = '_jx_b'
SELECTOR = 'strat_type'
STRATEGY_FILE = 'tugs_strategy.txt'
USE_BINARY = True
GB = 10**30
MAX_MEMORY = 10 * GB


def main():
    max_memory = 10 * GB
    initial_cache_size = 2**18
    bdd = _bdd.BDD(
        max_memory=max_memory,
        initial_cache_size=initial_cache_size)
    # load variables and order
    pickle_fname = 'bug_vars.pickle'
    with open(pickle_fname, 'rb') as fid:
        data = pickle.load(fid)
    dvars = data['dvars']
    epvars = data['epvars']
    for var in dvars:
        bdd.add_var(var)
    _bdd.reorder(bdd, dvars)
    cfg = dict(reordering=False)
    bdd.configure(cfg)
    # log
    log_bdd(bdd)
    cfg = bdd.configure()
    log.info(cfg)
    cfg = dict(max_growth=1.7)
    cfg = bdd.configure(cfg)
    log.info(cfg)
    # load bdd
    file_name = 'bug_bdd.txt'
    u = load_bdd(file_name, bdd)
    # file_name = 'bug_bdd_old.txt'
    # u_ = load_bdd(file_name, bdd)
    # assert u == u_, (u, u_)
    # del u_
    file_name = 'sys_action.txt'
    sys_action = load_bdd(file_name, bdd)
    dvars_ = var_order(bdd)
    assert dvars == dvars_, (dvars, dvars_)
    cfg = bdd.configure()
    log.info(cfg)
    stats = bdd.statistics()
    pprint.pprint(stats)
    log.info('done')
    exercise_bdd(u, sys_action, epvars, bdd)
    # t.bdd.dump(action, STRATEGY_FILE)
    del u, sys_action


def load_bdd(file_name, bdd):
    u = bdd.load(file_name)
    return u


def exercise_bdd(u, sys_action, epvars, bdd):
    log.info('++ exercise_bdd')
    # bdd.dump(x, 'bug_bdd.txt')
    cfg = dict(
        reordering=True,
        garbage_collection=True)
    bdd.configure(cfg)
    print_more_details(bdd)
    x = and_exists(u, sys_action,
                   epvars, bdd)
    cfg = dict(
        reordering=True,
        garbage_collection=True)
    bdd.configure(cfg)
    print_more_details(bdd)
    log.info('-- exercise_bdd')


def print_more_details(bdd):
    cfg = bdd.configure()
    s = pprint.pformat(cfg)
    log.info(s)
    stats = bdd.statistics()
    s = pprint.pformat(stats)
    log.info(s)


def log_bdd(bdd, name=''):
    if log.getEffectiveLevel() > logging.INFO:
        return
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
        name + 'reordering_time': reordering_time,
        name + 'n_reorderings': n_reorderings,
        name + 'total_nodes': len(bdd),
        name + 'peak_nodes': peak_nodes}
    log.info(dlog)


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


def log_reordering(fname):
    reordering_fname = 'reordering_{f}'.format(f=fname)
    log = logging.getLogger(REORDERING_LOG)
    h = logging.FileHandler(reordering_fname, 'w')
    log.addHandler(h)
    log.setLevel('ERROR')


def log_var_order(bdd):
    reordering_log = logging.getLogger(REORDERING_LOG)
    s = var_order(bdd)
    reordering_log.debug(repr(s))


def var_order(bdd):
    """Return `dict` that maps each variable to a level.

    @rtype: `dict(str: int)`
    """
    return {var: bdd.level_of_var(var) for var in bdd.vars}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
