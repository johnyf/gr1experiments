#!/usr/bin/env python
"""Compare new solver `gr1x` to `slugs`."""
import argparse
import datetime
import pprint
import logging
import multiprocessing as mp
import sys
import time
from dd import cudd
from omega.logic import bitvector as bv
from omega.symbolic import symbolic
from openpromela import slugs
import psutil
from tugs import solver
from tugs import utils


GR1X_LOG = 'tugs.solver'


def run_parallel():
    first = 2
    problem = 'bunny_many_goals'
    solver = 'gr1x'
    if solver == 'slugs':
        output = 'runs_slugs'
        target = run_slugs
    elif solver == 'gr1x':
        output = 'runs'
        target = run_gr1x
    else:
        raise ValueError('unknown solver "{s}"'.format(s=solver))
    i_str = '{i}'
    slugsin_path = '{problem}/slugsin/{problem}_{i}.txt'.format(
        problem=problem, i=i_str)
    details_path = '{problem}/{output}/details_{i}.txt'.format(
        problem=problem, output=output, i=i_str)
    strategy_path = '{problem}/{output}/strategy.txt'.format(
        problem=problem, output=output, i=i_str)
    psutil_path = '{problem}/{output}/psutil_{i}.txt'.format(
        problem=problem, output=output, i=i_str)
    n_cpus = psutil.cpu_count(logical=False)
    # all_cpus = range(n_cpus)
    all_cpus = [0, 1, 3, 4, 5, 6, 7]
    concurrent = len(all_cpus)
    repetitions = 5
    final = first + concurrent * repetitions
    print('will run from {first} to {final}'.format(
        first=first, final=final))
    assert concurrent <= n_cpus, (concurrent, n_cpus)
    n = first
    for j in xrange(repetitions):
        m = n + concurrent
        print('spawn {n} to {last}'.format(n=n, last=m - 1))
        jobs = list()
        for i in xrange(n, m):
            d = dict(
                slugsin_file=slugsin_path.format(i=i),
                details_file=details_path.format(i=i),
                strategy_file=strategy_path,
                psutil_file=psutil_path.format(i=i))
            jobs.append(d)
        n = m
        procs = list()
        for d, cpu in zip(jobs, all_cpus):
            d['affinity'] = [cpu]
            p = mp.Process(target=target, kwargs=d)
            procs.append(p)
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('all joined')


def run_slugs(slugsin_file, strategy_file,
              psutil_file, details_file, affinity=None):
    """Run `slugs` for problem define in `slugsin_file`."""
    print('Starting: {fname}'.format(fname=slugsin_file))
    # other_options = ['--fixedPointRecycling']
    other_options = list()
    # config logging
    h_psutil = utils.add_logfile(psutil_file, 'openpromela.slugs')
    log = logging.getLogger('openpromela.slugs')
    log.setLevel('DEBUG')
    # capture execution environment
    versions = utils.snapshot_versions(check=False)
    log.info(pprint.pformat(versions))
    # run
    t0 = time.time()
    r = slugs._call_slugs(
        filename=slugsin_file,
        symbolic=True,
        strategy_file=strategy_file,
        affinity=affinity,
        logfile=details_file,
        other_options=other_options)
    t1 = time.time()
    dt = datetime.timedelta(seconds=t1 - t0)
    # close log files
    utils.close_logfile(h_psutil, 'openpromela.slugs')
    assert r is not None, 'NOT REALISABLE !!!'
    print('Done with: {fname} in {dt}'.format(
        fname=slugsin_file, dt=dt))


def run_gr1x(slugsin_file, strategy_file,
             details_file, affinity=None, **kw):
    """Run `gr1x` for problem define in `slugsin_file`."""
    win_set_file = 'winning_set'
    proc = psutil.Process()
    proc.cpu_affinity(affinity)
    # log verbosity
    level = logging.INFO
    log = logging.getLogger(GR1X_LOG)
    log.setLevel(level)
    # dump log
    h = logging.FileHandler(details_file, mode='w')
    log.addHandler(h)
    # log versions
    versions = utils.snapshot_versions(check=False)
    log.info(pprint.pformat(versions))
    # synthesize
    with open(slugsin_file, 'r') as f:
        s = f.read()
    t0 = time.time()
    solver.solve_game(
        s,
        win_set_fname=win_set_file,
        strategy_fname=strategy_file)
    t1 = time.time()
    dt = datetime.timedelta(seconds=t1 - t0)
    print('Done with: {fname} in {dt}'.format(
        fname=slugsin_file, dt=dt))
    # close log file
    log.removeHandler(h)
    h.close()
    sys.stdout.flush()


def run_gr1x_slugs_comparison(slugsin_file):
    """Check that both solvers return same winning set."""
    print('compare for: {f}'.format(f=slugsin_file))
    slugs_winning_set_file = 'winning_set_bdd.txt'
    slugs_strategy_file = 'slugs_strategy.txt'
    gr1x_strategy_file = 'gr1x_strategy.txt'
    utils.snapshot_versions()
    # call gr1x
    with open('slugsin_file', 'r') as f:
        slugsin = f.read()
    d = solver.parse_slugsin(slugsin)
    bdd = cudd.BDD()
    aut = solver.make_automaton(d, bdd)
    z = solver.compute_winning_set(aut)
    t = solver.construct_streett_transducer(z, aut)
    t.bdd.dump(t.action['sys'][0], gr1x_strategy_file)
    # call slugs
    symb = True
    slugs._call_slugs(slugsin_file, symb, slugs_strategy_file)
    # compare
    z_ = bdd.load(slugs_winning_set_file)
    assert z == z_, (z, z_)
    compare_strategies(
        slugsin, slugs_strategy_file, gr1x_strategy_file)


def compare_strategies(s, slugs_file, gr1x_file):
    """Check that both solvers return same strategy."""
    print('++ compare strategies')
    COUNTER = solver.COUNTER
    d = solver.parse_slugsin(s)
    n_goals = len(d['sys_win'])
    aux_vars = ['{c}_{i}'.format(c=COUNTER, i=i)
                for i in xrange(n_goals)]
    aux_vars.extend(
        ['{c}{i}'.format(c=COUNTER, i=i)
         for i in xrange(n_goals)])
    aux_vars.append(solver.SELECTOR)
    dvars = d['input'] + d['output'] + aux_vars
    # add primed
    dvars.extend(["{var}'".format(var=var) for var in dvars])
    bdd = cudd.BDD()
    for var in dvars:
        bdd.add_var(var)
    print('load slugs file')
    p = bdd.load(slugs_file)
    print('load gr1x file')
    q = bdd.load(gr1x_file)
    print('compare')
    dvars = {
        '{c}{i}'.format(c=COUNTER, i=i):
        '{c}_{i}'.format(c=COUNTER, i=i)
        for i in xrange(n_goals)}
    p = cudd.rename(p, bdd, dvars)
    table = {COUNTER: dict(dom=(0, n_goals), type='int', owner='sys')}
    table, _, _ = bv.bitblast_table(table)
    for j in xrange(n_goals):
        u = symbolic.cofactor(p, COUNTER, j, bdd, table)
        v = symbolic.cofactor(q, COUNTER, j, bdd, table)
        assert u == v
    print('-- done comparing strategies')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min', default=2, type=int,
                   help='from this # of masters')
    p.add_argument('--max', default=2, type=int,
                   help='to this # of masters')
    p.add_argument('--debug', type=int, default=logging.ERROR,
                   help='python logging level')
    p.add_argument('--repeat', default=1, type=int,
                   help='multiple runs from min to max')
    p.add_argument('--solver', default='slugs', type=str,
                   choices=['slugs', 'gr1x', 'compare'])
    args = p.parse_args()
    # multiple runs should be w/o plots
    assert args.repeat == 1 or not args.plot
    # multiple runs
    run_parallel()
    return
    for i in xrange(args.repeat):
        print('run: {i}'.format(i=i))
        if args.solver == 'slugs':
            run_slugs(args)
        elif args.solver == 'gr1x':
            run_gr1x(args)
        elif args.solver == 'compare':
            run_gr1x_slugs_comparison(args)
        else:
            raise Exception(
                'unknown solver: {s}'.format(s=args.solver))


if __name__ == '__main__':
    main()
