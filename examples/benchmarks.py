#!/usr/bin/env python
"""Compare new solver `gr1x` to `slugs`."""
import argparse
import datetime
import pprint
import logging
import shutil
import sys
import time
from dd import cudd
from omega.logic import bitvector as bv
from omega.symbolic import symbolic
from openpromela import logic
from openpromela import slugs
import psutil
from tugs import solver
from tugs import utils


GR1X_LOG = 'tugs.solver'
JSON_FILE = 'details.json'
INPUT_FILE = 'amba_conj.pml'
CONFIG_FILE = 'config.json'
N = 2
M = 17


def run_slugs(args):
    """Run `slugs` for a range of AMBA spec instances."""
    n = args.min
    m = args.max + 1
    # config logging
    level = args.debug
    loggers = ['openpromela.slugs']
    for logname in loggers:
        log = logging.getLogger(logname)
        log.setLevel(level)
    # capture execution environment
    utils.snapshot_versions()
    # run
    psutil_file = 'psutil.txt'
    details_file = 'details.txt'
    for i in xrange(n, m):
        print('starting {i} masters...'.format(i=i))
        bdd_file = 'bdd_{i}_masters.txt'.format(i=i)
        # log
        h_psutil = utils.add_logfile(psutil_file, 'openpromela.slugs')
        # run
        t0 = time.time()
        code = generate_code(i)
        r = logic.synthesize(code, symbolic=True, filename=bdd_file)
        t1 = time.time()
        dt = datetime.timedelta(seconds=t1 - t0)
        # close log files
        utils.close_logfile(h_psutil, 'openpromela.slugs')
        assert r is not None, 'NOT REALISABLE !!!'
        print('Done with {i} masters in {dt}.'.format(i=i, dt=dt))
        # copy log file
        i_psutil_file = 'log_{i}_masters.txt'.format(i=i)
        i_details_file = 'details_{i}_masters.txt'.format(i=i)
        shutil.copy(psutil_file, i_psutil_file)
        shutil.copy(details_file, i_details_file)


def run_gr1x(args):
    """Run `gr1x` for a range of AMBA spec instances."""
    n = args.min
    m = args.max + 1
    level = args.debug
    # capture execution environment
    versions = utils.snapshot_versions()
    # config log
    log = logging.getLogger(GR1X_LOG)
    log.setLevel(level)
    for i in xrange(n, m):
        print('starting {i} masters...'.format(i=i))
        # setup log
        details_log = 'details_{i}_masters.txt'.format(i=i)
        h = logging.FileHandler(details_log, mode='w')
        h.setLevel(level)
        log = logging.getLogger(GR1X_LOG)
        log.addHandler(h)
        log.info(pprint.pformat(versions))
        # synthesize
        code = generate_code(i)
        t0 = time.time()
        spec = logic.compile_spec(code)
        aut = slugs._symbolic._bitblast(spec)
        s = slugs._to_slugs(aut)
        solver.solve_game(s)
        # TODO: dump symbolic strategy
        # TODO: model check dumped strategy
        t1 = time.time()
        dt = datetime.timedelta(seconds=t1 - t0)
        print('Done synthesizing {i} masters in {dt}.'.format(
            i=i, dt=dt))
        # close log file
        log = logging.getLogger(GR1X_LOG)
        log.removeHandler(h)
        h.close()
        sys.stdout.flush()


def run_gr1x_slugs_comparison(args):
    """Check that both solvers return same winning set."""
    slugs_winning_set_file = 'winning_set_bdd.txt'
    slugs_strategy_file = 'slugs_strategy.txt'
    gr1x_strategy_file = 'gr1x_strategy.txt'
    n = args.min
    m = args.max + 1
    utils.snapshot_versions()
    for i in xrange(n, m):
        print('start {i} masters...'.format(i=i))
        # translate to slugsin
        code = generate_code(i)
        spec = logic.compile_spec(code)
        aut = slugs._symbolic._bitblast(spec)
        slugsin = slugs._to_slugs(aut)
        # call gr1x
        d = solver.parse_slugsin(slugsin)
        bdd = cudd.BDD()
        aut = solver.make_automaton(d, bdd)
        z = solver.compute_winning_set(aut)
        t = solver.construct_streett_transducer(z, aut)
        t.bdd.dump(t.action['sys'][0], gr1x_strategy_file)
        # call slugs
        logic.synthesize(code, symbolic=True,
                         filename=slugs_strategy_file)
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


def generate_code(i):
    # check if other users
    users = psutil.users()
    if len(users) > 1:
        print('warning: other users logged in'
              '(may start running expensive jobs).')
    s = load_jcss12_amba_code(i)
    return s


def load_test():
    fname = 'test.pml'
    with open(fname, 'r') as f:
        s = f.read()
    return s


def load_jcss12_amba_code(i):
    fname = 'jcss12/amba_{i}.txt'.format(i=i)
    with open(fname, 'r') as f:
        s = f.read()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min', default=N, type=int,
                   help='from this # of masters')
    p.add_argument('--max', default=M, type=int,
                   help='to this # of masters')
    p.add_argument('--debug', type=int, default=logging.ERROR,
                   help='python logging level')
    p.add_argument('--run', default=False, action='store_true',
                   help='synthesize')
    p.add_argument('--repeat', default=1, type=int,
                   help='multiple runs from min to max')
    p.add_argument('--solver', default='slugs', type=str,
                   choices=['slugs', 'gr1x', 'compare'])
    args = p.parse_args()
    # multiple runs should be w/o plots
    assert args.repeat == 1 or not args.plot
    # multiple runs
    if args.run:
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
