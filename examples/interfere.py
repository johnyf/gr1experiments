#!/usr/bin/env python
"""Measure interference among concurrent solver instances."""
import argparse
import datetime
import pprint
import logging
import multiprocessing as mp
import sys
import time
import psutil
from tugs import solver
from tugs import utils
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


GB = 2**30
GR1X_LOG = 'tugs.solver'


def run_parallel():
    """Run concurrent instances, increasing their number."""
    print('run increasingly larger groups of instances.')
    problem = 'synt15'
    output = 'runs_testing'
    target = run_gr1x
    slugsin_file = '{problem}/slugsin/{problem}_20.txt'.format(
        problem=problem)
    details_file = '{problem}/{output}/details_{s}.txt'.format(
        problem=problem, output=output, s='{n}_{cpu}')
    strategy_file = '{problem}/{output}/strategy.txt'.format(
        problem=problem, output=output)
    n_cpus = psutil.cpu_count(logical=False)
    print('{n_cpus} physical CPUs'.format(n_cpus=n_cpus))
    n_cpus = psutil.cpu_count(logical=True)
    print('{n_cpus} logical CPUs'.format(n_cpus=n_cpus))
    for n in xrange(1, n_cpus + 1):
        print('trying {n} CPUs'.format(n=n))
        procs = list()
        for cpu in xrange(n):
            d = dict(
                affinity=[cpu],
                slugsin_file=slugsin_file,
                details_file=details_file.format(n=n, cpu=cpu),
                strategy_file=strategy_file)
            p = mp.Process(target=target, kwargs=d)
            procs.append(p)
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print('all joined')


def run_gr1x(slugsin_file, strategy_file,
             details_file, affinity=None, **kw):
    """Run `gr1x` instance with given affinity."""
    win_set_file = 'winning_set'
    proc = psutil.Process()
    proc.cpu_affinity(affinity)
    # log verbosity
    level = logging.ERROR
    log = logging.getLogger(GR1X_LOG)
    log.setLevel(level)
    level = logging.DEBUG
    log = logging.getLogger(__name__)
    log.setLevel(level)
    # dump log
    h = logging.FileHandler(details_file, mode='w')
    log.addHandler(h)
    # capture execution environment
    versions = utils.snapshot_versions(check=False)
    log.info(pprint.pformat(versions))
    # synthesize
    with open(slugsin_file, 'r') as f:
        s = f.read()
    t0 = time.time()
    log.info(dict(time=t0, parse_slugsin=True))
    solver.solve_game(
        s,
        win_set_fname=win_set_file,
        strategy_fname=strategy_file,
        max_memory=1 * GB)
    t1 = time.time()
    log.info(dict(time=t1, make_transducer_end=True))
    dt = datetime.timedelta(seconds=t1 - t0)
    print('Done with: {fname} in {dt}'.format(
        fname=slugsin_file, dt=dt))
    # close log file
    log.removeHandler(h)
    h.close()
    sys.stdout.flush()


def plot_saturation():
    """Plot time versus number of processors active."""
    print('plot saturating effect')
    fig_fname = 'cpu_saturation.pdf'
    problem = 'synt15'
    output = 'runs_testing'
    details_file = '{problem}/{output}/details_{s}.txt'.format(
        problem=problem, output=output, s='{n}_{cpu}')
    fig = plt.figure()
    fig.set_size_inches(5, 10)
    total_time = dict()
    n_cpus = psutil.cpu_count(logical=True)
    cpus = range(1, n_cpus + 1)
    for n in cpus:
        print('load data of {n} CPUs'.format(n=n))
        times = list()
        for cpu in xrange(n):
            fname = details_file.format(n=n, cpu=cpu)
            data = utils.load_log_file(fname)
            t0 = data['parse_slugsin']['time'][0]
            t1 = data['make_transducer_end']['time'][0]
            t = t1 - t0
            times.append(t)
        total_time[n] = times
        plt.plot([n] * len(times), times, 'bo')
    plt.xlabel('number of logical cores used')
    plt.ylabel('Time (sec)')
    plt.grid(True)
    plt.savefig(fig_fname, bbox_inches='tight')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true',
                   help='run instances and log measurements')
    args = p.parse_args()
    if args.run:
        run_parallel()
    plot_saturation()
