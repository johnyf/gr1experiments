#!/usr/bin/env python
"""Measure interference among concurrent solver instances."""
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
    """Measure the effect on runtime of multiple instances."""
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
    print('{n_cpus} physical CPUs')
    n_cpus = psutil.cpu_count(logical=True)
    print('{n_cpus} logical CPUs')
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
    """Run concurrent instances, increasing their number."""
    win_set_file = 'winning_set'
    proc = psutil.Process()
    # proc.cpu_affinity(affinity)
    # capture execution environment
    versions = utils.snapshot_versions(check=False)
    # config log
    level = logging.DEBUG
    log = logging.getLogger(GR1X_LOG)
    log.setLevel(level)
    # setup log
    h = logging.FileHandler(details_file, mode='w')
    h.setLevel(level)
    log = logging.getLogger(GR1X_LOG)
    log.addHandler(h)
    log.info(pprint.pformat(versions))
    # synthesize
    with open(slugsin_file, 'r') as f:
        s = f.read()
    t0 = time.time()
    solver.solve_game(
        s,
        win_set_fname=win_set_file,
        strategy_fname=strategy_file,
        max_memory=1 * GB)
    t1 = time.time()
    dt = datetime.timedelta(seconds=t1 - t0)
    print('Done with: {fname} in {dt}'.format(
        fname=slugsin_file, dt=dt))
    # close log file
    log = logging.getLogger(GR1X_LOG)
    log.removeHandler(h)
    h.close()
    sys.stdout.flush()


def plot_saturation():
    """Plot time versus number of processors active."""
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
        times = list()
        for cpu in xrange(n):
            fname = details_file.format(n=n, cpu=cpu)
            data = utils.load_log_file(fname)
            t0 = data['parse_slugsin']['time'][0]
            t1 = data['make_transducer_end']['time'][0]
            t = t1 - t0
            times.append(t)
        total_time[n] = times
        plt.plot([n] * len(times), times)
    plt.savefig(fig_fname, bbox_inches='tight')


if __name__ == '__main__':
    # run_parallel()
    plot_saturation()
