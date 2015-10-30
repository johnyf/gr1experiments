#!/usr/bin/env python
"""Compare new solver `gr1x` to `slugs`.

Analyze logs and plot results.
"""
import argparse
import datetime
from itertools import cycle
import pickle
import logging
import re
import shutil
import sys
import time
from dd import cudd
import matplotlib as mpl
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from omega.logic import bitvector as bv
from omega.symbolic import symbolic
from openpromela import logic
from openpromela import slugs
import psutil
from tugs import solver
from tugs import utils


mpl.rc('xtick', labelsize=7)
mpl.rc('ytick', labelsize=7)
mpl.rc('font', size=7)
col_gen = cycle('bgrcmk')
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
    # capture execution environment
    utils.snapshot_versions()
    # config log
    level = logging.DEBUG
    SOLVER_LOG = 'solver'
    log = logging.getLogger(SOLVER_LOG)
    log.setLevel(level)
    for i in xrange(n, m):
        print('starting {i} masters...'.format(i=i))
        # setup log
        details_log = 'details_{i}_masters.txt'.format(i=i)
        h = logging.FileHandler(details_log, mode='w')
        h.setLevel(level)
        log = logging.getLogger(SOLVER_LOG)
        log.addHandler(h)
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
        log = logging.getLogger(SOLVER_LOG)
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
    s = load_synt15_amba_code(i)
    # s = load_test()
    return s


def load_test():
    fname = 'test.pml'
    with open(fname, 'r') as f:
        s = f.read()
    return s


def load_synt15_amba_code(i):
    """Form open Promela code for AMBA instance with `i` masters."""
    fname = INPUT_FILE
    with open(fname, 'r') as f:
        s = f.read()
    # set number of masters
    j = i - 1
    newline = '#define N {j}'.format(j=j)
    code = re.sub('#define N.*', newline, s)
    # add multiple weak fairness assumptions
    code += form_progress(i)
    return code


def form_progress(i):
    """Return conjunction of LTL formulae for progress."""
    prog = ' && '.join(
        '[]<>(request[{k}] -> (master == {k}))'.format(k=k)
        for k in xrange(i))
    return 'assert ltl { ' + prog + ' }'


def jcss12_amba_code(i):
    fname = 'amba_{i}.txt'.format(i=i)
    with open(fname, 'r') as f:
        s = f.read()
    return s


def plot_trends_for_experiments(args):
    """Plot time, ratios, BDD nodes over parameterized experiments.

      - time
      - winning set computation / total time
      - reordering / total time
      - total nodes
      - peak nodes
    """
    n = args.min
    m = args.max + 1
    fsz = 20
    tsz = 15
    n_masters = list()
    total_time = list()
    win_ratio = list()
    upratio_0 = list()
    upratio_1 = list()
    total_nodes_0 = list()
    total_nodes_1 = list()
    peak_nodes_0 = list()
    peak_nodes_1 = list()
    for i in xrange(n, m):
        fname = 'details_{i}_masters.txt'.format(i=i)
        try:
            data = utils.load_log_file(fname)
            n_masters.append(i)
            print('open "{fname}"'.format(fname=fname))
        except:
            print('Skip: missing log file for {i} masters.'.format(i=i))
            continue
        # total time
        t0 = data['parse_slugsin']['time'][0]
        t1 = data['make_transducer_end']['time'][0]
        t = t1 - t0
        total_time.append(t)
        # winning set / total time
        t0 = data['winning_set_start']['time'][0]
        t1 = data['winning_set_end']['time'][0]
        t_win = t1 - t0
        r = t_win / t
        win_ratio.append(r)
        # construction time
        t0 = data['make_transducer_start']['time'][0]
        t1 = data['make_transducer_end']['time'][0]
        t_make = t1 - t0
        # uptime BDD 0
        rt = data['reordering_time']['value'][-1]
        r = rt / t
        upratio_0.append(r)
        # total nodes 0
        tn = data['total_nodes']['value'][-1]
        total_nodes_0.append(tn)
        # peak nodes 0
        p = data['peak_nodes']['value'][-1]
        peak_nodes_0.append(p)
        if 'other_reordering_time' in data:
            # uptime BDD 1
            rt = data['other_reordering_time']['value'][-1]
            r = rt / t_make
            upratio_1.append(r)
            # total nodes 1
            tn = data['other_total_nodes']['value'][-1]
            total_nodes_1.append(tn)
            # peak nodes 1
            p = data['other_peak_nodes']['value'][-1]
            peak_nodes_1.append(p)
            bdd2 = True
        else:
            bdd2 = False
    # np arrays
    n_masters = np.array(n_masters)
    total_time = np.array(total_time)
    win_ratio = np.array(win_ratio)
    upratio_0 = np.array(upratio_0)
    upratio_1 = np.array(upratio_1)
    total_nodes_0 = np.array(total_nodes_0)
    total_nodes_1 = np.array(total_nodes_1)
    peak_nodes_0 = np.array(peak_nodes_0)
    peak_nodes_1 = np.array(peak_nodes_1)
    # plot
    fig = plt.figure()
    fig.set_size_inches(5, 10)
    plt.clf()
    plt.subplots_adjust(hspace=0.3)
    # times
    ax = plt.subplot(3, 1, 1)
    plt.plot(n_masters, total_time, 'b-', label='Total time')
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid()
    plt.xlabel('Number of masters', fontsize=fsz)
    plt.ylabel('Time (sec)', fontsize=fsz)
    plt.legend(loc='upper left')
    # ratios
    ax = plt.subplot(3, 1, 2)
    plt.plot(n_masters, win_ratio, 'b-.', label='Win / total time')
    plt.plot(n_masters, upratio_0, 'b-o',
             label='Up ratio (1)', markevery=10)
    if bdd2:
        plt.plot(n_masters, upratio_1, 'r--o',
                 label='Up ratio (2)', markevery=10)
    # annotate
    # ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid()
    ax.set_ylim([0, 1])
    plt.xlabel('Number of masters', fontsize=fsz)
    plt.ylabel('Ratios', fontsize=fsz)
    plt.legend(loc='upper left')
    # nodes
    ax = plt.subplot(3, 1, 3)
    plt.plot(n_masters, total_nodes_0, 'b-+',
             label='Total (1)', markevery=10)
    plt.plot(n_masters, peak_nodes_0, 'b-*',
             label='Peak (1)', markevery=10)
    if bdd2:
        plt.plot(n_masters, total_nodes_1, 'r--+',
                 label='Total (2)', markevery=10)
        plt.plot(n_masters, peak_nodes_1, 'r--*',
                 label='Peak (2)', markevery=10)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid()
    plt.xlabel('Number of masters', fontsize=fsz)
    plt.ylabel('BDD Nodes', fontsize=fsz)
    plt.legend(loc='upper left')
    # save
    fname = 'stats.pdf'
    plt.savefig(fname, bbox_inches='tight')
    # dump
    d = dict(
        n_masters=n_masters,
        total_time=total_time,
        peak_nodes_0=peak_nodes_0,
        peak_nodes_1=peak_nodes_1)
    with open('data.pickle', 'w') as f:
        pickle.dump(d, f)


def plot_single_experiment(details_file, i, fig_file):
    """Plot BDD node changes during an experiment.

    For each BDD manager:

      - total nodes
      - peak nodes
      - reordering / total time
    """
    data = utils.load_log_file(details_file)
    # total time interval
    t_min = min(v['time'][0] for v in data.itervalues())
    t_max = max(v['time'][-1] for v in data.itervalues())
    # subplot arrangement
    n = 3
    n_markers = 3
    fig = plt.figure()
    fig.set_size_inches(5, 10)
    plt.clf()
    plt.subplots_adjust(hspace=0.3)
    ax = plt.subplot(n, 1, 1)
    # total nodes
    t, total_nodes = utils.get_signal('total_nodes', data)
    t = t - t_min
    plt.plot(t, total_nodes, 'b-', label='Total nodes (1)')
    # peak nodes
    t, peak_nodes = utils.get_signal('peak_nodes', data)
    t = t - t_min
    plt.plot(t, peak_nodes, 'b--', label='Peak nodes (1)')
    # total nodes (other BDD)
    t, total_nodes = utils.get_signal('other_total_nodes', data)
    t = t - t_min
    plt.plot(t, total_nodes, 'r-o', label='Total nodes (2)',
             markevery=max(int(len(t) / n_markers), 1))
    # peak nodes (other BDD)
    t, peak_nodes = utils.get_signal('other_peak_nodes', data)
    t = t - t_min
    plt.plot(t, peak_nodes, 'r--o', label='Peak nodes (2)',
             markevery=max(int(len(t) / n_markers), 1))
    # annotate
    ax.set_yscale('log')
    plt.grid()
    plt.xlabel('Time (sec)')
    plt.ylabel('BDD nodes')
    plt.legend(loc='upper left')
    # uptime ratio
    ax = plt.subplot(n, 1, 2)
    t, reordering_time = utils.get_signal('reordering_time', data)
    t = t - t_min
    y = reordering_time / t
    plt.plot(t, y, label='BDD 1')
    # uptime ratio (other BDD)
    ax = plt.subplot(n, 1, 2)
    t, reordering_time = utils.get_signal('other_reordering_time', data)
    t_min_other = t[0]
    t_other = t - t_min_other
    y = reordering_time / t_other
    t = t - t_min
    plt.plot(t, y, 'r--', label='BDD 2')
    # annotate
    ax.set_yscale('log')
    plt.grid()
    plt.xlabel('Time (sec)')
    plt.ylabel('Reordering / total time')
    plt.legend(loc='upper left')
    # save
    plt.savefig(fig_file, bbox_inches='tight')


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
    p.add_argument('--plot', action='store_true',
                   help='generate plots')
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
    # plot
    if args.plot:
        plot_trends_for_experiments(args)
        # i = args.min
        # fname = 'details_{i}_masters.txt'.format(i=i)
        # fig_file = 'test.pdf'
        # plot_single_experiment(fname, i, fig_file)


if __name__ == '__main__':
    main()
