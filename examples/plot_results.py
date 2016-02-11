#!/usr/bin/env python
"""Analyze logs and plot results."""
import argparse
from itertools import cycle
import pickle
import logging
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from tugs import utils


mpl.rc('xtick', labelsize=7)
mpl.rc('ytick', labelsize=7)
mpl.rc('font', size=7)
col_gen = cycle('bgrcmk')
GR1X_LOG = 'tugs.solver'
JSON_FILE = 'details.json'
INPUT_FILE = 'amba_conj.pml'
CONFIG_FILE = 'config.json'
N = 2
M = 17


def plot_report():
    paths = {
        #'synt15/runs/': (2, 67),
        'synt15/runs_slugs/': (2, 20),
        #'bunny/runs/': (2, 97)
    }
    for path, (first, last) in paths.iteritems():
        plot_vs_parameter(path, first, last)


def plot_vs_parameter(path, first, last):
    """Plot time, ratios, BDD nodes over parameterized experiments.

      - time
      - winning set computation / total time
      - reordering / total time
      - total nodes
      - peak nodes
    """
    log_fname = '{path}details_{i}.txt'.format(
        path=path, i='{i}')
    fig_fname = '{path}stats.pdf'.format(path=path)
    pickle_fname = '{path}data.pickle'.format(path=path)
    n = first
    m = last + 1
    fsz = 20
    tsz = 15
    n_masters = list()
    total_time = list()
    reordering_time_0 = list()
    reordering_time_1 = list()
    win_time = list()
    win_ratio = list()
    win_dump_ratio = list()
    strategy_dump_ratio = list()
    upratio_0 = list()
    upratio_1 = list()
    total_nodes_0 = list()
    total_nodes_1 = list()
    peak_nodes_0 = list()
    peak_nodes_1 = list()
    transducer_nodes = list()
    for i in xrange(n, m):
        fname = log_fname.format(i=i)
        try:
            data = utils.load_log_file(fname)
            n_masters.append(i)
            print('open "{fname}"'.format(fname=fname))
        except:
            print('Skip: missing log file "{f}"'.format(
                f=fname))
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
        win_time.append(t_win)
        # winning set dump time / total time
        if 'dump_winning_set_start' in data:
            t0 = data['dump_winning_set_start']['time'][0]
            t1 = data['dump_winning_set_end']['time'][0]
            t_dump_win = t1 - t0
            r = t_dump_win / t
            win_dump_ratio.append(r)
        # strategy dump time / total time
        if 'dump_strategy_start' in data:
            t0 = data['dump_strategy_start']['time'][0]
            t1 = data['dump_strategy_end']['time'][0]
            t_dump_strategy = t1 - t0
            r = t_dump_strategy / t
            strategy_dump_ratio.append(r)
        if 'transducer_nodes' in data:
            x = data['transducer_nodes']['value'][-1]
            transducer_nodes.append(x)
        # construction time
        t0 = data['make_transducer_start']['time'][0]
        t1 = data['make_transducer_end']['time'][0]
        t_make = t1 - t0
        # reordering BDD 0
        rt = data['reordering_time']['value'][-1]
        r = rt / t
        upratio_0.append(r)
        reordering_time_0.append(rt)
        # total nodes 0
        tn = data['total_nodes']['value'][-1]
        total_nodes_0.append(tn)
        # peak nodes 0
        p = data['peak_nodes']['value'][-1]
        peak_nodes_0.append(p)
        if 'b3_reordering_time' in data:
            # reordering BDD 1
            rt = data['b3_reordering_time']['value'][-1]
            r = rt / t_make
            upratio_1.append(r)
            reordering_time_1.append(r)
            # total nodes 1
            tn = data['b3_total_nodes']['value'][-1]
            total_nodes_1.append(tn)
            # peak nodes 1
            p = data['b3_peak_nodes']['value'][-1]
            peak_nodes_1.append(p)
            bdd2 = True
        else:
            bdd2 = False
    # np arrays
    n_masters = np.array(n_masters)
    total_time = np.array(total_time)
    reordering_time_0 = np.array(reordering_time_0)
    reordering_time_1 = np.array(reordering_time_1)
    win_time = np.array(win_time)
    win_ratio = np.array(win_ratio)
    win_dump_ratio = np.array(win_dump_ratio)
    strategy_dump_ratio = np.array(strategy_dump_ratio)
    upratio_0 = np.array(upratio_0)
    upratio_1 = np.array(upratio_1)
    total_nodes_0 = np.array(total_nodes_0)
    total_nodes_1 = np.array(total_nodes_1)
    peak_nodes_0 = np.array(peak_nodes_0)
    peak_nodes_1 = np.array(peak_nodes_1)
    transducer_nodes = np.array(transducer_nodes)
    # plot
    fig = plt.figure()
    fig.set_size_inches(5, 10)
    plt.clf()
    plt.subplots_adjust(hspace=0.3)
    # times
    ax = plt.subplot(3, 1, 1)
    plt.plot(n_masters, total_time, 'b-', label='Total time')
    plt.plot(n_masters, win_time, 'r--', label='Winning set fixpoint')
    if len(reordering_time_1):
        total_reordering_time = reordering_time_0 + reordering_time_1
    else:
        total_reordering_time = reordering_time_0
    plt.plot(n_masters, total_reordering_time, 'g-o',
             label='Total reordering time')
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
             label='Reordering ratio (1)', markevery=10)
    if bdd2:
        plt.plot(n_masters, upratio_1, 'r--o',
                 label='Reordering ratio (2)', markevery=10)
    if len(win_dump_ratio) == len(n_masters):
        plt.plot(n_masters, win_dump_ratio, 'g-*',
                 label='Win set dump / total time')
    if len(strategy_dump_ratio) == len(n_masters):
        plt.plot(n_masters, strategy_dump_ratio, 'm--*',
                 label='Strategy dump / total time')
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
    if len(transducer_nodes) == len(n_masters):
        plt.plot(n_masters, transducer_nodes, 'g--o',
                 label='Strategy', markevery=10)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid()
    plt.xlabel('Number of masters', fontsize=fsz)
    plt.ylabel('BDD Nodes', fontsize=fsz)
    plt.legend(loc='upper left')
    # save
    plt.savefig(fig_fname, bbox_inches='tight')
    # dump
    d = dict(
        n_masters=n_masters,
        total_time=total_time,
        peak_nodes_0=peak_nodes_0,
        peak_nodes_1=peak_nodes_1)
    with open(pickle_fname, 'w') as f:
        pickle.dump(d, f)


def plot_multiple_experiments_vs_time(args):
    n = args.min
    m = args.max + 1
    for i in xrange(n, m):
        log_fname = 'details_{i}_masters.txt'.format(i=i)
        fig_fname = 'details_{i}.pdf'.format(i=i)
        plot_single_experiment_vs_time(log_fname, fig_fname)


def plot_single_experiment_vs_time(details_file, fig_file):
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
    t, total_nodes = utils.get_signal('b3_total_nodes', data)
    t = t - t_min
    plt.plot(t, total_nodes, 'r-o', label='Total nodes (2)',
             markevery=max(int(len(t) / n_markers), 1))
    # peak nodes (other BDD)
    t, peak_nodes = utils.get_signal('b3_peak_nodes', data)
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
    t, reordering_time = utils.get_signal('b3_reordering_time', data)
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
    plot_report()


if __name__ == '__main__':
    main()
