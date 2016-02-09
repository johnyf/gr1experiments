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


def plot_multiple_experiments(args):
    n = args.min
    m = args.max + 1
    for i in xrange(n, m):
        log_fname = 'details_{i}_masters.txt'.format(i=i)
        fig_fname = 'details_{i}.pdf'.format(i=i)
        plot_single_experiment(log_fname, fig_fname)


def plot_single_experiment(details_file, fig_file):
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
    if args.plot:
        # plot_trends_for_experiments(args)
        plot_multiple_experiments(args)


if __name__ == '__main__':
    main()
