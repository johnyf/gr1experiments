#!/usr/bin/env python
"""Analyze logs and plot results."""
import argparse
from itertools import cycle
import os
import pickle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from tugs import utils


mpl.rc('xtick', labelsize=7)
mpl.rc('ytick', labelsize=7)
mpl.rc('font', size=7)
col_gen = cycle('bgrcmk')


def plot_report(repickle):
    paths = {
        # 'bunny/runs/': (2, 113),
        # 'bunny/runs_slugs/': (2, 63),
        # 'bunny_goals/runs/': (2, 32),
        # 'bunny_goals/runs_slugs/': (2, 33),
        # 'bunny_many_goals/runs/': (2, 49),
        # 'bunny_many_goals/runs_slugs/': (2, 41),
        # 'synt15/runs_slugs/': (2, 49),
        'synt15/runs/': (2, 50),
        #
        # 'synt15/runs_gr1x_linear_conj/': (2, 65),
        # 'synt15/runs_gr1x_logging_debug/': (2, 80),
        # 'synt15/runs_gr1x_logging_info/': (2, 105),
        # 'synt15/runs_gr1x_logging_info_missing_bdd/': (2, 65),
        # 'synt15/runs_gr1x_memoize/': (2, 65),
        # 'synt15/runs_gr1x_one_manager/': (2, 65),
        # 'bunny/runs/': (2, 97)
    }
    for path, (first, last) in paths.iteritems():
        plot_vs_parameter(path, first, last, repickle=repickle)


def plot_comparison_report():
    paths = dict(
        # new='bunny_many_goals/runs',
        # slugs='bunny_many_goals/runs_slugs'
        #
        # new='synt15/runs_gr1x_logging_info',
        # memoize='synt15/runs_gr1x_memoize',
        #
        # new='synt15/runs_gr1x_logging_info',
        # one_manager='synt15/runs_gr1x_one_manager',
        #
        # new='synt15/runs_gr1x_logging_info',
        # linear_conj='synt15/runs_gr1x_linear_conj',
        #
        new='synt15/runs_gr1x_logging_info',
        no_defer='synt15/runs',
        #
        # new='synt15/runs_gr1x_logging_info',
        # slugs='synt15/runs_slugs',
    )
    plot_comparison(paths)


def plot_vs_parameter(path, first, last, repickle=False):
    """Plot time, ratios, BDD nodes over parameterized experiments.

      - time
      - winning set computation / total time
      - reordering / total time
      - total nodes
      - peak nodes
    """
    measurements = pickle_results(path, first, last, repickle)
    # expand
    n_masters = measurements['n_masters']
    total_time = measurements['total_time']
    reordering_time_0 = measurements['reordering_time_0']
    reordering_time_1 = measurements['reordering_time_1']
    win_time = measurements['win_time']
    win_ratio = measurements['win_ratio']
    win_dump_ratio = measurements['win_dump_ratio']
    strategy_dump_ratio = measurements['strategy_dump_ratio']
    upratio_0 = measurements['upratio_0']
    upratio_1 = measurements['upratio_1']
    total_nodes_0 = measurements['total_nodes_0']
    total_nodes_1 = measurements['total_nodes_1']
    peak_nodes_0 = measurements['peak_nodes_0']
    peak_nodes_1 = measurements['peak_nodes_1']
    transducer_nodes = measurements['transducer_nodes']
    # plot
    fig_fname = '{path}stats.pdf'.format(path=path)
    fsz = 20
    tsz = 15
    fig = plt.figure()
    fig.set_size_inches(5, 10)
    plt.clf()
    plt.subplots_adjust(hspace=0.3)
    #
    # times
    ax = plt.subplot(3, 1, 1)
    plt.plot(n_masters, total_time, 'b-', label='Total time')
    if len(win_time) == len(n_masters):
        plt.plot(n_masters, win_time, 'r--',
                 label='Winning set fixpoint')
    if len(reordering_time_1):
        total_reordering_time = reordering_time_0 + reordering_time_1
    else:
        total_reordering_time = reordering_time_0
    if len(total_reordering_time) == len(n_masters):
        plt.plot(n_masters, total_reordering_time, 'g-o',
                 label='Total reordering time')
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid()
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Time (sec)', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # ratios
    ax = plt.subplot(3, 1, 2)
    if len(win_ratio) == len(n_masters):
        plt.plot(n_masters, win_ratio, 'b-.', label='Win / total time')
    if len(upratio_0) == len(n_masters):
        plt.plot(n_masters, upratio_0, 'b-o',
                 label='Reordering ratio (1)', markevery=10)
    if len(upratio_1) == len(n_masters):
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
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Ratios', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # nodes
    ax = plt.subplot(3, 1, 3)
    if len(total_nodes_0) == len(n_masters):
        plt.plot(n_masters, total_nodes_0, 'b-+',
                 label='Total (1)', markevery=10)
        plt.plot(n_masters, peak_nodes_0, 'b-*',
                 label='Peak (1)', markevery=10)
    if len(total_nodes_1) == len(n_masters):
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
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('BDD Nodes', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    # save
    plt.savefig(fig_fname, bbox_inches='tight')


def plot_comparison(paths):
    """Plot time, ratios, BDD nodes over parameterized experiments.

      - time
      - winning set computation / total time
      - reordering / total time
      - total nodes
      - peak nodes
    """
    assert len(paths) >= 2, paths
    data_paths = dict(paths)
    if 'numerator' in data_paths:
        data_paths.pop('numerator')
    else:
        paths['numerator'] = next(iter(paths))
    measurements = dict()
    for k, path in data_paths.iteritems():
        fname = 'data.pickle'.format(path=path)
        pickle_fname = os.path.join(path, fname)
        print('open "{f}"'.format(f=pickle_fname))
        with open(pickle_fname, 'r') as f:
            measurements[k] = pickle.load(f)
    # plot
    head, _ = os.path.split(path)
    fig_fname = os.path.join(head, 'comparison.pdf')
    fig = plt.figure()
    fig.set_size_inches(5, 12.5)
    plt.clf()
    plt.subplots_adjust(hspace=0.3)
    styles = ['b-', 'r--']
    for (k, d), style in zip(measurements.iteritems(), styles):
        plot_single_experiment_vs_parameter(d, k, style)
    plot_total_time_ratio(measurements, paths, data_paths)
    # save
    print('save "{f}"'.format(f=fig_fname))
    plt.savefig(fig_fname, bbox_inches='tight')


def plot_total_time_ratio(measurements, paths, data_paths):
    assert len(measurements) == 2, measurements
    ax = plt.subplot(5, 1, 5)
    ax.set_yscale('log')
    end = min(len(d['n_masters'])
              for d in measurements.itervalues())
    n_masters = xrange(2, end)
    numerator = paths['numerator']
    denominator = set(data_paths)
    denominator.remove(numerator)
    (denominator,) = denominator
    y = (
        measurements[numerator]['total_time'][2:end] /
        measurements[denominator]['total_time'][2:end])
    plt.plot(n_masters, y, 'b-')
    plt.plot([2, end], [1.0, 1.0], 'g--')
    # annotate
    fsz = 12
    tsz = 12
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    title = 'Ratio of total time\n({a} / {b})'.format(
        a=numerator, b=denominator)
    plt.ylabel(title, fontsize=fsz)


def plot_single_experiment_vs_parameter(measurements, name, style):
    fsz = 12
    tsz = 12
    # expand
    n_masters = measurements['n_masters']
    total_time = measurements['total_time']
    reordering_time_0 = measurements['reordering_time_0']
    reordering_time_1 = measurements['reordering_time_1']
    total_nodes_0 = measurements['total_nodes_0']
    total_nodes_1 = measurements['total_nodes_1']
    peak_nodes_0 = measurements['peak_nodes_0']
    peak_nodes_1 = measurements['peak_nodes_1']
    #
    # total time
    ax = plt.subplot(5, 1, 1)
    plt.plot(n_masters, total_time, style, label=name)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Total time (sec)', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # reordering time
    ax = plt.subplot(5, 1, 2)
    if len(reordering_time_1):
        total_reordering_time = reordering_time_0 + reordering_time_1
    else:
        total_reordering_time = reordering_time_0
    if len(total_reordering_time) == len(n_masters):
        plt.plot(n_masters, total_reordering_time, style,
                 label=name)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Total reordering time (sec)', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # peak BDD nodes
    ax = plt.subplot(5, 1, 3)
    if len(total_nodes_0) == len(n_masters):
        plt.plot(n_masters, peak_nodes_0, style + 'o',
                 label='{name} (1)'.format(name=name),
                 markevery=10)
    if len(total_nodes_1) == len(n_masters):
        plt.plot(n_masters, peak_nodes_1, style,
                 label='{name} (2)'.format(name=name),
                 markevery=10)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Peak BDD Nodes', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # peak BDD nodes
    ax = plt.subplot(5, 1, 4)
    if len(total_nodes_0) == len(n_masters):
        plt.plot(n_masters, total_nodes_0, style,
                 label='{name} (1)'.format(name=name),
                 markevery=10)
    if len(total_nodes_1) == len(n_masters):
        plt.plot(n_masters, total_nodes_1, style + 'o',
                 label='{name} (2)'.format(name=name),
                 markevery=10)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Total BDD Nodes', fontsize=fsz)
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)


def pickle_results(path, first, last, repickle):
    """Dump to pickle file all measurements from a directory."""
    log_fname = '{path}details_{i}.txt'.format(
        path=path, i='{i}')
    pickle_fname = '{path}data.pickle'.format(path=path)
    try:
        with open(pickle_fname, 'r') as f:
            measurements = pickle.load(f)
    except IOError:
        measurements = None
    if measurements and not repickle:
        print('found pickled data')
        return measurements
    measurements = dict(
        n_masters=list(),
        total_time=list(),
        reordering_time_0=list(),
        reordering_time_1=list(),
        win_time=list(),
        win_ratio=list(),
        win_dump_ratio=list(),
        strategy_dump_ratio=list(),
        upratio_0=list(),
        upratio_1=list(),
        total_nodes_0=list(),
        total_nodes_1=list(),
        peak_nodes_0=list(),
        peak_nodes_1=list(),
        transducer_nodes=list())
    n = first
    m = last + 1
    for i in xrange(n, m):
        fname = log_fname.format(i=i)
        try:
            data = utils.load_log_file(fname)
            print('open "{fname}"'.format(fname=fname))
        except:
            print('Skip: missing log file "{f}"'.format(
                f=fname))
            continue
        collect_measurements(data, measurements)
        measurements['n_masters'].append(i)
    # np arrays
    for k, v in measurements.iteritems():
        measurements[k] = np.array(v)
    # dump
    with open(pickle_fname, 'w') as f:
        pickle.dump(measurements, f)
    return measurements


def collect_measurements(data, measurements):
    """Collect measurements from `data`."""
    # total time
    t0 = data['parse_slugsin']['time'][0]
    t1 = data['make_transducer_end']['time'][0]
    t = t1 - t0
    measurements['total_time'].append(t)
    # winning set / total time
    if 'winning_set_start' in data:
        t0 = data['winning_set_start']['time'][0]
        t1 = data['winning_set_end']['time'][0]
        t_win = t1 - t0
        r = t_win / t
        measurements['win_ratio'].append(r)
        measurements['win_time'].append(t_win)
    # winning set dump time / total time
    if 'dump_winning_set_start' in data:
        t0 = data['dump_winning_set_start']['time'][0]
        t1 = data['dump_winning_set_end']['time'][0]
        t_dump_win = t1 - t0
        r = t_dump_win / t
        measurements['win_dump_ratio'].append(r)
    # strategy dump time / total time
    if 'dump_strategy_start' in data:
        t0 = data['dump_strategy_start']['time'][0]
        t1 = data['dump_strategy_end']['time'][0]
        t_dump_strategy = t1 - t0
        r = t_dump_strategy / t
        measurements['strategy_dump_ratio'].append(r)
    if 'transducer_nodes' in data:
        x = data['transducer_nodes']['value'][-1]
        measurements['transducer_nodes'].append(x)
    # construction time
    t0 = data['make_transducer_start']['time'][0]
    t1 = data['make_transducer_end']['time'][0]
    t_make = t1 - t0
    if 'reordering_time' in data:
        # reordering BDD 0
        rt = data['reordering_time']['value'][-1]
        r = rt / t
        measurements['upratio_0'].append(r)
        measurements['reordering_time_0'].append(rt)
        # total nodes 0
        tn = data['total_nodes']['value'][-1]
        measurements['total_nodes_0'].append(tn)
        # peak nodes 0
        p = data['peak_nodes']['value'][-1]
        measurements['peak_nodes_0'].append(p)
    else:
        print('Warning: no BDD manager 0')
    if 'b3_reordering_time' in data:
        # reordering BDD 1
        rt = data['b3_reordering_time']['value'][-1]
        r = rt / t_make
        measurements['upratio_1'].append(r)
        measurements['reordering_time_1'].append(rt)
        # total nodes 1
        tn = data['b3_total_nodes']['value'][-1]
        measurements['total_nodes_1'].append(tn)
        # peak nodes 1
        p = data['b3_peak_nodes']['value'][-1]
        measurements['peak_nodes_1'].append(p)
    else:
        print('Warning: no BDD manager 1')


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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--compare', action='store_true',
                   help='plot comparison of two directories')
    p.add_argument('--repickle', action='store_true',
                   help='ignore older pickled data')
    args = p.parse_args()
    if args.compare:
        plot_comparison_report()
    else:
        plot_report(args.repickle)
