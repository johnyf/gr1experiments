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
        # 'bunny_many_goals/runs_slugs/': (2, 49),
        # 'synt15/runs_slugs/': (2, 49),
        # 'synt15/runs_slugs_browne/': (2, 41),
        #
        'synt15/runs/': (2, 60),
        #
        # 'synt15/runs_gr1x_linear_conj/': (2, 65),
        # 'synt15/runs_gr1x_logging_debug/': (2, 80),
        # 'synt15/runs_gr1x_logging_info/': (2, 105),
        # 'synt15/runs_gr1x_logging_info_missing_bdd/': (2, 65),
        # 'synt15/runs_gr1x_memoize/': (2, 65),
        # 'synt15/runs_gr1x_one_manager/': (2, 65),
        # 'bunny/runs/': (2, 97)
        # 'cinderella/runs': (0, 10),
        # 'cinderella/runs_slugs': (0, 8)
        # 'jcss12/runs': (2, 22),
        # 'jcss12/runs_gr1x_no_defer': (2, 25),
        # 'jcss12/runs_slugs': (2, 16),
        #
        # 'genbuf/runs': (2, 90),
        # 'genbuf/runs_slugs': (2, 63),
    }
    for path, (first, last) in paths.iteritems():
        plot_vs_parameter(path, first, last, repickle=repickle)


def plot_comparison_report(ignore=False):
    pairs = [
        # new='bunny_many_goals/runs',
        # slugs='bunny_many_goals/runs_slugs',
        # numerator='slugs',
        #
        dict(
            new='synt15/runs_gr1x_fdbk_no_tight',
            memoize='synt15/runs',
            numerator='memoize',
            fname='comparison_memoize.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            one_manager='synt15/runs_gr1x_one_manager',
            numerator='one_manager',
            fname='comparison_one_manager.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            linear_conj='synt15/runs_gr1x_linear_conj',
            numerator='linear_conj',
            fname='comparison_linear_conj.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            no_defer='synt15/runs_gr1x_no_defer',
            numerator='no_defer',
            fname='comparison_no_defer.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            slugs='synt15/runs_slugs',
            numerator='slugs',
            fname='comparison_gr1x_slugs.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            no_defer_no_fdbk='synt15/runs_gr1x_no_defer_no_fdbk',
            numerator='no_defer_no_fdbk',
            fname='comparison_no_defer_no_fdbk.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            no_defer_no_fdbk_xz='synt15/runs_gr1x_no_defer_no_fdbk_xz',
            numerator='no_defer_no_fdbk_xz',
            fname='comparison_no_defer_no_fdbk_xz.pdf'),
        dict(
            no_defer_no_fdbk='synt15/runs_gr1x_no_defer_no_fdbk',
            no_defer_no_fdbk_xz='synt15/runs_gr1x_no_defer_no_fdbk_xz',
            numerator='no_defer_no_fdbk_xz',
            fname='comparison_no_defer_no_fdbk_vs_xz.pdf'),
        dict(
            new='synt15/runs_gr1x_logging_info',
            fdbk_no_tight='synt15/runs_gr1x_fdbk_no_tight',
            numerator='fdbk_no_tight',
            fname='comparison_fdbk_no_tight.pdf'),
        dict(
            tight='synt15/runs_gr1x_no_defer_no_fdbk_xz',
            no_tight='synt15/runs_gr1x_fdbk_no_tight',
            numerator='no_tight',
            fname='comparison_fdbk_no_vs_tight.pdf'),
        dict(
            slugs='synt15/runs_slugs',
            slugs_browne='synt15/runs_slugs_browne',
            numerator='slugs_browne',
            fname='comparison_slugs_browne.pdf'),
        dict(
            no_defer='jcss12/runs_gr1x_no_defer',
            slugs='jcss12/runs_slugs_1',
            numerator='slugs',
            fname='comparison_no_defer_slugs.pdf'),
        dict(
            binary='jcss12/runs_gr1x_defer_binary',
            slugs='jcss12/runs_slugs_1',
            numerator='slugs',
            fname='comparison_binary_slugs.pdf'),
        dict(
            fdbk_no_tight='jcss12/runs',
            slugs='jcss12/runs_slugs_1',
            numerator='slugs',
            fname='comparison_fdbk_no_tight_slugs.pdf'),
        dict(
            defer_binary='jcss12/runs_gr1x_defer_binary',
            fdbk_no_tight='jcss12/runs',
            numerator='fdbk_no_tight',
            fname='comparison_fdbk_no_tight.pdf'),
        dict(
            defer_binary='jcss12/runs_gr1x_defer_binary',
            no_defer='jcss12/runs_gr1x_no_defer',
            numerator='no_defer',
            fname='comparison_no_defer.pdf'),
        dict(
            fdbk_no_tight='genbuf/runs',
            slugs='genbuf/runs_slugs',
            numerator='fdbk_no_tight',
            fname='comparison_fdbk_no_tight_vs_slugs.pdf'),
        # dict(
        #     new='cinderella/runs',
        #     slugs='cinderella/runs_slugs',
        #     numerator='slugs',
        #     fname='comparison_gr1x_slugs.pdf')
    ]
    for p in pairs:
        plot_comparison(p, ignore)


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
    fig_fname = os.path.join(path, 'stats.pdf')
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


def plot_comparison(paths, ignore):
    """Plot time, ratios, BDD nodes over parameterized experiments.

      - time
      - winning set computation / total time
      - reordering / total time
      - total nodes
      - peak nodes
    """
    assert len(paths) >= 2, paths
    # paths
    data_paths = dict(paths)
    if 'numerator' in data_paths:
        data_paths.pop('numerator')
        data_paths.pop('fname')
    else:
        paths['numerator'] = next(iter(paths))
    # file up to date ?
    path = next(data_paths.itervalues())
    head, _ = os.path.split(path)
    fig_fname = os.path.join(head, paths['fname'])
    try:
        fig_time = os.path.getmtime(fig_fname)
    except OSError:
        fig_time = 0
    # load
    measurements = dict()
    fname = 'data.pickle'
    older = True
    for path in data_paths.itervalues():
        path = os.path.join(path, fname)
        t = os.path.getmtime(path)
        if t > fig_time:
            older = False
            break
    if older and not ignore:
        print('skip "{f}"\n'.format(f=fig_fname))
        return
    for k, path in data_paths.iteritems():
        pickle_fname = os.path.join(path, fname)
        print('open "{f}"'.format(f=pickle_fname))
        with open(pickle_fname, 'r') as f:
            measurements[k] = pickle.load(f)
    # plot
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    plt.clf()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # styles = ['b-', 'r--']
    for k, d in measurements.iteritems():
        if k == paths['numerator']:
            style = 'b-'
        else:
            style = 'r--'
        plot_single_experiment_vs_parameter(d, k, style)
    # plot_total_time_ratio(measurements, paths, data_paths)
    # save
    print('save "{f}"\n'.format(f=fig_fname))
    plt.savefig(fig_fname, bbox_inches='tight')


def plot_total_time_ratio(measurements, paths, data_paths):
    assert len(measurements) == 2, measurements
    ax = plt.subplot(5, 1, 5)
    ax.set_yscale('log')
    numerator = paths['numerator']
    denominator = set(data_paths)
    denominator.remove(numerator)
    (denominator,) = denominator
    param_num = measurements[numerator]['n_masters']
    param_den = measurements[denominator]['n_masters']
    time_num = measurements[numerator]['total_time']
    time_den = measurements[denominator]['total_time']
    common = set(param_num).intersection(param_den)
    param = [k for k in param_num if k in common]
    time_num = [v for k, v in zip(param_num, time_num)
                if k in common]
    time_den = [v for k, v in zip(param_den, time_den)
                if k in common]
    time_num = np.array(time_num)
    time_den = np.array(time_den)
    y = time_num / time_den
    plt.plot(param, y, 'b-')
    plt.plot([param[0], param[-1]], [1.0, 1.0], 'g--')
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
    ax = plt.subplot(2, 2, 1)
    plt.plot(n_masters, total_time, style, label=name)
    # annotate
    ax.set_yscale('log')
    ax.tick_params(labelsize=tsz)
    plt.grid(True)
    plt.xlabel('Parameter', fontsize=fsz)
    plt.ylabel('Total time\n(sec)', fontsize=fsz)
    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # reordering time
    ax = plt.subplot(2, 2, 3)
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
    plt.ylabel('Total reordering time\n(sec)', fontsize=fsz)
    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # peak BDD nodes
    ax = plt.subplot(2, 2, 2)
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
    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #
    # peak BDD nodes
    ax = plt.subplot(2, 2, 4)
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
    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)


def pickle_results(path, first, last, repickle):
    """Dump to pickle file all measurements from a directory."""
    fname = 'details_{i}.txt'.format(i='{i}')
    log_fname = os.path.join(path, fname)
    fname = 'data.pickle'
    pickle_fname = os.path.join(path, fname)
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
    # realizable ? (slugs only, gr1x always makes strategy)
    if 'make_transducer_end' in data:
        t1 = data['make_transducer_end']['time'][0]
    elif 'winning_set_end' in data:
        print('Warning: unrealizable found')
        t1 = data['winning_set_end']['time'][0]
    else:
        raise Exception('Winning set unfinished!')
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
    # realizable (slugs) ?
    if 'make_transducer_start' in data:
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
    p.add_argument('--ignore', action='store_true',
                   help='ignore older comparison PDF files')
    args = p.parse_args()
    if args.compare:
        plot_comparison_report(args.ignore)
    else:
        plot_report(args.repickle)
