#!/usr/bin/env python
"""Print digest of logs."""
import datetime
import os
import psutil
from tugs import utils


def scan_directory():
    path = './bunny/runs'
    logname = 'details_'
    incomplete_files = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.startswith(logname):
                file_path = os.path.join(root, f)
                file_path = os.path.abspath(file_path)
                data = utils.load_log_file(file_path)
                if 'dump_strategy_end' in data:
                    continue
                # still running or killed
                incomplete_files.append(file_path)
    open_files = find_file_pids(incomplete_files)
    for f in open_files:
        print_progress(f)


def find_file_pids(files):
    open_files = list()
    for p in psutil.process_iter():
        try:
            flist = p.open_files()
            pid = p.pid
            name = p.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not flist:
            continue
        for path, _ in flist:
            if path not in files:
                continue
            print(pid, name, path)
            open_files.append(path)
    assert all(f in files for f in open_files), (
        files, open_files)
    return open_files


def print_progress(f):
    data = utils.load_log_file(f)
    if 'winning_set_start' in data:
        print('\nstarted win set')
        t0 = data['winning_set_start']['time'][0]
        date = datetime.datetime.fromtimestamp(t0)
        s = date.strftime('%Y-%m-%d %H:%M:%S')
        print('win set end at: {s}'.format(s=s))
    if 'winning_set_end' in data:
        print('finished win set')
        t1 = data['winning_set_end']['time'][0]
        t_win = t1 - t0
        print('win set time: {t:1.2f} sec'.format(t=t_win))
        date = datetime.datetime.fromtimestamp(t1)
        s = date.strftime('%Y-%m-%d %H:%M:%S')
        print('win set end at: {s}'.format(s=s))
    if 'make_transducer_start' in data:
        print('started making transducer')
    if 'make_transducer_end' in data:
        print('finished making transducer')


if __name__ == '__main__':
    scan_directory()
