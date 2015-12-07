"""Parsing of signal logs from experiments, and version logging."""
import importlib
import json
import logging
import os
import subprocess
import time
import git
import numpy as np


logger = logging.getLogger(__name__)
CONFIG_FILE = 'config.json'


def git_version(path):
    """Return SHA-dirty for repo under `path`."""
    repo = git.Repo(path)
    sha = repo.head.commit.hexsha
    dirty = repo.is_dirty()
    return sha + ('-dirty' if dirty else '')


def snapshot_versions():
    """Log versions of software used."""
    # existing log ?
    try:
        with open(CONFIG_FILE, 'r') as f:
            d_old = json.load(f)
    except IOError:
        d_old = None
    # get SHA
    paths = [
        '~/github/omega']
    d = dict()
    for path in paths:
        sha = git_version(path)
        d[path] = sha
    # slugs binary version
    cmd = ['slugs', '--version']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        print(p.returncode)
        raise Exception('`slugs` not found on path')
    slugs_version = p.stdout.read().strip()
    d['slugs'] = slugs_version
    # versions of python packages
    packages = [
        'dd', 'omega', 'tugs',
        'openpromela', 'promela']
    for s in packages:
        pkg = importlib.import_module(s)
        d[s] = pkg.__version__
    t_now = time.strftime('%Y-%b-%d-%A-%T-%Z')
    d['time'] = t_now
    d['platform'] = os.uname()
    # check versions
    compare = list(packages)
    compare.append('slugs')
    if d_old is not None:
        for k in compare:
            assert (
                d[k] == d_old[k],
                'versions differ from {cfg}'.format(
                    cfg=CONFIG_FILE))
    # dump
    with open(CONFIG_FILE, 'w') as f:
        json.dump(d, f, indent=4)
    return d


def add_logfile(fname, logger_name):
    h = logging.FileHandler(fname, mode='w')
    log = logging.getLogger(logger_name)
    log.addHandler(h)
    return h


def close_logfile(h, logger_name):
    log = logging.getLogger(logger_name)
    log.removeHandler(h)
    h.close()


def load_log_file(fname):
    data = dict()
    with open(fname, 'r') as f:
        for line in f:
            if "'time'" not in line:
                continue
            # repr
            d = eval(line)
            split_data(d, data)
    for k, v in data.iteritems():
        for q, r in v.iteritems():
            data[k][q] = np.array(r, dtype=float)
    return data


def split_data(d, data):
    """Store sample in `d` as a signal in `data`.

    @type d: `dict`
    @type data: `dict(dict(time=list(), value=list()))`
    """
    t = d['time']
    for k, v in d.iteritems():
        if k == 'time':
            continue
        # is a signal
        # new ?
        if k not in data:
            data[k] = dict(time=list(), value=list())
        data[k]['time'].append(t)
        data[k]['value'].append(v)


def get_signal(name, data):
    return data[name]['time'], data[name]['value']


def inspect_data(data):
    for k in data:
        t = data[k]['time']
        v = data[k]['value']
        print(k, len(t), len(v))
