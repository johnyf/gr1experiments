#!/usr/bin/env python
"""Dump instances for SYTN'15, in Promela and SlugsIn."""
import argparse
import pprint
import logging
import re
from tugs import utils


log = logging.getLogger(__name__)
INPUT_FILE = 'amba_conj.pml'
PROMELA_PATH = 'pml/synt15_{i}.txt'
SLUGSIN_PATH = 'slugsin/synt15_{i}.txt'
N = 2
M = 17


def dump_promela(n, m):
    """Dump instances of Promela for SYNT'15."""
    for i in xrange(n, m):
        code = make_synt15_promela(i)
        promela_file = PROMELA_PATH.format(i=i)
        with open(promela_file, 'w') as f:
            f.write(code)
        log.info('dumped Promela for {i} masters'.format(i=i))


def dump_slugsin(n, m):
    for i in xrange(n, m):
        promela_file = PROMELA_PATH.format(i=i)
        with open(promela_file, 'r') as f:
            pml_code = f.read()
        slugsin_code = utils.translate_promela_to_slugsin(pml_code)
        slugsin_file = SLUGSIN_PATH.format(i=i)
        with open(slugsin_file, 'w') as f:
            f.write(slugsin_code)
        log.info('dumped SlugsIn for {i} masters'.format(i=i))


def make_synt15_promela(n):
    """Return Promela code for instance with `n` masters."""
    fname = INPUT_FILE
    with open(fname, 'r') as f:
        s = f.read()
    # set number of masters
    j = n - 1
    newline = '#define N {j}'.format(j=j)
    code = re.sub('#define N.*', newline, s)
    # add multiple weak fairness assumptions
    code += form_progress(n)
    return code


def form_progress(n):
    """Return conjunction of LTL formulae for progress."""
    prog = ' && '.join(
        '[]<>(request[{k}] -> (master == {k}))'.format(k=k)
        for k in xrange(n))
    return 'assert ltl { ' + prog + ' }'


def main():
    # log
    fh = logging.FileHandler('code_generator_log.txt', mode='w')
    log.addHandler(fh)
    log.setLevel(logging.DEBUG)
    # tugs log
    log1 = logging.getLogger('tugs.utils')
    log1.addHandler(fh)
    log1.setLevel(logging.DEBUG)
    # record env
    versions = utils.snapshot_versions()
    log.info(pprint.pformat(versions))
    # args
    p = argparse.ArgumentParser()
    p.add_argument('--min', default=N, type=int,
                   help='from this # of masters')
    p.add_argument('--max', default=M, type=int,
                   help='to this # of masters')
    # p.add_argument('--debug', type=int, default=logging.ERROR,
    #                help='python logging level')
    args = p.parse_args()
    n = args.min
    m = args.max + 1
    dump_promela(n, m)
    dump_slugsin(n, m)


if __name__ == '__main__':
    main()
