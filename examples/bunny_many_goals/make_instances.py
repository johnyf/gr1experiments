#!/usr/bin/env python
"""Dump instances for bunny, in Promela and SlugsIn."""
import argparse
import itertools
import pprint
import logging
import re
from tugs import utils


log = logging.getLogger(__name__)
INPUT_FILE = 'bunny.pml'
PROMELA_PATH = 'pml/bunny_many_goals_{i}.txt'
SLUGSIN_PATH = 'slugsin/bunny_many_goals_{i}.txt'


def dump_promela(n, m):
    """Dump instances of Promela."""
    for i in xrange(n, m):
        code = make_promela(i)
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


def make_promela(n):
    """Return Promela code for instance with size `n`."""
    fname = INPUT_FILE
    with open(fname, 'r') as f:
        s = f.read()
    # set number of cells
    newline = '#define H {n}'.format(n=n)
    code = re.sub('#define H.*', newline, s)
    newline = '#define W {m}'.format(m=n-1)
    code = re.sub('#define W.*', newline, code)
    # add multiple weak fairness assumptions
    code += form_progress(n)
    return code


def form_progress(n):
    """Return conjunction of LTL formulae for progress."""
    g0 = ('[]<>((x == {k}) && (y == 0))'.format(k=k)
          for k in xrange(n))
    g1 = ('[]<>((x == 0) && (y == {k}))'.format(k=k)
          for k in xrange(n))
    c = itertools.chain(g0, g1)
    prog = ' && '.join(c)
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
    p.add_argument('--min', type=int,
                   help='from this # of masters')
    p.add_argument('--max', type=int,
                   help='to this # of masters')
    args = p.parse_args()
    n = args.min
    m = args.max + 1
    dump_promela(n, m)
    dump_slugsin(n, m)


if __name__ == '__main__':
    main()
