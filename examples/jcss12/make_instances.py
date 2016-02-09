#!/usr/bin/env python
"""Dump instances for JCSS'12, in Promela and SlugsIn."""
import argparse
import pprint
import logging
from tugs import utils
from . import amba_generator


log = logging.getLogger(__name__)
PROMELA_PATH = 'pml/jcss12_{i}_masters.txt'
SLUGSIN_PATH = 'slugsin/jcss12_{i}_masters.txt'
N = 2
M = 17


def dump_promela(n, m):
    """Dump instances as Promela."""
    for i in xrange(n, m):
        code = amba_generator.generate_spec(i, use_ba=False)
        promela_file = PROMELA_PATH.format(i=i)
        with open(promela_file, 'w') as f:
            f.write(code)
        log.info('dumped Promela for {i} masters'.format(i=i))


def dump_slugsin(n, m):
    """Dump instances as SlugsIn."""
    for i in xrange(n, m):
        promela_file = PROMELA_PATH.format(i=i)
        with open(promela_file, 'r') as f:
            pml_code = f.read()
        slugsin_code = utils.translate_promela_to_slugsin(pml_code)
        slugsin_file = SLUGSIN_PATH.format(i=i)
        with open(slugsin_file, 'w') as f:
            f.write(slugsin_code)
        log.info('dumped SlugsIn for {i} masters'.format(i=i))


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
