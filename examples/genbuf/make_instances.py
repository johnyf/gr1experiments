#!/usr/bin/env python
"""Dump instances for GenBuf, in SlugsIn."""
import argparse
import logging
import subprocess as sub


log = logging.getLogger(__name__)
STRUCTURED_PATH = 'structured_slugs.txt'
SLUGSIN_PATH = 'slugsin/genbuf_{i}.txt'
GENERATOR = 'genbuf_spec_generator.pl'
TRANSLATOR = (
    '../../../slugs/tools/'
    'StructuredSlugsParser/compiler.py')
N = 2
M = 60


def dump_slugsin(n, m):
    """Dump instances as SlugsIn."""
    for i in xrange(n, m):
        print(i)
        struct_file = STRUCTURED_PATH
        sub.call(['perl', GENERATOR, str(i), struct_file])
        slugsin_file = SLUGSIN_PATH.format(i=i)
        p = sub.Popen(
            [TRANSLATOR, struct_file],
            stdout=sub.PIPE, stderr=sub.PIPE)
        out, err = p.communicate()
        with open(slugsin_file, 'w') as f:
            f.write(out)


def main():
    # args
    p = argparse.ArgumentParser()
    p.add_argument('--min', default=N, type=int,
                   help='from this # of masters')
    p.add_argument('--max', default=M, type=int,
                   help='to this # of masters')
    args = p.parse_args()
    n = args.min
    m = args.max + 1
    dump_slugsin(n, m)


if __name__ == '__main__':
    main()
