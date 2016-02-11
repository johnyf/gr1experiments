#!/usr/bin/env python
"""Translate open Promela to SlugsIn."""
import argparse
from tugs import utils


def dump_slugsin():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=str,
                   help='input file')
    p.add_argument('target', type=str,
                   help='output file')
    args = p.parse_args()
    with open(args.source, 'r') as f:
        pml_code = f.read()
    slugsin_code = utils.translate_promela_to_slugsin(pml_code)
    with open(args.target, 'w') as f:
        f.write(slugsin_code)


if __name__ == '__main__':
    dump_slugsin()
