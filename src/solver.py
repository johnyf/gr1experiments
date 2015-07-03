#!/usr/bin/env python
import argparse
import logging
import simple


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str, help='slugsin input file')
    args = p.parse_args()
    fname = args.file
    # fname = 'slugs_small.txt'
    logger = logging.getLogger('simple')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')
    simple.solve_game(fname)
