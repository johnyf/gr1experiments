#!/usr/bin/env python
"""Collect all plots for a benchmark into one directory."""
import os
import shutil


def main():
    path = './synt15/'
    fig_fname = 'stats.pdf'
    destination = 'all_plots'
    for root, dirs, files in os.walk(path):
        for f in files:
            if f != fig_fname:
                continue
            _, tail = os.path.split(root)
            fname = tail + '.pdf'
            a = os.path.join(root, f)
            b = os.path.join(path, destination, fname)
            print(a, b)
            shutil.copy(a, b)


if __name__ == '__main__':
    main()
