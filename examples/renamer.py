#!/usr/bin/env python
"""Rename numbered files."""
import os
import shutil


def main():
    path = './jcss12/pml/'
    for f in os.listdir(path):
        print(f)
        newf = f.replace('_masters', '')
        print(newf)
        a = path + f
        b = path + newf
        shutil.move(a, b)


if __name__ == '__main__':
    main()
