#!/usr/bin/env python
"""Rename numbered files."""
import os
import shutil


def main():
    path = './bunny_goals/slugsin/'
    for f in os.listdir(path):
        print(f)
        newf = f.replace('bunny_', 'bunny_goals_')
        print(newf)
        a = os.path.join(path, f)
        b = os.path.join(path, newf)
        shutil.move(a, b)


if __name__ == '__main__':
    main()
