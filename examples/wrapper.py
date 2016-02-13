#!/usr/bin/env python
"""Wrapper to circumvent the entry point.

Because, if development versions of dependencies are installed,
but `install_requires` contains no local identifiers,
then the entry point raises a `VersionConflict` for its context.
"""
import sys
from tugs import solver


if __name__ == '__main__':
    solver.command_line_wrapper(args=sys.argv[1:])
