#!/usr/bin/env python
"""Dump discretized cinderella-stepmother games."""
import logging
from tugs import utils


TEMPLATE = """\
/* {comment} */

{inLines}
{outLines}
free sys int(0, {nofBucketsMinusOne}) e;

assume ltl {{
{envInit}
&& []({stepMotherRestrictionLine})
&& []<>(true)
}}

assert ltl {{
{sysInit}
&& []({sysTransA})
&& []({sysTransB})
&& []<>(true)
}}

"""
PROMELA_PATH = 'pml/cinderella_{i}.txt'
SLUGSIN_PATH = 'slugsin/cinderella_{i}.txt'
log = logging.getLogger(__name__)
# ==================================
# Elements in each configuration mean:
# 1. Number of Buckets
# 2. Number of adjacent buckets that may be
#    emptied by cinderella in each step
# 3. Bucket capacity
# 4. Number of elements that the stepmother
#    might add in total in each step
# ==================================
configurations = [
    (5, 2, 6, 4),
    (5, 2, 7, 4),
    (5, 3, 12, 9),
    (5, 3, 13, 9),
    (6, 1, 14, 5),
    (6, 1, 15, 5),
    (6, 1, 29, 10),
    (6, 2, 14, 6),
    (6, 2, 13, 6),
    (6, 3, 11, 6),
    (6, 3, 10, 6),
    (7, 1, 33, 10)
]
fname = 'cinderella.txt'
translator = ('../../../slugs/tools/'
              'StructuredSlugsParser/compiler.py')


def dump_promela(params, pml_fname):
    (nofBuckets, cinderellaPower,
     bucketCapacity, stepmotherPower) = params
    comment = "n{a} c{b}, {c} by {d}".format(
        a=nofBuckets,
        b=cinderellaPower,
        c=bucketCapacity,
        d=stepmotherPower)
    inlines = [
        'free env int(0, {b}) x{i};'.format(i=i, b=stepmotherPower)
        for i in xrange(0, nofBuckets)]
    outlines = [
        'free sys int(0, {b}) y{i};'.format(i=i, b=bucketCapacity)
        for i in xrange(0, nofBuckets)]
    envinit = [
        'x{i} == 0'.format(i=i)
        for i in xrange(0, nofBuckets)]
    sysinit = [
        'y{i} == 0'.format(i=i)
        for i in xrange(0, nofBuckets)]
    sys_action = [
        ("!((e <= {i} && e + {cinderellaPower} > {i}) "
         "|| (e + {c1} >= {c2})) -> (y{i}' == y{i} + x{i}')").format(
             i=i,
             cinderellaPower=cinderellaPower,
             c1=cinderellaPower - 1,
             c2=i + nofBuckets)
        for i in xrange(0, nofBuckets)]
    env_action = [
        ("((e <= {i} && e + {cinderellaPower} > {i}) "
         "|| (e + {c1} >= {c2})) -> (y{i}' == x{i}')").format(
             i=i,
             cinderellaPower=cinderellaPower,
             c1=cinderellaPower - 1,
             c2=i + nofBuckets)
        for i in xrange(0, nofBuckets)]
    restrict = [
        "x{i}'".format(i=i)
        for i in xrange(0, nofBuckets)]
    text = TEMPLATE.format(
        comment=comment,
        inLines='\n'.join(inlines),
        outLines='\n'.join(outlines),
        nofBucketsMinusOne=nofBuckets - 1,
        stepMotherRestrictionLine=(
            "+".join(restrict) + " <= " + str(stepmotherPower)),
        envInit=conj(envinit),
        sysInit=conj(sysinit),
        sysTransA=conj(sys_action),
        sysTransB=conj(env_action))
    with open(pml_fname, 'w') as f:
        f.write(text)


def dump_slugsin(i):
    promela_file = PROMELA_PATH.format(i=i)
    with open(promela_file, 'r') as f:
        pml_code = f.read()
    slugsin_code = utils.translate_promela_to_slugsin(pml_code)
    slugsin_file = SLUGSIN_PATH.format(i=i)
    with open(slugsin_file, 'w') as f:
        f.write(slugsin_code)
    log.info('dumped SlugsIn for {i} masters'.format(i=i))


def conj(x):
    return '\n && '.join('(' + u + ')' for u in x)


if __name__ == '__main__':
    for i, p in enumerate(configurations):
        pml_fname = PROMELA_PATH.format(i=i)
        dump_promela(p, pml_fname)
        dump_slugsin(i)
