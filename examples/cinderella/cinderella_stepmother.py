#!/usr/bin/env python
"""Dump discretized cinderella-stepmother games."""
TEMPLATE = """\
/* {comment} */

{inLines}
{outLines}
sys int(0, {nofBucketsMinusOne}) e;

assume ltl {{
{envInit}
&& []({stepMotherRestrictionLine})
&& []<>(true)
}}

assert ltl {{
{sysInit}
&& []({sysTransA})
&& []<>(true)
}}

"""
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


def make_code(params, pml_fname):
    (nofBuckets, cinderellaPower,
     bucketCapacity, stepmotherPower) = params
    comment = "n{a} c{b}, {c} by {d}".format(
        a=nofBuckets,
        b=cinderellaPower,
        c=bucketCapacity,
        d=stepmotherPower)
    inlines = [
        'env int(0, {b}) x{i};'.format(i=i, b=stepmotherPower)
        for i in xrange(0, nofBuckets)]
    outlines = [
        'sys int(0, {b}) y{i};'.format(i=i, b=bucketCapacity)
        for i in xrange(0, nofBuckets)]
    envinit = [
        'x{i} = 0'.format(i=i)
        for i in xrange(0, nofBuckets)]
    sysinit = [
        'y{i} = 0'.format(i=i)
        for i in xrange(0, nofBuckets)]
    sys_action = [
        "!((e<=" + str(i) + " && e+" +
        str(cinderellaPower) + " > " + str(i) +
        ") | (e+" + str(cinderellaPower - 1) +
        " >= " + str(i+nofBuckets) + ")) -> y" + str(i) +
        "' = y" + str(i) + " + x" + str(i) +
        "'"
        for i in xrange(0, nofBuckets)]
    env_action = [
        "((e<=" + str(i) + " && e+" +
        str(cinderellaPower) + " > " +
        str(i) + ") | (e+" + str(cinderellaPower - 1) +
        " >= " + str(i + nofBuckets) +
        ")) -> y" + str(i) + "' = x" + str(i) +
        "'"
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


def conj(x):
    return '\n && '.join('(' + u + ')' for u in x)


if __name__ == '__main__':
    for i, p in enumerate(configurations):
        pml_fname = 'pml/cinderella_{i}.txt'.format(i=i)
        make_code(p, pml_fname)
