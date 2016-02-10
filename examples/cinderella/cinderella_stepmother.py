#!/usr/bin/env python
"""Creates some benchmarks for the discretized
cinderella-stepmother game.
"""
import subprocess


TEMPLATE = """\
# {comment}

[INPUT]
%(inLines)s

[OUTPUT]
%(outLines)s
e:0...%(nofBucketsMinusOne)d

[ENV_TRANS]
%(stepMotherRestrictionLine)s

[ENV_INIT]
%(envInit)s

[SYS_INIT]
%(sysInit)s

[SYS_TRANS]
%(sysTransA)s
%(sysTransB)s

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


def make_code(params, slugsin_file):
    (nofBuckets, cinderellaPower,
     bucketCapacity, stepmotherPower) = params
    comment = "n{a} c{b}, {c} by {d}".format(
        a=nofBuckets,
        b=cinderellaPower,
        c=bucketCapacity,
        d=stepmotherPower)
    inlines = [
        "x" + str(i) + ":0..." + str(stepmotherPower)
        for i in xrange(0, nofBuckets)]
    outlines = [
        "y" + str(i) + ":0..." + str(bucketCapacity)
        for i in xrange(0, nofBuckets)]
    envinit = [
        "x" + str(i) + " = 0"
        for i in xrange(0, nofBuckets)]
    sysinit = [
        "y" + str(i) + " = 0"
        for i in xrange(0, nofBuckets)]
    sys_action = [
        "!((e<=" + str(i) + " & e+" +
        str(cinderellaPower) + " > " + str(i) +
        ") | (e+" + str(cinderellaPower - 1) +
        " >= " + str(i+nofBuckets) + ")) -> y" + str(i) +
        "' = y" + str(i) + " + x" + str(i) +
        "'"
        for i in xrange(0, nofBuckets)]
    env_action = [
        "((e<=" + str(i) + " & e+" +
        str(cinderellaPower) + " > " +
        str(i) + ") | (e+" + str(cinderellaPower - 1) +
        " >= " + str(i + nofBuckets) +
        ")) -> y" + str(i) + "' = x" + str(i) +
        "'"
        for i in xrange(0, nofBuckets)]
    restrict = [
        "x" + str(i) + "'"
        for i in xrange(0, nofBuckets)]
    text = TEMPLATE.format(
        comment=comment,
        inLines="\n".join(inlines),
        outLines="\n".join(outlines),
        nofBucketsMinusOne=nofBuckets - 1,
        stepMotherRestrictionLine=(
            "+".join(restrict) + " <= " + str(stepmotherPower)),
        envInit="\n".join(envinit),
        sysInit="\n".join(sysinit),
        sysTransA="\n".join(sys_action),
        sysTransB="\n".join(env_action))
    with open(fname, 'w') as f:
        f.write(text)
    # to slugsin
    s = subprocess.check_output([translator, fname])
    with open(slugsin_file, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    for i, p in enumerate(configurations):
        slugsin_file = 'slugsin/cinderella_{i}.txt'.format(i=i)
        make_code(p, slugsin_file)
