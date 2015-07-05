# CFLAGS="-ggdb -DSIZEOF_LONG=8 -DCUDD_COMPILER_OPTIONS_SET -arch x86_64 -Wall -W -DUSE_CUDD -mtune=native -DHAVE_IEEE_754 -DBSD -DSIZEOF_VOID_P=8 -pthread -fPIC -fwrapv -O0 -Wall -fno-strict-aliasing -I /Users/ifilippi/.virtualenvs/newslugs/include/python2.7 -I /Users/ifilippi/github/slugs/lib/cudd-2.5.0/include/"
# LDFLAGS="-L../../slugs/lib/cudd-2.5.0/cudd -L../../slugs/lib/cudd-2.5.0/util -L../../slugs/lib/cudd-2.5.0/mtr -L../../slugs/lib/cudd-2.5.0/st -L../../slugs/lib/cudd-2.5.0/dddmp -L../../slugs/lib/cudd-2.5.0/epd -lcudd -ldddmp -lmtr -lepd -lst -lutil"

CFLAGS="-I /opt/local/include"
LDFLAGS="-L /opt/local/lib -lbdd"

cudd:
	CFLAGS=$(CFLAGS) \
	LDFLAGS=$(LDFLAGS) \
	python setup.py build_ext --inplace -g

clean:
	-rm *.so

