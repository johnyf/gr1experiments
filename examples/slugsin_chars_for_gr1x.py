import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=str,
                   help='input file')
    p.add_argument('target', type=str,
                   help='output file')
    args = p.parse_args()
    with open(args.source, 'r') as f:
        s = f.read()
    snew = s.replace('.', 'dot')
    snew = snew.replace('@', 'at')
    with open(args.target, 'w') as f:
        f.write(snew)


if __name__ == '__main__':
    main()
