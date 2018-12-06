#!/usr/local/bin/python3
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random numbers.')
    parser.add_argument('--num', dest='num_data', default='100000',
                        help='number of data points of position (default: 100000)')
    parser.add_argument('--dist', dest='dist', default='uniform',
                        help='distribution of data points (default: uniform)')
    parser.add_argument('parameters', metavar='P', type=float, nargs='*',
                        help='parameters for the distribution')

    args = parser.parse_args()
    num_data = int(args.num_data)
    dist = args.dist
    pa = args.parameters

    random.seed()

    dt = []

    if dist == 'uniform':
        low_range = 0.0
        max_range = 0.0
        if len(pa) == 0:
            low_range = 0.0
            max_range = 300000.0
        elif len(pa) == 1:
            max_range = pa[0]
        elif len(pa) == 2:
            low_range = pa[0]
            max_range = pa[1]
        else:
            print("uniform distribution requires 0, 1, or 2 parameters")
            exit(1)
        length = max_range - low_range
        for i in range(num_data):
            dt.append(int(random.random() * length + low_range))
    elif dist == 'exp':
        lbd = 0.0
        if len(pa) == 0:
            lbd = 1.0/50000.0
        elif len(pa) == 1:
            lbd = 1.0/pa[0]
        else:
            print("exponential distribution requires 0 or 1 parameters (mean of dist)")
            exit(1)
        for i in range(num_data):
            dt.append(int(random.expovariate(lbd)))

    dt = sorted(dt)

    for i in range(num_data):
        print("{},{}".format(dt[i], i))

