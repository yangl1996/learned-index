import argparse
import random

parser = argparse.ArgumentParser(description='Generate random numbers.')
parser.add_argument('--num', dest='num_data', default='100000',
                    help='number of data points of position (default: 100000)')
parser.add_argument('--range', dest='data_range', default='300000',
                    help='range of data points (default: 300000)')

args = parser.parse_args()
num_data = int(args.num_data)
data_range = int(args.data_range)

random.seed()

dt = []

for i in range(num_data):
    dt.append(random.randrange(data_range))

dt = sorted(dt)

for i in range(num_data):
    print("{},{}".format(dt[i], i))

