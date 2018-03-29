import sys
import argparse


parser = argparse.ArgumentParser(description='Give me oracle file.')
parser.add_argument('-p', type=str, help='oracle file')
args = parser.parse_args()
count = 0
for test_line in open(args.p):
    if test_line.startswith("#"):
        count = 1
    else:
        if count > 0:
            count=count + 1
        if count == 3:
            print test_line[:-1]
