#!/usr/bin/env python
# pylint: disable=invalid-name
"""
This scripts allows user to plot *.csv file.
"""

from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt

DESCRIPTION = 'csvplot - Plot the data based on *.csv file contents.'
EXAMPLES = 'example: cat yerevan_april_9.csv | csvplot -x area -y price,num_rooms \
--xlabel area --ylabels price,num_rooms'

def main():
    """
    Reads line per line and writes it in output_stream.
    """
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'w') if args.output_file else sys.stdout
    headers = input_stream.readline().strip().split(args.separator)
    if args.xlabel:
        X_label = args.xlabel.strip()
    else:
        X_label = args.x.strip()
    if args.ylabels:
        Y_labels = args.ylabels.strip().split(args.separator)
    else:
        Y_labels = args.y.srtip().split(args.separator)
    Y_axes = args.y.strip().split(args.separator)
    num = len(Y_axes)
    lines = input_stream.readlines()
    if args.x:
        X_axes = args.x.strip().split(args.separator)
    else:
        X_axes = range(len(lines))

    x_val = [line.strip().split(',')[headers.index(X_axes[0])]
             for line in lines]
    for i in range(num):
        y_val = [line.strip().split(',')[headers.index(Y_labels[i])]
                 for line in lines]
        plt.xlabel(X_label)
        plt.plot(x_val, y_val, label=Y_axes[i])
        plt.legend()
    plt.show()

    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()


def parse_args():
    """
    Adds arguments to parser.
    """
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)
    parser.add_argument('-s', '--separator', type=str,
                        help='Separator to be used', default=',')
    parser.add_argument('-x', type=str, help='Specify the key to iterate over x-axes. \
                        If not provided, uses row number instead.')
    parser.add_argument('--xlabel', type=str, help='Label to be used for X axis.')
    parser.add_argument('-y', type=str, help='Specify columns to be plotted.')
    parser.add_argument('--ylabels', type=str, help='Labeles to be used for Y axis.')
    parser.add_argument('-o', '--output_file', type=str,
                        help='Output file. stdout is used by default')
    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')

    args = parser.parse_args()

    return args

main()
