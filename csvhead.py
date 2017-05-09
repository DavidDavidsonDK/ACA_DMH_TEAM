#!/usr/bin/env python
"""
This scripts allows user to see first n lines in *.csv file.
"""

from argparse import ArgumentParser
import sys

DESCRIPTION = 'csvpp - Print header and first lines of input.'
EXAMPLES = 'example: cat file.csv | csvhead -n 100'

def print_row(row, column_widths, output_stream):
    """
    Prints a row in human-readable format taking column widths into account
    :param row: row represented as a list of columns
    :param column_widths: a list of column list widths to be used for pretty printing
    :param output_stream: a stream to pretty print the row
    """
    output_line = '|'
    for i, column in enumerate(row):
        output_line += ' ' + column + ' ' * (column_widths[i] - len(column) + 1) + '|'
    output_line += '\n'
    output_stream.write(output_line)


def main():
    """
    Reads line per line and writes it in output_stream.
    """
    args = parse_args()
    print(args)
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'r') if args.output_file else sys.stdout

    lines = input_stream.readlines()

    for line in lines[0:args.number_of_lines+1]:
        output_stream.write(line)

    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()


def parse_args():
    """
    Adds arguments to parser.
    """
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)
    parser.add_argument('-n', '--number_of_lines', type=int,
                        help='Number of first rows to print', default=5)
    parser.add_argument('-o', '--output_file', type=str,
                        help='Output file. stdout is used by default')
    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')

    args = parser.parse_args()

    return args

main()
