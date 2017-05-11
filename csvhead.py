#!/usr/bin/env python
"""
This scripts allows user to see first n lines in *.csv file.
"""

from argparse import ArgumentParser
import sys

DESCRIPTION = 'csvhead - Print header and first lines of input.'
EXAMPLES = 'example: cat file.csv | csvhead -n 100'

def main():
    """
    Reads line per line and writes it in output_stream.
    """
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'r') if args.output_file else sys.stdout
    for _ in range(0, args.number_of_lines+1):
        output_stream.write(input_stream.readline())

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
