"""skip first number_lines rows and write file.csv till the end """
from argparse import ArgumentParser
import sys

from utils import report_error, InputError



DESCRIPTION = 'csvtail - Print header and last lines of input'
EXAMPLES = 'example: cat file.csv | csvtail -n -100'


def main(args):
    '''
    :param args: parser argumnets
    :return: none
    :procces: writing csv onto output file
    '''

    #by default
    input_stream = sys.stdin
    output_stream = sys.stdout

    #handling errors
    try:
        if args.file:
            input_stream = open(args.file, 'r')
        if args.output_file:
            output_stream = open(args.output_file, 'w')

        #reading first row from input_stream expect: column names
        columns_names = input_stream.readline()
        output_stream.write(columns_names)

        for i, line in enumerate(input_stream):
            if  i >= args.number_of_lines:
                output_stream.write(line)


    except FileNotFoundError:
        report_error("File {} doesn't exist ".format(args.file))
    except InputError as e:
        report_error(e.message + '. Row: ' + str(e.expression))
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        # The following line prevents python to inform you about the broken pipe
        sys.stderr.close()
    except Exception as e:
        report_error('Caught unknown exception. Please report to developers: {}'.format(e))
    finally:
        if input_stream and input_stream != sys.stdin:
            input_stream.close()
        if output_stream:
            output_stream.close()


def parse_args():
    '''
    :process: get user argumnets
    :return: args
    '''
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)

    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')
    parser.add_argument('-o', '--output_file', type=str, help='Output file. stdout is used by default')
    #parser.add_argument('-h', '--help', help='show help message')

    parser.add_argument('-q', '--quiet', help="Don't print information regarding errors", action='store_true')
    parser.add_argument('--careful', help='Stop if input contains an incorrect row', action='store_true')

    parser.add_argument('-s', '--separator', type=str, help='Separator to be used', default=',')
    parser.add_argument('-f', '--format_floats', help='Format floating-point numbers nicely', action='store_true')



    parser.add_argument('-n', '--number_of_lines', type=int,
                        help='Number of last rows to print if positive `ROWS_COUNT`.\
                             Else skips `ROWS_COUNT` lines and prints till the end of input.', default=0)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
