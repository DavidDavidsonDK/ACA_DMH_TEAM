import sys
import re

def print_row(row, column_widths):
    """
    Prints a row in human-readable format taking column widths into account
    :param row: row represented as a list of columns
    :param column_widths: a list of column list widths to be used for pretty printing
    :param output_stream: a stream to pretty print the row
    """
    output_line = '|'
    for i, column in enumerate(row):
        output_line += ' ' + column + ' ' * (column_widths[i] - len(column) + 1) + '|'
    print(output_line)



def log_csv(dataFrame, columns):
    """
        printing csv in human-readable format 
    """
    column_widths = [max([len(column) for column in [row[i] for row in dataFrame]]) for i in range(len(columns))]
    for row in dataFrame:
        print_row(row, column_widths)


def read_csv(file_path, lines_number = None, separator = ','):
    """
        param: file_path name of file from which read_csv has to read content
        param: lines_number indicate how much lines read_csv has to read
        param: separator is a character by default it's a comma(csv)
        process: reading from file line by line , handle it and store in a list
        
    """


    assert file_path
    input_stream = open(file_path, 'r')

    str = input_stream.readline().strip()
    str = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '@', str)

    if  not bool(lines_number):
        lines_number =  2147483647
    
    columns = str.split(separator)
    first_rows = [columns]

    
    for i in range(lines_number):
         str = input_stream.readline().strip()
         if not str:
             break
         str = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '@', str)

         first_rows.append(str.split(separator))

    return (first_rows, columns)




