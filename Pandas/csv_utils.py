import sys
import re

def read_csv(file_path, lines_number = 6233, separator = ','):
    """
        param: file_path path of file
        param: lines_number number of reading lines
        param: separator is a character
        return: readed rows
        
    """

    assert file_path
    input_stream = open(file_path, 'r')

    rows = []
    for i in range(lines_number):
         str = input_stream.readline().strip()
         if not str:
             break
         str = re.sub('(?!(([^"]*"){2})*[^"]*$)B', 'b', str)
         rows.append(str.split(separator))

    return rows



