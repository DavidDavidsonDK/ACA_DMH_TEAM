# csvtools

### csvtail

usage: `csvtail [-h] [-q] [--careful] [-o OUTPUT_FILE] [-n ROWS_COUNT] [file]`

Print header and last lines of input.

positional arguments:
  `file`
  > File to read input from. stdin is used by default

optional arguments:
  `-h, --help`
  > show help message and exit

  `-q, --quiet`
  > Don't print information regarding errors

  `--careful`
  > Stop if input contains an incorrect row

  `-n ROWS_COUNT, --number_of_lines ROWS_COUNT`
  > Number of last rows to print if positive `ROWS_COUNT`. Else skips `ROWS_COUNT` lines and prints till the end of
input.

  `-o OUTPUT_FILE, --output_file OUTPUT_FILE`
  > Output file. stdout is used by default

examples:
  `cat file.csv | csvtail -n -100`  skip first 100 rows and print file.csv till the end.



