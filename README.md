# Langmuir probe data parser
Langmuir Probe data parser en plotter

## Dependencies 

Plase install the following python dependencies with PIP or your sistem packet managet
- python 2.7.x
- argparse
- ephem
- pandas
- numpy
- cartopy
- matplotlib

## Run

### How to use

    usage: langmuir_parser.py [-h] [-a, --animate] [-s, --save] files [files ...]

    positional arguments:
    files          CSV Files to process

    optional arguments:
    -h, --help     show this help message and exit
    -a, --animate  Show an animated plot
    -s, --save     Save figure

### Examples

- Simply plot one datafile

        python2 langmuir_parser.py LP_20180719_070200UTC.csv
    
- Plot one datafile and save
    
        python2 langmuir_parser.py LP_20180719_070200UTC.csv -s
    
- Show an animated plot

        python2 langmuir_parser.py LP_20180719_070200UTC.csv -a
    
- Show and animated plot and save a video

        python2 langmuir_parser.py LP_20180719_070200UTC.csv -a -s
    
## Contact

Any doubt just contact me: carlgonz@uchile.cl

    
