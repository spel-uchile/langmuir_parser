import sys, os
import argparse


def get_parameters():
    """
    Parse command line parameters
    :return: namespace with parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=str, nargs='+', help="Path to folders where the files are")
    parser.add_argument('-c, --concatenate', dest="concatenate", action="store_true",
                        help="Concatenate all files in one")
    parser.add_argument('-a, --animate', dest="animate", action="store_true", help="Show an animated plot")
    parser.add_argument('-s, --save', dest="save", action="store_true", help="Save figure")
    parser.add_argument('-l', '--list', dest="color_min_max", nargs=6,
                        help='Colorbar range min and max for each graphic')
    return parser.parse_args()

def filter_files(file_list):
    """
        Filter folders and files of a file_list that are not in csv format
        :param file_list: List of file names
        :return: Filtered list
        """
    final_file_list = file_list[:]
    for file in file_list:
      file_extension = file.split('.')
      print(file)
      print(len(file_extension))
      if (len(file_extension) == 1) or (file_extension[len(file_extension) - 1] != "csv"):
          final_file_list.remove(file)
    return final_file_list

def get_file_list(directory):
    """
        Obtains a list with all the file names inside a directory
        :param directory: Path (string) to the folder that contains the files
        :return: List with the file names
        """
    return os.listdir(directory)

def files_to_string(dir_name):
    """
        Converts a list of files (that are inside the directory dir_name)
        into a string
        :param dir_name: Path (string) to the folder that contains the files
        :return: String of files paths concatenated by a single space between
        each other
        """
    files_str = ""
    file_names = filter_files(get_file_list(dir_name))
    for name in file_names:
        files_str += dir_name+"/"+name+" "
    return files_str


"""
MAIN FUNCTION
"""
if __name__ == "__main__":
    args = get_parameters()
    #Final_call is the call to the langmuir_parser script with their respective arguments
    final_call = "python2 langmuir_parser.py "
    for folder_name in args.folders:
        final_call += files_to_string(folder_name)
    if args.concatenate:
        final_call += " -c"
    if args.animate:
        final_call += " -a"
    if args.save:
        final_call += " -s"
    if args.color_min_max:
        final_call += " -l"
        for val in args.color_min_max:
            final_call += " " + str(val)
    os.system(final_call)
