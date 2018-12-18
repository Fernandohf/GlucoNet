import gzip
import os
import shutil
import sys
import pandas as pd


def unzip_all(dir_name=""):
    """
    Unzip all file in given directory.

    Args:
        dir_name: Relative directory name.
    """
    # Current and data directory
    local = os.getcwd()
    data_dir = os.path.join(local, dir_name)

    # For each file in directory
    for file in os.listdir(data_dir):
        if file.endswith(".gz"):
            # Read file content
            file_dir = os.path.join(data_dir, file)
            with gzip.open(file_dir, 'rb') as f_in:
                # Saves as extracted
                with open(file_dir[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def combine_files(file_names, dir_name="", file_type='.json'):
    """
    Combine all file starting with specific name in one file.

    Args:
        file_names: List of file names to combine.
        dir_name: Relative search directory name.
        file_type: Only search for this types of file.
    """
    # Current and data directory
    local = os.getcwd()
    data_dir = os.path.join(local, dir_name)

    # For each name in file_nanmes
    for name in file_names:
        data = pd.DataFrame()
        # For each file in directory
        for file in os.listdir(data_dir):
            if file.startswith(name):
                # Read file content
                file_dir = os.path.join(data_dir, file)
                if file_type == '.json':
                    new_data = pd.read_json(file_dir)
                elif file_type == '.csv':
                    new_data = pd.read_csv(file_dir)
                else:
                    raise Exception("Data fle type not supported.")
                # Join data
                data = pd.concat([data, new_data], sort=True)

        # Save file
        data.to_csv(name + '_combined' + file_type)

if __name__ == '__main__':
    # Unzip
    try:
        d = sys.argv[1]
    except IndexError:
        print("No data folder passed. Searching in local directory.")
        d = ""
    unzip_all(d)

    # Combine
    try:
        n = sys.argv[2]
    except IndexError:
        print("Searching default names: 'entries' and 'treatments'.")
        n = ['entries', 'treatments']

    combine_files(n, d)
