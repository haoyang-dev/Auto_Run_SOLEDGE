import json
import subprocess
import importlib.util
import time
import datetime
from matplotlib.path import Path
from scipy.interpolate import griddata
import os
import re

import numpy as np


def define_controlled_code_folder_path(path='defaut_path'):
    """
    path: str
    If no path is provided in the brackets,
    the Python script will use the upper-level path by default.
    To specify a custom path, please enter it within the brackets.
    """
    if path == 'defaut_path':
        used_directory = os.getcwd()
    else:
        used_directory = path

    return used_directory


def define_script_folder_path(path='defaut_path'):
    """
    path: str
    If there is nothing in the brackets,
    the path where the py script is located will be used by default.
    If you want to specify it, please enter the path in the brackets.
    """
    if path == 'defaut_path':
        current_folder_path = os.path.dirname(os.path.abspath(__file__))  # where the py scripts localized
    else:
        current_folder_path = path

    return current_folder_path


def load_dict_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# def save_dict_to_file(dictionary, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(dictionary, file, indent=4)


def update_dict_to_file(dictionary, file_path):
    try:
        # Try to read the existing JSON file
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        directory = os.path.dirname(file_path)
        # If the directory does not exist, create all necessary folders
        if not os.path.exists(directory):
            os.makedirs(directory)

        # If the file does not exist, create an empty dictionary
        existing_data = {}

    # Update the existing data
    existing_data.update(dictionary)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


# def run_command(working_directory_str, launch_command):
#     try:
#         # message_command = f"Executing '{launch_command}' in {working_directory_str}"
#         # print(message_command)
#         process = subprocess.Popen(launch_command, cwd=working_directory_str, shell=True,
#                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#         while process.poll() is None:
#             # time.sleep(0.1)
#             # Continuously output the results of command execution
#             output = process.stdout.readline().decode().strip()
#             if output:
#                 print(output)
#
#         stdout, stderr = process.communicate()
#         remaining_output = stdout.decode().strip()
#         if remaining_output:
#             print(remaining_output)
#
#         if process.poll() == 0:
#             print(f"Executing '{launch_command}' in {working_directory_str}: Success")
#         else:
#             print(f"Executing '{launch_command}' in {working_directory_str}: Failed!!!")
#             raise subprocess.CalledProcessError(process.returncode, launch_command, stderr.decode())
#
#     except subprocess.CalledProcessError as e:
#         print(f"\n#################################################################################")
#         print(f"Error: launch script terminated unexpectedly with exit code: {e.returncode}")
#         print("Error details:")
#         print(e.output)
#
#         raise e
#
# import subprocess

def run_command(working_directory_str, launch_command, continue_on_failure=False, silent_mode=False):
    try:
        process = subprocess.Popen(launch_command, cwd=working_directory_str, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while process.poll() is None:
            output = process.stdout.readline().decode().strip()
            if output:
                print(output)

        stdout, stderr = process.communicate()
        remaining_output = stdout.decode().strip()
        if remaining_output:
            print(remaining_output)

        if process.poll() == 0:
            if not silent_mode:
                print(f"Executing '{launch_command}' in {working_directory_str}: Success")
        else:
            if not silent_mode:
                print(f"Executing '{launch_command}' in {working_directory_str}: Failed!!!")
            if not continue_on_failure:
                raise subprocess.CalledProcessError(process.returncode, launch_command, stderr.decode())

    except subprocess.CalledProcessError as e:
        if not silent_mode:
            print(f"\n#################################################################################")
            print(f"Error: launch script terminated unexpectedly with exit code: {e.returncode}")
            print("Error details:")
            print(e.output)

        if not continue_on_failure:
            raise e


# Example usage:
# run_command("/path/to/directory", "some_command", continue_on_failure=True)


def check_file_exists(file_path):
    """Check if a file exists at the specified file path."""
    if os.path.exists(file_path):
        return True
    else:
        return False


def load_module(file_path):
    """
    Dynamically load a module from the specified file path.

    Parameters:
    - file_path (str): The path to the module file.

    Returns:
    - module: The loaded module object.
    """
    # Use a temporary name for the module, as module_name is not used directly
    module_name = 'loaded_module'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_suffix_to_folder(path, suffix_name):
    # Check if the path exists and is a directory
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory path.")
        return

    # Get the path and name of the folder
    folder_path, folder_name = os.path.split(path)

    # Add the suffix to the folder name
    new_folder_name = f"{folder_name}_{suffix_name}"
    new_folder_path = os.path.join(folder_path, new_folder_name)

    try:
        # Rename the folder with the new name
        os.rename(path, new_folder_path)
        print(f"Folder '{folder_name}' renamed to '{new_folder_name}'.")
    except Exception as e:
        print(f"Error: Failed to rename folder '{folder_name}': {e}")


def print_fixed_length(A, B, fixed_length):
    if len(A) < fixed_length:
        A = A.ljust(fixed_length)

    combined_string = A + B
    print(combined_string)


def find_file_path_from_two_folders(file_name_full, folder_path_class1, folder_path_class2):
    # Create the full paths for both directories
    path_a_full = os.path.join(folder_path_class1, file_name_full)
    path_b_full = os.path.join(folder_path_class2, file_name_full)

    # Check if the file exists in path A
    if check_file_exists(path_a_full):
        file_path = path_a_full
    elif check_file_exists(path_b_full):
        file_path = path_b_full
    else:
        raise FileNotFoundError(f"{file_name_full} not found in both {folder_path_class1} and {folder_path_class2}")

    return file_path


def in_slurm_environment():
    if 'SLURM_JOB_ID' in os.environ:
        return True
    else:
        return False


def get_slurm_info():
    job_id = os.environ.get('SLURM_JOB_ID')

    # Wait until the job starts
    start_time = None
    while start_time is None:
        result = subprocess.run(['scontrol', 'show', 'job', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        for line in result.stdout.split('\n'):
            if 'StartTime=' in line:
                start_time_str = line.split('StartTime=')[1].split()[0]
                if start_time_str != 'Unknown':
                    start_time = start_time_str
                    break
        if start_time is None:
            time.sleep(1)

    # Get job details
    result = subprocess.run(['scontrol', 'show', 'job', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    job_info = result.stdout

    # Parse job details
    job_details = {}
    for line in job_info.split('\n'):
        if 'JobId=' in line:
            job_details['JobId'] = line.split('JobId=')[1].split()[0]
        if 'Command=' in line:
            job_details['Command'] = line.split('Command=')[1].split()[0]
        if 'RunTime=' in line:
            job_details['RunTime'] = line.split('RunTime=')[1].split()[0]
        if 'TimeLimit=' in line:
            job_details['TimeLimit'] = line.split('TimeLimit=')[1].split()[0]
        if 'StartTime=' in line:
            job_details['StartTime'] = line.split('StartTime=')[1].split()[0]

    # Calculate remaining time
    if 'StartTime' in job_details and 'TimeLimit' in job_details:
        start_time_dt = datetime.datetime.strptime(job_details['StartTime'], '%Y-%m-%dT%H:%M:%S')
        start_time_ts = start_time_dt.timestamp()
        time_limit = job_details['TimeLimit']

        # Convert TimeLimit to seconds
        days, time_part = time_limit.split('-') if '-' in time_limit else (0, time_limit)
        h, m, s = map(int, time_part.split(':'))
        total_allowed_time = int(days) * 86400 + h * 3600 + m * 60 + s

        # Convert RunTime to seconds
        h, m, s = map(int, job_details['RunTime'].split(':'))
        run_time_seconds = h * 3600 + m * 60 + s

        # Calculate remaining time
        current_time = time.time()
        elapsed_time = current_time - start_time_ts
        remaining_time = total_allowed_time - elapsed_time

        job_details['RunTime'] = run_time_seconds
        job_details['TimeLimit'] = total_allowed_time
        job_details['StartTime'] = start_time_ts
        job_details['RemainingTime'] = remaining_time

    return job_details


def modify_variable_in_pyfile(file_path, variable_name, new_value):
    pattern = re.compile(rf'(\s*{variable_name}\s*=\s*)([^#]*)(#.*)?')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            prefix, old_value, comment = match.groups()
            old_value_stripped = old_value.strip()

            # if new_value is None:
            #     return old_value_stripped

            new_value_formatted = str(new_value)

            if old_value_stripped == new_value_formatted:
                return

            leading_spaces = len(old_value) - len(old_value.lstrip())
            trailing_spaces = len(old_value) - len(old_value.rstrip())
            new_value_formatted_full = ' ' * leading_spaces + new_value_formatted + ' ' * trailing_spaces

            lines[i] = f'{prefix}{new_value_formatted_full}{comment or ""}\n'

            with open(file_path, 'w') as file:
                file.writelines(lines)

            print(f"Value of {variable_name} in {file_path}: Updated successfully.")
            return

    print(f"Variable '{variable_name}' in {file_path}: Not found.")
    return None


def get_variable_from_module(module, variable_name):
    """
    Get the value of a variable from a module by its name, with a default of None if not found.

    Parameters:
    - module: The loaded module object.
    - variable_name (str): The name of the variable to retrieve.

    Returns:
    - The value of the variable, or None if not found.
    """
    return getattr(module, variable_name, None)


def replace_filename_part(path, old_part, new_part):
    # Split the path into directory and filename
    directory, filename = os.path.split(path)
    # Replace the specified part of the filename
    new_filename = filename.replace(old_part, new_part)
    # Join the directory and new filename to form the new path
    new_path = os.path.join(directory, new_filename)
    return new_path

# # to be changed
# def find_data(file_path, key_word):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             if key_word in line:
#                 return line


def create_folder_path(path):
    # Use os.makedirs() to create the directory, and it won't raise an error if it already exists
    os.makedirs(path, exist_ok=True)
    # print(f"Directory '{path}' created successfully.")


def interp2linear_default(z, xi, yi, extrapval=np.nan):
    """
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB

    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required
    @param yi: matrix of y coordinates where interpolation is required
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """

    x = xi.copy()
    y = yi.copy()
    nrows, ncols = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")

    # find x values out of range
    x_bad = ((x < 0) | (x > ncols - 1))
    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    z = z.ravel()
    f = (z[ndx] * one_minus_t + z[ndx + ncols] * y) * (1 - x) + (
            z[ndx + 1] * one_minus_t + z[ndx + ncols + 1] * y) * x

    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f


def interp2linear(X, Y, V, Xq, Yq):
    x_1d = X[:, 1]
    y_1d = Y[1]
    xi = np.interp(Xq, x_1d, range(len(x_1d)))
    yi = np.interp(Yq, y_1d, range(len(y_1d)))
    # print(xi)
    # print(yi)

    # result = interp2linear_default(V, xi, yi)
    result = interp2linear_default(V, yi, xi)

    return result

def griddata_classic(polygons_x, polygons_y, polygons_value, grid_x, grid_y, method='linear'):
    coordinate_xy = np.vstack((polygons_x.flatten(), polygons_y.flatten()))
    coordinate_xy_t = coordinate_xy.T
    grid_value = griddata(coordinate_xy_t, polygons_value.flatten(), (grid_x, grid_y), method=method)

    return grid_value


def find_param_data(file_path, key_word1, key_word2):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if (key_word1 in line) and (key_word2 in line):
                return line

def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def boolean_points_inout_polygon(points_x, points_y, polygon_x, polygon_y, side="in", method="contains_points"):
    """
    :param points_x/points_y: numpy.ndarray, 1D/2D
    :param polygon_x/polygon_y: numpy.ndarray, 1D
    :param side: "in"/"out"
    :return: index of points
    """

    points_x_flatten = points_x.flatten()
    points_y_flatten = points_y.flatten()

    poly_data = np.vstack((polygon_x, polygon_y)).T
    points_data = np.vstack((points_x_flatten, points_y_flatten)).T

    if method == "contains_points":
        p = Path(poly_data)  # make a polygon
        bol_list = p.contains_points(points_data)
    elif method == "is_in_poly":
        bol_list = []
        for i in range(len(points_x_flatten)):
            print(i)
            point_data = points_data[i]
            if is_in_poly(point_data, poly_data):
                bol_value = True
            else:
                bol_value = False
        bol_list.append(bol_value)

    bol_list = np.array(bol_list)

    points_bol = bol_list.reshape(points_x.shape)

    if side == "out":
        points_bol = (points_bol == False)

    return points_bol


def polygon_area(polygon):
    """
    compute polygon area
    polygon: list with shape [n, 2], n is the number of polygon points
    """
    area = 0
    q = polygon[-1]
    for p in polygon:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area) / 2.0


def polygons_integral_volume(polygons_x, polygons_y, polygons_value):
    value_int = 0
    for i in range(polygons_value.shape[0]):
        list_x = polygons_x[i, :]
        list_y = polygons_y[i, :]
        value = polygons_value[i, 0]
        polygon = np.vstack((list_x, list_y)).T
        area = polygon_area(polygon)

        value_int += value * area * 2 * np.pi * np.mean(list_x)

    return value_int

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr


def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def flipR(arr):
    i0 = arr.shape[0]
    i1 = arr.shape[1]
    arr_new = np.zeros((i0, i1))
    for i in range(i0):
        for j in range(i1):
            arr_new[i, j] = arr[i, i1 - j - 1]
            # print(i,j,i0-i-1,i1-j-1)
    return arr_new

# def detect_key_word(file_path, key_word):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             if key_word in line:
#                 return True