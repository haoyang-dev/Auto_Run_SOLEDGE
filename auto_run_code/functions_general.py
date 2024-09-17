import json
import os
import subprocess
import importlib.util
import time
import datetime
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


def job_termination_and_mark(job_running_mode, controlled_code_folder_path,
                             running_info_file_path, jobid_file_path, decision_of_loop,
                             change_folder_name_after_finished, marker, exit_code):
    if job_running_mode == 'slurm':
        update_dict_to_file({'slurm_job_id': np.nan}, running_info_file_path)
        if check_file_exists(jobid_file_path):
            run_command(controlled_code_folder_path, f"rm -r {jobid_file_path}")

    update_dict_to_file({
        'slurm_iteration_number': 0,
        'running_state': decision_of_loop,
        'manual_stop': 'off',
    }, running_info_file_path)

    if change_folder_name_after_finished == 'on' and marker:
        add_suffix_to_folder(controlled_code_folder_path, marker)

    exit(exit_code)


if __name__ == "__main__":
    pass
