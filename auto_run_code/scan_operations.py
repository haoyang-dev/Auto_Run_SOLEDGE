import os
import subprocess
import sys


def run_command(working_directory_str, launch_command):
    try:
        # message_command = f"Excute command: {launch_command}\nWorking directory: {working_directory_str}"
        # print(message_command)

        process = subprocess.Popen(launch_command, cwd=working_directory_str, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while process.poll() is None:
            # time.sleep(0.1)
            # Continuously output the results of command execution
            output = process.stdout.readline().decode().strip()
            if output:
                print(output)

        stdout, stderr = process.communicate()
        remaining_output = stdout.decode().strip()
        if remaining_output:
            print(remaining_output)

        if process.poll() == 0:
            pass
            # print(f"The subprocess '{launch_command}' executed successfully!")
        else:
            raise subprocess.CalledProcessError(process.returncode, launch_command, stderr.decode())

    except subprocess.CalledProcessError as e:
        print(f"\n#################################################################################")
        print(f"Error: launch script terminated unexpectedly with exit code: {e.returncode}")
        print("Error details:")
        print(e.output)

        raise e


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


def find_folders_with_string(root_dir, target_string):
    matching_folders = []
    for root, dirs, files in os.walk(root_dir):
        if root == root_dir:  # Check if current directory is root directory
            for dir_name in dirs:
                if target_string in dir_name:
                    matching_folders.append(os.path.join(root, dir_name))
        else:
            break  # Stop iteration if current directory is not root directory
    matching_folders.sort()
    return matching_folders


if __name__ == "__main__":
    controlled_code_folder_path = define_controlled_code_folder_path()

    folder_keywords = input("====> Enter the target string to search for: ")

    matching_folders = find_folders_with_string(controlled_code_folder_path, folder_keywords)
    if matching_folders:
        print(f"{len(matching_folders)} folders containing '{folder_keywords}' in their name:")
        for folder_path in matching_folders:
            print(folder_path)
    else:
        print(f"No folders found containing '{folder_keywords}' in their name.")

    command_str = input("====> Enter the command to execute:")

    for folder_path in matching_folders:
        print(f"\nOperations in path: {folder_path}")
        run_command(folder_path, command_str)
        print(f"{command_str}: Success")
