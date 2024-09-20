import os
import re
import argparse

from functions_general import define_controlled_code_folder_path, define_script_folder_path, run_command


def modify_slurm_script(input_file, output_file, auto_run_setup, script_folder_path):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        if is_slurm_script(input_file):
            for line in lines:
                if line.startswith('source'):
                    match = re.search(r'source\s+(.+)', line)
                    script_name = match.group(1)
                    export_command = f'export execute_commands="source {script_name}"\n'
                    file.write(export_command)
                elif line.startswith('srun'):
                    match = re.search(r"srun\s+(.+)", line)
                    script_name = match.group(1)
                    export_command = f'export execute_commands="srun {script_name}"\n'
                    file.write(export_command)
                else:
                    file.write(line)
        else:
            file.write("#!/bin/bash\n\n")
            commands = extract_commands_from_bash_file(input_file)
            command_string = convert_to_command_string(commands)
            export_command = f'export execute_commands="{command_string}"\n'
            file.write(export_command)


        file.write('export running_time_limit_in_hours=10000\n')
        file.write('export iteration_number_limit=10000\n')

        file.write('\n')

        file.write(f'export auto_run_setup="{auto_run_setup}"\n')

        if is_in_same_folder(script_folder_path, input_file):
            file.write('export code_folder_path="auto_run_code"\n')
        else:
            file.write(f'export code_folder_path="{script_folder_path}"\n')

        file.write('\n')
        file.write('chmod +x $code_folder_path/launch_plugin.sh\n')
        file.write('$code_folder_path/launch_plugin.sh\n')


def extract_commands_from_bash_file(file_path):
    # Use a regular expression to match common Bash command patterns
    # Assume commands are at the start of a line, excluding comments and empty lines
    command_pattern = re.compile(r'^\s*(?!#)(.+)$')

    commands = []
    with open(file_path, 'r') as file:
        for line in file:
            match = command_pattern.match(line)
            if match:
                commands.append(match.group(1).strip())

    return commands


def convert_to_command_string(commands):
    """
    Convert a list of commands into a single string separated by semicolons.

    Args:
        commands (list): A list of command strings.

    Returns:
        str: A single string with commands separated by semicolons.
    """
    return '; '.join(commands)


def is_slurm_script(file_path):
    """
    Check if the given file is a Slurm script.

    :param file_path: Path to the Bash script file.
    :return: True if the script contains Slurm directives, False otherwise.
    """
    slurm_directives = ["#SBATCH", "srun", "sbatch", "salloc"]

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if any(directive in line for directive in slurm_directives):
                    return True
        return False
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def is_in_same_folder(script_folder_path, input_file):
    parent_path1 = os.path.dirname(script_folder_path)
    parent_path2 = os.path.dirname(input_file)
    if parent_path1 == parent_path2:
        return True
    else:
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that takes two arguments, with the second argument being optional.")

    # Add the first argument (required)
    parser.add_argument("launch_script_name", help="The first argument, launch_script_name")

    # Add the second argument (optional) with a default value
    parser.add_argument("auto_run_setup", nargs="?", default="setup_default.py", help="The second argument (optional), auto_run_setup")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    # main(args.launch_script_name, args.auto_run_setup)

    controlled_code_folder_path = define_controlled_code_folder_path()
    script_folder_path = define_script_folder_path()





    input_filename = args.launch_script_name  # replace with your original file name
    output_filename = f"auto_run_{input_filename}"

    input_file_path = os.path.join(controlled_code_folder_path, input_filename)
    output_file_path = os.path.join(controlled_code_folder_path, output_filename)

    modify_slurm_script(input_file_path, output_file_path, args.auto_run_setup, script_folder_path)
    run_command(controlled_code_folder_path, f"chmod +x {output_file_path}", silent_success=True)




    print(f"Modified script saved as {output_filename}\n")

    if is_slurm_script(input_file_path):
        if is_in_same_folder(script_folder_path, input_file_path):
            print("To run the Slurm job, use:")
            print(f"python3 auto_run_code/running_control.py slurm_run {output_filename}")
            print("\nTo safely stop the Slurm job, use:")
            print(f"python3 auto_run_code/running_control.py safe_stop {output_filename}")
        else:
            print("To run the Slurm job, use:")
            print(f"python3 {os.path.join(script_folder_path, 'running_control.py')} slurm_run {output_filename}")
            print("\nTo safely stop the Slurm job, use:")
            print(f"python3 {os.path.join(script_folder_path, 'running_control.py')} safe_stop {output_filename}")
    else:
        if is_in_same_folder(script_folder_path, input_file_path):
            print("To run the local job, use:")
            print(f"./{output_filename}")
            print("\nTo safely stop the local job, use:")
            print(f"python3 auto_run_code/running_control.py safe_stop {output_filename}")
        else:
            print("To run the local job, use:")
            print(f"./{output_filename}")
            print("\nTo safely stop the local job, use:")
            print(f"python3 {os.path.join(script_folder_path, 'running_control.py')} safe_stop {output_filename}")



