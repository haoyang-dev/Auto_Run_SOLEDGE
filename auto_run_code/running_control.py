import os
import sys
import re
import numpy as np

from functions_general import check_file_exists, update_dict_to_file, run_command, load_dict_from_file, \
    add_suffix_to_folder, define_controlled_code_folder_path, define_script_folder_path, \
    find_file_path_from_two_folders, load_module


def operation_functions(operation, launch_script_name):
    if operation == 'slurm_run':
        run_command(controlled_code_folder_path, f"rm -r {jobid_file_path}", continue_on_failure=True, silent_mode=True)

        if check_file_exists(running_info_file_path):
            dict_init = {'slurm_job_id': np.nan,
                         'manual_stop': 'off',
                         'slurm_iteration_number': 0,
                         'running_state': 'not_start'}
            update_dict_to_file(dict_init, running_info_file_path)

        sbatch_command = f"sbatch ./{launch_script_name}"
        save_slurm_jobid_command = f" | tee {jobid_file_name}"
        sbatch_command_full = f"{sbatch_command}{save_slurm_jobid_command}"
        run_command(controlled_code_folder_path, sbatch_command_full, silent_success=True)

    elif operation == 'safe_stop':
        if check_file_exists(jobid_file_path):
            if check_file_exists(running_info_file_path):
                running_info_dict = load_dict_from_file(running_info_file_path)
                if running_info_dict.get('running_state') in {'running', 'temporary_stop'}:
                    update_dict_to_file({'manual_stop': 'on'}, running_info_file_path)
                    print("Job will be stopped when this iteration completes, please wait...")
                else:
                    jobid_str = read_number_from_file(jobid_file_path)
                    perform_quick_stop(jobid_str)
            else:
                jobid_str = read_number_from_file(jobid_file_path)
                perform_quick_stop(jobid_str)

        elif check_file_exists(running_info_file_path):
            running_info_dict = load_dict_from_file(running_info_file_path)
            if running_info_dict.get('running_state') in {'running', 'temporary_stop'}:
                update_dict_to_file({'manual_stop': 'on'}, running_info_file_path)
                print("Job will be stopped when this iteration completes, please wait...")
            else:
                apply_safe_stop(running_info_file_path)
        else:
            apply_safe_stop(running_info_file_path)

    elif operation == 'slurm_quick_stop':
        if check_file_exists(jobid_file_path):
            jobid_str = read_number_from_file(jobid_file_path)
            perform_quick_stop(jobid_str)
        elif check_file_exists(running_info_file_path):
            running_info_dict = load_dict_from_file(running_info_file_path)
            if running_info_dict.get('running_state') == 'running':
                perform_quick_stop(running_info_dict['slurm_job_id'])
            else:
                apply_safe_stop(running_info_file_path)
        else:
            apply_safe_stop(running_info_file_path)

    else:
        raise ValueError(f'Operation {operation} not recognized')


def perform_quick_stop(jobid_str):
    change_folder_name_after_finished = setup_module.change_folder_name_after_finished

    update_dict_to_file({'slurm_job_id': np.nan}, running_info_file_path)
    if check_file_exists(jobid_file_path):
        run_command(controlled_code_folder_path, f"rm -r {jobid_file_path}", silent_success=True)

    update_dict_to_file({
        'slurm_iteration_number': 0,
        'running_state': 'quick_stop',
        'manual_stop': 'off',
    }, running_info_file_path)

    if change_folder_name_after_finished == 'on':
        add_suffix_to_folder(controlled_code_folder_path, 'quick_stop')

    run_command(controlled_code_folder_path, f"scancel {jobid_str}", silent_success=True)

    print(f"Job {jobid_str} stopped successfully")


def apply_safe_stop(running_info_file_path):
    update_dict_to_file({'manual_stop': 'on'}, running_info_file_path)
    print(
        "Job not executed or pending, but safe stop is applied. Pending job will not start anyway")


# def read_number_from_file(file_path):
#     with open(file_path, 'r') as file:
#         return file.read().strip()

def read_number_from_file(file_path):
    job_id = None
    with open(file_path, 'r') as file:
        content = file.read()
        # Use regular expression to extract the first continuous number sequence
        match = re.search(r'\b\d+\b', content)
        if match:
            job_id = match.group(0)
    return job_id


# def read_variable_from_py(file_path, variable_name):
#     spec = importlib.util.spec_from_file_location("module_name", file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return getattr(module, variable_name, None)


def extract_auto_run_setup_name_from_file(file_path):
    with open(file_path, 'r') as f:
        bash_script = f.read()

    pattern = r'export\s+auto_run_setup\s*=\s*"([^"]+)"'
    match = re.search(pattern, bash_script)
    if match:
        return match.group(1)
    else:
        return None


if __name__ == "__main__":
    output_folder_name = 'auto_run_output'
    running_info_file_name = 'running_info.json'
    jobid_file_name = 'auto_run_jobid.txt'

    controlled_code_folder_path = define_controlled_code_folder_path()
    output_folder_path = os.path.join(controlled_code_folder_path, output_folder_name)
    running_info_file_path = os.path.join(output_folder_path, running_info_file_name)
    jobid_file_path = os.path.join(controlled_code_folder_path, jobid_file_name)
    script_folder_path = define_script_folder_path()

    if len(sys.argv) > 1:
        operation = sys.argv[1]
        # print(f"operation: {operation}")

        if len(sys.argv) > 2:
            launch_script_name = sys.argv[2]
        else:
            launch_script_name = input("Please provide the launch file name: ")
    else:
        print("Example of operations: slurm_run, safe_stop, slurm_quick_stop")
        operation = input("====> Enter the operation: ")
        launch_script_name = input("Please provide the launch file name: ")

    launch_script_path = os.path.join(controlled_code_folder_path, launch_script_name)
    setup_file_name_full = extract_auto_run_setup_name_from_file(launch_script_path)

    setup_file_path = find_file_path_from_two_folders(setup_file_name_full, controlled_code_folder_path,
                                                      script_folder_path)
    setup_module = load_module(setup_file_path)

    operation_functions(operation, launch_script_name)
