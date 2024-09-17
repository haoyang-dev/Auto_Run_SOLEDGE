"""
Author: Hao YANG
Email: hao.yang.ac@outlook.com
Date: 2024-06
Description: This script includes setup for the auto_run_code and functions that are executed at different stages
             of a loop, including initialization, post treatment tasks, value checks,
             and finalization.
"""
# ================================================
# Basic Settings
# These settings control the runtime duration and the number of iterations.
# ================================================
# running_time_limit_in_hours = 7 * 24  # 7 days
# iteration_number_limit = 100

# ================================================
# Advanced Settings
# These settings control the thresholds and allow for running specific processes at the start and end of the computation.
# ================================================
change_folder_name_after_finished = 'off'  # on / off, be careful to turn on this function
loop_stopped_by = 'time_iteration_limit'  # min, max critical_value_reached, mission_completed, time_iteration_limit
critical_value = 1e-10


def missions_before_loop(run_info_dict):
    """
    This function is executed once before the loop starts.
    You can add any initialization or setup code here.
    """
    # Uncomment the lines below for debugging
    # print("\n+++++++++++++++++++++++++++")
    # print("This is the run before loop")
    # print("+++++++++++++++++++++++++++")
    # print(f"controlled_code_folder_path: {run_info_dict['controlled_code_folder_path']}")
    # print(f"script_folder_path: {run_info_dict['script_folder_path']}")
    pass


def missions_after_command_during_loop(run_info_dict):
    """
    This function is executed after each iteration of the loop.
    You can add any post treatment tasks or checks here.
    """
    # Uncomment the lines below for debugging
    # print("\n+++++++++++++++++++++++++++++++")
    # print("This is the run after each loop")
    # print("+++++++++++++++++++++++++++++++")
    pass


def update_check_value_during_loop(run_info_dict):
    """
    This function updates and returns a check value during each iteration of the loop.
    You can add logic to update and return the check value here.

    Returns:
        check_value: The updated check value after this iteration.
    """
    # Uncomment the lines below for debugging
    # print("\n+++++++++++++++++++++++++++++++")
    # print("This is the run for check value")
    # print("+++++++++++++++++++++++++++++++")
    check_value = 1
    return check_value


def missions_after_loop(run_info_dict):
    """
    This function is executed once after the loop ends.
    You can add any cleanup or finalization code here.
    """
    # Uncomment the lines below for debugging
    # print("\n+++++++++++++++++++++++++++++++")
    # print("This is the run of end")
    # print("+++++++++++++++++++++++++++++++")
    pass
