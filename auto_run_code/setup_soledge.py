"""
Author: Hao YANG
Email: hao.yang.ac@outlook.com
Date: 2024-06
Description: This script includes setup for the auto_run_code and functions that are executed at different stages
             of a loop, including initialization, post treatment tasks, value checks,
             and finalization.
"""

from functions_soledge import s3x_missions_after_command_during_loop, s3x_missions_before_loop


# ================================================
# Advanced Settings
# These settings control the thresholds and allow for running specific processes at the start and end of the computation.
# ================================================
change_folder_name_after_finished = 'off'  # on / off, be careful to turn on this function
loop_stopped_by = 'time_iteration_limit'  # min, max critical_value_reached, mission_completed, time_iteration_limit
critical_value = 1e-10


def missions_before_loop(run_info_dict):
    s3x_missions_before_loop(run_info_dict)

def missions_after_command_during_loop(run_info_dict):
    s3x_missions_after_command_during_loop(run_info_dict)


def update_check_value_during_loop(run_info_dict):
    """
    This function updates and returns a check value during each iteration of the loop.
    You can add logic to update and return the check value here.

    Returns:
        check_value: The updated check value after this iteration.
    """

    check_value = 1
    return check_value


def missions_after_loop(run_info_dict):
    """
    This function is executed once after the loop ends.
    You can add any cleanup or finalization code here.
    """
    pass
