"""
Author: Hao YANG
Email: hao.yang.ac@outlook.com
Date: 2024-06
"""
import argparse
import math
import os
import time

import numpy as np

from functions_general import define_controlled_code_folder_path, define_script_folder_path, check_file_exists, \
    load_dict_from_file, run_command, print_fixed_length, update_dict_to_file, load_module, \
    find_file_path_from_two_folders, in_slurm_environment, get_slurm_info, add_suffix_to_folder


class CodePath:
    def __init__(self):
        output_folder_name = "auto_run_output"
        running_info_file_name = "running_info.json"
        jobid_file_name = "auto_run_jobid.txt"

        # self.controlled_code_folder_path = controlled_code_folder_path
        self.controlled_code_folder_path = define_controlled_code_folder_path()  # the path where the controlled code localised
        self.script_folder_path = define_script_folder_path()
        self.output_folder_path = os.path.join(self.controlled_code_folder_path, output_folder_name)
        self.jobid_file_path = os.path.join(self.controlled_code_folder_path, jobid_file_name)
        self.running_info_file_path = os.path.join(self.output_folder_path, running_info_file_name)
        # running_info_file_path
        # jobid_file_path

        # update_dict_to_file({'slurm_job_id': np.nan}, )
        # if check_file_exists():
        #     run_command(, f"rm -r {}")
        # output_folder_name = "auto_run_output"
        # self.controlled_code_folder_path = controlled_code_folder_path
        # self.case_folder_name = os.path.basename(controlled_code_folder_path)
        # self.parent_directory = os.path.dirname(controlled_code_folder_path)
        # setup_file_path = find_file_path_from_two_folders(setup_file, self.controlled_code_folder_path,
        #                                                   self.script_folder_path)


# class RunInfo:
#     def __init__(self, controlled_code_folder_path, script_folder_path, job_running_mode):
#         self.controlled_code_folder_path = controlled_code_folder_path
#         self.script_folder_path = script_folder_path
#         self.job_running_mode = job_running_mode
#
#         # self.run_info_dict = {'controlled_code_folder_path': self.code_paths.controlled_code_folder_path,
#         #                       'script_folder_path': self.code_paths.script_folder_path,
#         #                       'job_running_mode': self.job_running_mode}

#
class RunningManager:
    def __init__(self, run_mode, setup_file, running_time_limit_in_hours, iteration_number_limit):
        # controlled_code_folder_path = define_controlled_code_folder_path()  # the path where the controlled code localised
        # self.script_folder_path = define_script_folder_path()
        self.code_paths = CodePath()

        setup_file_path = find_file_path_from_two_folders(setup_file, self.code_paths.controlled_code_folder_path,
                                                          self.code_paths.script_folder_path)
        self.setup_module = load_module(setup_file_path)

        self.iteration_number = -1
        self.slurm_iteration_number = -1
        self.left_time_ratio = 1.25
        initialize_manual_stop = False
        self.sbatch_command_full = None

        # default setting, better not to modify
        # sbatch_cache_file_name = "sbatch_cache.json"
        # output_folder_name = "auto_run_output"
        # running_info_file_name = "running_info.json"
        # self.jobid_file_name = "auto_run_jobid.txt"
        self.sbatch_command = "sbatch"
        self.normally_stopped_code = 211
        self.running_time_limit_in_hours = running_time_limit_in_hours
        self.iteration_number_limit = iteration_number_limit

        self.save_slurm_jobid_command = f"| tee {self.code_paths.jobid_file_path}"
        # self.output_folder_path = os.path.join(self.controlled_code_folder_path, output_folder_name)
        # self.jobid_file_path = os.path.join(self.controlled_code_folder_path, self.jobid_file_name)

        os.makedirs(self.code_paths.output_folder_path, exist_ok=True)

        # self.sbatch_cache_file_path = os.path.join(self.output_folder_path, sbatch_cache_file_name)
        # self.running_info_file_path = os.path.join(self.output_folder_path, running_info_file_name)

        if in_slurm_environment():
            self.job_running_mode = "slurm"
        else:
            self.job_running_mode = "local"

        # if manual_stop = on, in the initialize turn, stop directly, in the supervise turn, doing nothing.
        if check_file_exists(self.code_paths.running_info_file_path):
            self.running_info = load_dict_from_file(self.code_paths.running_info_file_path)
            if self.running_info.get('manual_stop') == 'on':
                if run_mode == "initialize":
                    self.decision_of_loop = 'manual_stop'
                    self.step_after_decision()
                    run_command(self.code_paths.controlled_code_folder_path,
                                f"rm -r {self.code_paths.output_folder_path}")
                    exit(self.normally_stopped_code)
                else:
                    pass
            else:
                initialize_manual_stop = True
        else:
            initialize_manual_stop = True

        if check_file_exists(self.code_paths.running_info_file_path):
            self.running_info = load_dict_from_file(self.code_paths.running_info_file_path)
            if 'time_used' in self.running_info:  # only do the following when calculation starts
                self.iteration_number = self.running_info['iteration_number']
                self.slurm_iteration_number = self.running_info['slurm_iteration_number']
                self.time_used = self.running_info['time_used']

        if run_mode == "initialize":
            self.current_iteration_start_time = time.time()
            update_dict_to_file({
                'running_state': 'running',
            }, self.code_paths.running_info_file_path)
        elif run_mode == "supervise":
            self.current_iteration_start_time = self.running_info['current_iteration_start_time']
        else:
            raise ValueError(f"Invalid run mode: {run_mode}")

        # class RunInfo:
        #     def __init__(self):
        #         self.controlled_code_folder_path = self.code_paths.controlled_code_folder_path

        # self.run_info = RunInfo(controlled_code_folder_path=self.code_paths.controlled_code_folder_path,
        #                         script_folder_path=self.code_paths.script_folder_path,
        #                         job_running_mode=self.job_running_mode,
        #                         running_info_file_path=self.code_paths.running_info_file_path,
        #                         jobid_file_path=)

        self.run_info_dict = {'controlled_code_folder_path': self.code_paths.controlled_code_folder_path,
                              'script_folder_path': self.code_paths.script_folder_path,
                              'job_running_mode': self.job_running_mode,
                              'running_info_file_path': self.code_paths.running_info_file_path,
                              'jobid_file_path': self.code_paths.jobid_file_path,
                              'change_folder_name_after_finished': self.setup_module.change_folder_name_after_finished
                              }

        if self.iteration_number == -1:
            print(f"================ Execution Started ================")
            self.setup_module.missions_before_loop(self.run_info_dict)
            self.time_used = 0

        if not (run_mode == "initialize" and self.iteration_number > -1):
            self.iteration_number += 1
            self.slurm_iteration_number += 1

        update_dict_to_file({
            'job_runing_mode': self.job_running_mode,
            'controlled_code_folder_path': self.code_paths.controlled_code_folder_path,
            'setup_file_path': setup_file_path,
            'current_iteration_start_time': self.current_iteration_start_time,
            'iteration_number': self.iteration_number,
            'slurm_iteration_number': self.slurm_iteration_number,
            'time_used': self.time_used,
        }, self.code_paths.running_info_file_path)

        if initialize_manual_stop:
            update_dict_to_file({
                'manual_stop': 'off',
            }, self.code_paths.running_info_file_path)

        if self.job_running_mode == 'slurm':
            slurm_info = get_slurm_info()
            slurm_job_id = slurm_info.get('JobId', 'N/A')
            self.slurm_command = slurm_info.get('Command', 'N/A')
            self.slurm_start_time = float(slurm_info.get('StartTime', 'N/A'))
            self.slurm_time_limit = float(slurm_info.get('TimeLimit', 'N/A'))
            self.slurm_remaining_time = float(slurm_info.get('RemainingTime', 'N/A'))
            self.slurm_run_time = float(slurm_info.get('RunTime', 'N/A'))
            update_dict_to_file({'slurm_job_id': slurm_job_id, 'slurm_command': self.slurm_command},
                                self.code_paths.running_info_file_path)

            self.sbatch_command_full = f"{self.sbatch_command} {self.slurm_command} {self.save_slurm_jobid_command}"

        self.command_executed_message = f"\n>>>>> Command executed, iteration number: {self.iteration_number + 1} <<<<<"

        if run_mode == "initialize":
            print(self.command_executed_message)

    def missions_after_command_during_loop(self):
        print(f">>>>> Command execution complete <<<<<")
        self.setup_module.missions_after_command_during_loop(self.run_info_dict)
        self.check_value = self.setup_module.update_check_value_during_loop(self.run_info_dict)

    def update_info_of_auto_run(self):

        self.realtime_after_command_complete = time.time()
        self.current_time_used = self.realtime_after_command_complete - self.current_iteration_start_time + self.time_used

        update_dict_to_file({
            'iteration_number': self.iteration_number,
            'slurm_iteration_number': self.slurm_iteration_number,
            'time_used': self.current_time_used,
            'current_iteration_start_time': self.realtime_after_command_complete,
        }, self.code_paths.running_info_file_path)

        if self.job_running_mode == 'local':
            self.time_per_loop = self.current_time_used / self.iteration_number
        elif self.job_running_mode == 'slurm':
            self.time_per_loop = self.slurm_run_time / self.slurm_iteration_number
        else:
            raise ValueError

        formatted_time_per_loop = '{:.3f}'.format(self.time_per_loop / 3600)
        formatted_time_used = '{:.3f}'.format(self.current_time_used / 3600)
        formatted_hours_time_limit = '{:.3f}'.format(self.running_time_limit_in_hours)

        print(f"\n##### Information of auto run #####")

        fixed_length = 55

        print_fixed_length("Average time per loop (hour):", f"{formatted_time_per_loop}", fixed_length)

        if self.job_running_mode == 'slurm':
            estimated_slurm_limitation = max(1, math.ceil(
                self.slurm_time_limit / self.time_per_loop - self.left_time_ratio))
            print_fixed_length(
                f"Slurm iteration number / Estimated limitation:",
                f"{self.slurm_iteration_number} / {estimated_slurm_limitation}", fixed_length)

        print_fixed_length(
            f"Total iteration number completed / Maximum iterations:",
            f"{self.iteration_number} / {self.iteration_number_limit}", fixed_length)

        if self.job_running_mode == 'slurm':
            sbatch_formatted_hours_time_limit = '{:.3f}'.format(self.slurm_time_limit / 3600)
            sbatch_formatted_run_time = '{:.3f}'.format(self.slurm_run_time / 3600)
            print_fixed_length(
                f"Slurm running time used / Maximum allowed (hours):",
                f"{sbatch_formatted_run_time} / {sbatch_formatted_hours_time_limit}", fixed_length)

        print_fixed_length(f"Total running time used / Maximum allowed (hours):",
                           f"{formatted_time_used} / {formatted_hours_time_limit}", fixed_length)

        print_fixed_length(f"Check value / Critical value:", f"{self.check_value} / {self.setup_module.critical_value}",
                           fixed_length)

    def decision_of_running(self):

        iteration_number_limit_return = False
        hours_time_limit_return = False
        loop_stopped_by_return = False
        sbatch_hours_time_limit_return = False
        manual_stop_return = False

        # check the iteration_number_limit
        if self.iteration_number >= self.iteration_number_limit:
            print(f"!!! Reach total iteration number limit !!! ")
            iteration_number_limit_return = True

        # check the hours_time_limit
        time_left = self.running_time_limit_in_hours * 3600 - self.current_time_used
        if time_left < self.time_per_loop * self.left_time_ratio:
            print("!!! Close to running time limitation !!!")
            hours_time_limit_return = True

        # check the sbatch_hours_time_limit
        if self.job_running_mode == 'slurm':
            if self.slurm_remaining_time < self.time_per_loop * self.left_time_ratio:
                print("!!! The remaining slurm time may not be enough to finish the next loop !!!")
                sbatch_hours_time_limit_return = True

        # check the loop_stopped_by
        loop_stopped_by = self.setup_module.loop_stopped_by
        if loop_stopped_by == 'min_critical_value_reached':
            if self.check_value <= self.setup_module.critical_value:
                print(
                    f"!!! min_critical_value_reached: check_value {self.check_value} <= critical_value {self.setup_module.critical_value} !!!")
                loop_stopped_by_return = True
        elif loop_stopped_by == 'max_critical_value_reached':
            if self.check_value >= self.setup_module.critical_value:
                print(
                    f"!!! max_critical_value_reached: check_value {self.check_value} >= critical_value {self.setup_module.critical_value} !!!")
                loop_stopped_by_return = True
        elif loop_stopped_by == 'mission_completed':
            if self.check_value == 1:
                print(f"!!! loop mission completed !!!")
                loop_stopped_by_return = True
        elif loop_stopped_by == 'time_iteration_limit':
            pass
        else:
            raise ValueError("Wrong loop_stopped_by name")

        # check manual_stop
        running_info_dict = load_dict_from_file(self.code_paths.running_info_file_path)
        if running_info_dict['manual_stop'] == 'on':
            manual_stop_return = True

        # final decision
        if manual_stop_return:
            result = 'manual_stop'
        elif iteration_number_limit_return or hours_time_limit_return or loop_stopped_by_return:
            result = 'real_stop'
        elif sbatch_hours_time_limit_return:
            result = 'temporary_stop'
        else:
            result = 'continue'

        self.decision_of_loop = result

    def step_after_decision(self):
        decision_of_loop = self.decision_of_loop

        if decision_of_loop == 'real_stop' or decision_of_loop == 'temporary_stop' or decision_of_loop == 'manual_stop':
            print(f"\n---------------- End of this part ----------------")
            print(f"Decision of loop: {decision_of_loop}")

            if self.job_running_mode == 'slurm':
                update_dict_to_file({'slurm_job_id': np.nan}, self.code_paths.running_info_file_path)
                if check_file_exists(self.code_paths.jobid_file_path):
                    run_command(self.code_paths.controlled_code_folder_path, f"rm -r {self.code_paths.jobid_file_path}")

            update_dict_to_file({
                'slurm_iteration_number': 0,
                'running_state': decision_of_loop,
                'manual_stop': 'off',
            }, self.code_paths.running_info_file_path)

            if decision_of_loop == 'real_stop':
                self.setup_module.missions_after_loop(self.run_info_dict)
                print(f"\n================ Execution Complete ================")
                # job_termination_and_mark(job_running_mode=self.job_running_mode,
                #                          controlled_code_folder_path=self.code_paths.controlled_code_folder_path,
                #                          running_info_file_path=self.code_paths.running_info_file_path,
                #                          jobid_file_path=self.code_paths.jobid_file_path,
                #                          decision_of_loop=decision_of_loop,
                #                          change_folder_name_after_finished=self.setup_module.change_folder_name_after_finished,
                #                          marker='finished',
                #                          exit_on=True,
                #                          exit_code=self.normally_stopped_code)
                if self.setup_module.change_folder_name_after_finished == 'on':
                    add_suffix_to_folder(self.code_paths.controlled_code_folder_path, "_finished")


            elif decision_of_loop == 'temporary_stop':
                # job_termination_and_mark(job_running_mode=self.job_running_mode,
                #                          controlled_code_folder_path=self.code_paths.controlled_code_folder_path,
                #                          running_info_file_path=self.code_paths.running_info_file_path,
                #                          jobid_file_path=self.code_paths.jobid_file_path,
                #                          decision_of_loop=decision_of_loop,
                #                          change_folder_name_after_finished=self.setup_module.change_folder_name_after_finished,
                #                          marker=None,
                #                          exit_on=False,
                #                          exit_code=self.normally_stopped_code)
                if self.sbatch_command_full:
                    run_command(self.code_paths.controlled_code_folder_path, self.sbatch_command_full)
                else:
                    raise ValueError("sbatch_command_full is empty")

            elif decision_of_loop == 'manual_stop':
                # job_termination_and_mark(job_running_mode=self.job_running_mode,
                #                          controlled_code_folder_path=self.code_paths.controlled_code_folder_path,
                #                          running_info_file_path=self.code_paths.running_info_file_path,
                #                          jobid_file_path=self.code_paths.jobid_file_path,
                #                          decision_of_loop=decision_of_loop,
                #                          change_folder_name_after_finished=self.setup_module.change_folder_name_after_finished,
                #                          marker='manual_stop',
                #                          exit_on=True,
                #                          exit_code=self.normally_stopped_code)
                if self.setup_module.change_folder_name_after_finished == 'on':
                    add_suffix_to_folder(self.code_paths.controlled_code_folder_path, "_manual_stop")

                # update_dict_to_file({
                #     'manual_stop': 'off',
                # }, self.code_paths.running_info_file_path)  # reset the manual stop

            exit(self.normally_stopped_code)  # "stop_running"

            # return self.normally_stopped_code  # "stop_running"

        else:
            print(self.command_executed_message)
            exit(0)  # "continue_running"
            # return 0  # "continue_running"


def arg_info():
    parser = argparse.ArgumentParser(description='Description of your script')

    # add_argument
    parser.add_argument('--mode', choices=['initialize', 'supervise'], default='supervise',
                        help='Specify the mode of operation (initialize or supervise)')

    parser.add_argument('--setup', metavar='SETUP_FILE', default='setup_default',
                        help='Specify the setup file (default: setup_defaut.py)')

    parser.add_argument('--time_limit', type=float, default=24,
                        help='Specify the time limit in hours (default: 24)')

    parser.add_argument('--iteration_limit', type=int, default=1000,
                        help='Specify the iteration limit (default: 1000)')

    args = parser.parse_args()

    return args


def main():
    args = arg_info()
    run_mode = args.mode
    setup_file = args.setup
    running_time_limit_in_hours = args.time_limit
    iteration_number_limit = args.iteration_limit

    running = RunningManager(run_mode, setup_file, running_time_limit_in_hours, iteration_number_limit)

    if run_mode == "supervise":
        running.missions_after_command_during_loop()
        running.update_info_of_auto_run()
        running.decision_of_running()
        running.step_after_decision()
    elif run_mode == "initialize":
        pass
    else:
        raise ValueError("Invalid run mode")


if __name__ == "__main__":
    main()
