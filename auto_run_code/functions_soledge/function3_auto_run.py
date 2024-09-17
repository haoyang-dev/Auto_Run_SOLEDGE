from .function1_basic import load_module, check_file_exists, run_command
from .function2_intermediate import update_target_value, PathSummary, get_data, update_gas_puff_map, \
    update_input_power_map, update, update_diff_map, save_plasma_file, Restart_check, Convergence_check, \
    job_termination_and_mark


def s3x_missions_before_loop(run_info_dict):
    paths = PathSummary(run_info_dict['controlled_code_folder_path'])
    save_plasma_file(paths)


def s3x_missions_after_command_during_loop(run_info_dict):
    paths = PathSummary(run_info_dict['controlled_code_folder_path'])
    control_setup = load_module(paths.control_setup_file_path)

    if check_file_exists(paths.plasma_final_file_path):

        get_data(paths)

        if control_setup.target_map == 'on':
            update_target_value(paths, run_info_dict)

        if control_setup.gas_puff_map == 'on':
            update_gas_puff_map(paths, run_info_dict)

        if control_setup.input_power_map == 'on':
            update_input_power_map(paths, run_info_dict)

        update(paths)

        if control_setup.profile_feedback == 'on':
            update_diff_map(paths)

        run_command(paths.controlled_code_folder_path, "mv run_dir/plasmaFinal.h5 run_dir/plasmaRestart.h5")

        save_plasma_file(paths)

        Restart_check(paths)

        # paths = PathSummary(run_info_dict['controlled_code_folder_path'])

        if control_setup.check_convergence == 'on':

            result = Convergence_check(paths)

            if result == 'converged':
                # save_plasma_file(paths)
                print('>>>>>> Calculation converged <<<<<<.')
                job_termination_and_mark(job_running_mode=run_info_dict['job_running_mode'],
                                         controlled_code_folder_path=run_info_dict['controlled_code_folder_path'],
                                         running_info_file_path=run_info_dict['running_info_file_path'],
                                         jobid_file_path=run_info_dict['jobid_file_path'],
                                         decision_of_loop='converged',
                                         change_folder_name_after_finished=run_info_dict[
                                             'change_folder_name_after_finished'],
                                         marker='converged',
                                         exit_code=211)



    else:
        print("calculation crashed")

        job_termination_and_mark(job_running_mode=run_info_dict['job_running_mode'],
                                 controlled_code_folder_path=run_info_dict['controlled_code_folder_path'],
                                 running_info_file_path=run_info_dict['running_info_file_path'],
                                 jobid_file_path=run_info_dict['jobid_file_path'],
                                 decision_of_loop='crashed',
                                 change_folder_name_after_finished=run_info_dict['change_folder_name_after_finished'],
                                 marker='crashed',
                                 exit_code=1)