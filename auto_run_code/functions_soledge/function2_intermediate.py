from itertools import groupby
from intersect import intersection
from scipy.interpolate import griddata
from scipy import integrate
import shutil
import traceback
import h5py
import pandas as pd
import os
import re

import numpy as np

from .function1_basic import update_dict_to_file, check_file_exists, run_command, add_suffix_to_folder, \
    load_module, create_folder_path, interp2linear, get_variable_from_module, find_param_data, griddata_classic, \
    boolean_points_inout_polygon, modify_variable_in_pyfile, polygons_integral_volume, replace_filename_part, \
    flip90_right, flip180, flip90_left, flipR


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


class Point:
    def __init__(self, r, z, value):
        self.r = r
        self.z = z
        self.value = value


class Curve:
    """composed by points"""

    def __init__(self, r, z, value):
        self.r = r
        self.z = z
        self.value = value


class Polygon:
    def __init__(self, r, z, value):
        self.r = r
        self.z = z
        self.value = value


class Matrix:
    """composed by Polygons"""

    def __init__(self, r, z, value):
        self.r = r
        self.z = z
        self.value = value


class Grid:
    def __init__(self, r, z, value):
        self.r = r
        self.z = z
        self.value = value


class PathSummary:
    def __init__(self, controlled_code_folder_path):
        output_folder_name = "auto_run_output"
        self.controlled_code_folder_path = controlled_code_folder_path
        self.case_folder_name = os.path.basename(controlled_code_folder_path)
        self.parent_directory = os.path.dirname(controlled_code_folder_path)
        self.output_folder_path = os.path.join(controlled_code_folder_path, output_folder_name)
        self.mesh_folder_path = os.path.join(controlled_code_folder_path, "mesh")
        self.rundir_folder_path = os.path.join(controlled_code_folder_path, "run_dir")
        self.control_setup_file_path = os.path.join(controlled_code_folder_path, "auto_run_S3XE_control_setup.py")
        self.save_plasma_final_folder_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                          "auto_run_Collection_plasmaFinal")
        self.plasma_restart_file_path = os.path.join(controlled_code_folder_path, "run_dir", "plasmaRestart.h5")
        self.plasma_final_file_path = os.path.join(controlled_code_folder_path, "run_dir", "plasmaFinal.h5")
        self.diffusion_file_path = os.path.join(controlled_code_folder_path, "run_dir", "diffusion.h5")
        self.mesh_file_path = os.path.join(controlled_code_folder_path, "run_dir", "mesh.h5")
        self.refparm_raptorx_file_path = os.path.join(controlled_code_folder_path, "run_dir", "refParam_raptorX.h5")
        self.mesh_raptorx_file_path = os.path.join(controlled_code_folder_path, "run_dir", "mesh_raptorX.h5")
        self.mesh_eirene_file_path = os.path.join(controlled_code_folder_path, "run_dir", "meshEIRENE.h5")
        self.balances_0_file_path = os.path.join(controlled_code_folder_path, "run_dir", "balances_0")
        self.balances_1_file_path = os.path.join(controlled_code_folder_path, "run_dir", "balances_1")
        self.eirene_coupling_file_path = os.path.join(controlled_code_folder_path, "eirene_coupling.txt")
        self.param_raptorx_file_path = os.path.join(controlled_code_folder_path, "param_raptorX.txt")
        self.jobid_file_path = os.path.join(controlled_code_folder_path, "auto_run_jobid.txt")
        self.running_info_file_path = os.path.join(controlled_code_folder_path, output_folder_name, "running_info.json")

        self.trace_data_file_path = os.path.join(controlled_code_folder_path, output_folder_name, "Trace_Data.csv")
        self.trace_feedback_file_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                     "Trace_feedback.csv")
        self.convergence_checker_file_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                          "DP_convergence_checker.csv")

        self.gas_puff_map_file_path = os.path.join(controlled_code_folder_path, "auto_run_gas_puff_map.csv")
        self.input_power_map_file_path = os.path.join(controlled_code_folder_path, "input_power_map.csv")
        self.target_value_map_file_path = os.path.join(controlled_code_folder_path, "target_value_map.csv")
        self.profile_feedback_file_path = os.path.join(controlled_code_folder_path, "profile_feedback.csv")
        self.profile_feedback_list_r_file_path = os.path.join(controlled_code_folder_path, "DP_list_r.csv")
        self.profile_feedback_list_d_file_path = os.path.join(controlled_code_folder_path, "DP_list_D.csv")
        self.profile_feedback_list_chie_file_path = os.path.join(controlled_code_folder_path, "DP_list_chie.csv")

        self.trace_ki_error_general_file_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                             "DP_Trace_ki_error_para.csv")
        self.diff_bg_general_file_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                      "DP_diff_bg_error_para.csv")
        self.trace_diff_general_file_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                         "DP_Trace_diff_para.csv")
        self.position_index_folder_path = os.path.join(controlled_code_folder_path, output_folder_name,
                                                       "DP_Position_index")


# paths = PathSummary(controlled_code_folder_path)


# control_setup = load_module(paths.control_setup_file_path)
#
# control_setup.cache_limit


def read_active_puff(paths):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    puff_rate = float(df_DP_Trace_Data['puff_rate'].iloc[-1])
    return puff_rate


def read_latest_data(data_name, paths):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)

    if data_name in list(df_DP_Trace_Data):
        data_value = float(df_DP_Trace_Data[data_name].iloc[-1])
    else:
        data_value = 'NaN'
    return data_value


def read_puff_rate(paths):
    List_puff_rate = read_modify_variable_in_eirene_file(paths.eirene_coupling_file_path, "puff_rate")
    # List_puff_rate = find_data(paths.eirene_coupling_file_path, 'puff_rate =')
    List_puff_rate2 = List_puff_rate.split(',')

    puff_list_dict = {}
    number_of_puff = len(List_puff_rate2)

    for ind in range(number_of_puff):
        name = 'puff_rate' + str(int(ind + 1))
        puff_list_dict[name] = float(List_puff_rate2[ind])

    return puff_list_dict, number_of_puff


def read_input_power(paths):
    List_input_power = read_modify_variable_in_para_file(paths.param_raptorx_file_path, "TBC")
    # List_puff_rate = find_data(paths.eirene_coupling_file_path, 'puff_rate =')
    List_input_power2 = List_input_power.split(',')

    pin_list_dict = {}
    number_of_pin = len(List_input_power2)

    for ind in range(number_of_pin):
        name = 'pin' + str(int(ind + 1))
        pin_list_dict[name] = float(List_input_power2[ind])

    return pin_list_dict, number_of_pin


def add_item_in_h5(input):
    try:
        file_path = input['file_path']
        f = h5py.File(file_path, 'a')
        for key_name in input.keys():
            if not key_name == 'file_path':
                f[key_name] = [input[key_name]]
        f.close()
    except Exception as error:
        print(f"catch error: {error} in DP_Save_plasmaFinal")
        print(error.args)
        print('===========================================')
        print(traceback.format_exc())
        traceback.print_exc(file=open('DP_error.txt', 'a'))


def limit_cache(paths):
    print('#################### DP_cache_limit.py ####################')
    print('Start processing ...')
    control_setup = load_module(paths.control_setup_file_path)

    if os.path.exists(paths.save_plasma_final_folder_path):
        if control_setup.cache_limit >= 0:
            file_name_list = os.listdir(paths.save_plasma_final_folder_path)
            if len(file_name_list) > control_setup.cache_limit:
                number_list = []
                for file_name in file_name_list:
                    file_name = file_name.replace('.h5', '')
                    file_number = file_name.replace('plasmaFinal_', '')
                    number_list.append(int(file_number))

                number_list.sort(key=int)
                number_delete_list = number_list[:-control_setup.cache_limit or None]
                print(number_delete_list)
                for number_delete in number_delete_list:
                    plasma_final_file_path = os.path.join(paths.save_plasma_final_folder_path,
                                                          f"plasmaFinal_{str(int(number_delete))}.h5")
                    run_command(paths.save_plasma_final_folder_path, f"rm -r {plasma_final_file_path}")
                print('Files removed ...')
            else:
                print('No need to remove files...')
        else:
            print('No limitation for cache file')
    else:
        print('DP_Collection_plasmaFinal is not exist, skip limit scan')

    print('>>>>>>Finish<<<<<<')


def save_plasma_file(paths):
    print('#################### DP_Save_plasmaFinal.py ####################')
    print('Start processing ...')

    control_setup = load_module(paths.control_setup_file_path)

    if check_file_exists(paths.trace_data_file_path):
        df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
        iteration_number = df_DP_Trace_Data['iteration_number'].iloc[-1]

        if int(iteration_number) % control_setup.num_interval == 0:
            os.makedirs(paths.save_plasma_final_folder_path, exist_ok=True)
            saved_file_path = os.path.join(paths.save_plasma_final_folder_path,
                                           f"plasmaFinal_{int(int(iteration_number) / control_setup.num_interval)}.h5")
            shutil.copyfile(paths.plasma_restart_file_path, saved_file_path)
            print('Collection of plasmaFinal.h5: Success')

            puff_rate_active = read_active_puff(paths)
            total_input_power = read_latest_data('total_input_power', paths)
            Pin_e = read_latest_data('Pin_e', paths)
            Pin_i = read_latest_data('Pin_i', paths)

            puff_list_dict_not_used, number_of_puff = read_puff_rate(paths)

            input_in_h5 = {'file_path': saved_file_path, 'gas_puff_rate': puff_rate_active,
                           'total_input_power': total_input_power, 'DP_Pin_e': Pin_e, 'DP_Pin_i': Pin_i}

            for ind_puff in range(number_of_puff):
                name = 'puff_rate' + str(int(ind_puff + 1))
                input_in_h5[name] = read_latest_data(name, paths)

            add_item_in_h5(input_in_h5)

            print('Adding puff rate: Success')


    else:
        restart = int(read_modify_variable_in_para_file(paths.param_raptorx_file_path, "restart"))
        if restart == 1:
            print('Save the file at time 0!')
            create_folder_path(paths.save_plasma_final_folder_path)
            shutil.copyfile(paths.plasma_restart_file_path,
                            os.path.join(paths.save_plasma_final_folder_path, "plasmaFinal_0.h5"))
            print('Collection of plasmaFinal.h5: Success')

        elif restart == 0:
            print('No file can be saved at time 0!')
            print('Collection of plasmaFinal.h5: Pass')
        else:
            raise Exception('restart error !!!')

    print('Processing complete')
    limit_cache(paths)


def read_modify_variable_in_para_file(file_path, variable_name, new_value="_do_not_define_wJxKfG_"):
    pattern = re.compile(rf'(\s*{variable_name}\s*=\s*)([^!]*)(!.*)?')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            prefix, old_value, comment = match.groups()
            old_value_stripped = old_value.strip()

            # Check if new_value is the unique sentinel object
            if new_value == "_do_not_define_wJxKfG_":
                return old_value_stripped

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


def read_modify_variable_in_eirene_file(file_path, variable_name, new_value="_do_not_define_wJxKfG_"):
    pattern = re.compile(rf'(\s*{variable_name}\s*=\s*)(.*)')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            prefix, old_value = match.groups()
            old_value_stripped = old_value.strip()

            if new_value == "_do_not_define_wJxKfG_":
                return old_value_stripped

            new_value_formatted = str(new_value)

            if old_value_stripped == new_value_formatted:
                return

            leading_spaces = len(old_value) - len(old_value.lstrip())
            trailing_spaces = len(old_value) - len(old_value.rstrip())
            new_value_formatted_full = ' ' * leading_spaces + new_value_formatted + ' ' * trailing_spaces

            lines[i] = f'{prefix}{new_value_formatted_full}\n'

            with open(file_path, 'w') as file:
                file.writelines(lines)

            print(f"Value of {variable_name} in {file_path}: Updated successfully.")
            return

    print(f"Variable '{variable_name}' in {file_path}: Not found.")
    return None


def Get_time(paths):
    File_Plasma_Final = h5py.File(paths.plasma_final_file_path, 'r')
    time_simu = File_Plasma_Final['/tsimu'][0] * common_variable('tau0', paths)
    File_Plasma_Final.close()
    return time_simu


def get_nions(paths):
    File_Plasma_Final = h5py.File(paths.plasma_final_file_path, 'r')
    nspecies = File_Plasma_Final['/plasma_composition/Nions'][0]
    File_Plasma_Final.close()
    return int(nspecies)


def get_nelts(paths):
    File_Plasma_Final = h5py.File(paths.plasma_final_file_path, 'r')
    nspecies = File_Plasma_Final['/plasma_composition/Nelts'][0]
    File_Plasma_Final.close()
    return int(nspecies)


def common_variable(variable_name, paths):
    var_dict = {}
    var_in_mesh = (
        'r', 'z', 'psi', 'psi_min', 'psicore', 'nsep', 'Rwall', 'Zwall', 'br_mag', 'bz_mag', 'bphi_mag', 'PSIsep',
        'PSIsep_out', 'NPSIsep', 'NPSIsep_out')
    var_in_refParam_raptorX = ('n0', 'T0', 'c0', 'rho0', 'tau0')
    var_in_mesh_raptorX = ('nzones', 'rho0_mesh')
    var_in_meshEIRENE = ('Rknots', 'Zknots', 'triKnots', 'triwall', 'triface', 'trisequence', 'Rtri', 'Ztri')
    var_others = ('small_gamma', 'constant_e')

    if variable_name in var_in_mesh:
        file_mesh = h5py.File(paths.mesh_file_path, 'r')
        var_dict['r'] = file_mesh['/config/r'][:]
        var_dict['z'] = file_mesh['/config/z'][:]
        var_dict['psi'] = file_mesh['/config/psi'][:]

        psi = var_dict['psi']
        var_dict['psi_min'] = psi.min()
        var_dict['psicore'] = file_mesh['/config/psicore'][0]
        var_dict['nsep'] = file_mesh['/config/nsep'][0]
        var_dict['Rwall'] = file_mesh['/walls/wall1/R'][:]
        var_dict['Zwall'] = file_mesh['/walls/wall1/Z'][:]

        var_dict['br_mag'] = file_mesh['/config/Br'][:]
        var_dict['bz_mag'] = file_mesh['/config/Bz'][:]
        var_dict['bphi_mag'] = file_mesh['/config/Bphi'][:]

        nsep = var_dict['nsep']
        psisep = [0] * nsep
        for i in range(nsep):
            psisep[i] = file_mesh['/config/psisep' + str(i + 1)][0]

        var_dict['PSIsep'] = min(psisep)
        # AA = var_dict['PSIsep']
        # print(AA)
        var_dict['NPSIsep'] = (var_dict['PSIsep'] - var_dict['psi_min']) / (var_dict['PSIsep'] - var_dict['psi_min'])

        # var_dict['PSIsep'] = psisep[0]
        if nsep > 1:
            var_dict['PSIsep_out'] = max(psisep)
            var_dict['NPSIsep_out'] = (var_dict['PSIsep_out'] - var_dict['psi_min']) / (
                    var_dict['PSIsep'] - var_dict['psi_min'])
            # var_dict['PSIsep_out'] = psisep[1]

        file_mesh.close()

    elif variable_name in var_in_refParam_raptorX:
        file_refParam = h5py.File(paths.refparm_raptorx_file_path, 'r')
        var_dict['n0'] = file_refParam['/n0'][0]
        var_dict['T0'] = file_refParam['/T0'][0]
        var_dict['c0'] = file_refParam['/c0'][0]
        var_dict['rho0'] = file_refParam['/rho0'][0]
        var_dict['tau0'] = file_refParam['/tau0'][0]
        file_refParam.close()

    elif variable_name in var_in_mesh_raptorX:

        file_mesh_raptorx = h5py.File(paths.mesh_raptorx_file_path, 'r')
        var_dict['nzones'] = file_mesh_raptorx['/NZones'][0]
        var_dict['rho0_mesh'] = file_mesh_raptorx['/rho0'][0]
        file_mesh_raptorx.close()

    elif variable_name in var_in_meshEIRENE:
        file_meshEIRENE = h5py.File(paths.mesh_eirene_file_path, 'r')

        var_dict['Rknots'] = file_meshEIRENE['/knots/R'][:] / 100
        var_dict['Zknots'] = file_meshEIRENE['/knots/Z'][:] / 100
        var_dict['triKnots'] = file_meshEIRENE['/triangles/tri_knots'][:]
        var_dict['triwall'] = file_meshEIRENE['/wall/triNum'][:]
        var_dict['triface'] = file_meshEIRENE['/wall/triFace'][:]
        var_dict['trisequence'] = file_meshEIRENE['/wall/triSequence1/tri'][:]

        triKnots = var_dict['triKnots']
        Rknots = var_dict['Rknots']
        Zknots = var_dict['Zknots']
        dim_triKnots = triKnots.shape
        Rtri = np.zeros(dim_triKnots)
        Ztri = np.zeros(dim_triKnots)
        for k in range(dim_triKnots[0]):
            for j in range(dim_triKnots[1]):
                Rtri[k, j] = Rknots[triKnots[k, j] - 1]
                Ztri[k, j] = Zknots[triKnots[k, j] - 1]
        var_dict['Rtri'] = Rtri
        var_dict['Ztri'] = Ztri
        file_meshEIRENE.close()

    elif variable_name in var_others:
        var_dict['small_gamma'] = 7.5
        var_dict['constant_e'] = 1.602176634e-19

    else:
        raise Exception('No variable_name information !!!')

    return var_dict[variable_name]


def read_detail_input_power(paths):
    # judge boundary condition
    # TBCtype_list = get_param_data('TBCtype = ', 'Type of boundary conditions for temperature')

    # df_Pin0 = pd.read_table(paths.balances_0_file_path, delim_whitespace=True)
    # DP_Pin_e = df_Pin0['fluxEIn'].iloc[-1]

    # df_Pin1 = pd.read_table(paths.balances_1_file_path, delim_whitespace=True)
    # DP_Pin_i = df_Pin1['fluxEIn'].iloc[-1]

    df_Pin0 = pd.read_table(paths.balances_0_file_path, sep=r'\s+', on_bad_lines='skip')
    DP_Pin_e = df_Pin0['fluxEIn'].iloc[-1]

    df_Pin1 = pd.read_table(paths.balances_1_file_path, sep=r'\s+', on_bad_lines='skip')
    DP_Pin_i = df_Pin1['fluxEIn'].iloc[-1]

    return float(DP_Pin_e), float(DP_Pin_i)


def distance(r, z):
    R1 = 2.0034
    R3 = 2.3616
    Z1 = -0.6178
    Z3 = -0.7622
    AA = (Z3 - Z1) / (R3 - R1)
    BB = -1
    CC = Z1 - R1 * AA

    vertical_distance = abs(AA * r + BB * z + CC) / ((AA ** 2 + BB ** 2) ** 0.5)
    return vertical_distance


def peak_value_in_field(para_name, matrix_type, paths):
    # input_matrix_para_define = {'para_name': str(para_name), 'matrix_type': str(matrix_type)}
    para_info = PT_fun_matrix_general(para_name=str(para_name), matrix_type=str(matrix_type), paths=paths)

    row_averages = np.mean(para_info.value, axis=1, keepdims=True)

    max_index = np.nanargmax(row_averages)

    matrix_r = para_info.r
    row_data_r = matrix_r[max_index]
    mean_r = np.mean(row_data_r)

    matrix_z = para_info.z
    row_data_z = matrix_z[max_index]
    mean_z = np.mean(row_data_z)

    matrix_para = para_info.value
    row_data_para = matrix_para[max_index]
    mean_para = np.mean(row_data_para)

    return Point(r=mean_r, z=mean_z, value=mean_para)


def radiator_distance(paths):
    output = peak_value_in_field(para_name='Prad_total', matrix_type='tri', paths=paths)
    return distance(output.r, output.z)


def PT_fun_positions(input, paths):
    control_setup = load_module(paths.control_setup_file_path)
    position_name = input['position_name']
    N_point = 500
    R_extend = []
    Z_extend = []
    if control_setup.Trace_mode == 'WEST':

        R1 = 1.9185
        R2 = 2.1919
        R3 = 2.443147993067367
        Z1 = -0.5836
        Z2 = -0.6938
        Z3 = -0.795232879591286

        R_DIVL = np.linspace(R1, R3, N_point)
        Z_DIVL = np.linspace(Z1, Z3, N_point)

        # for upper target
        Ra = 1.9014
        Rc = 2.1853
        Rb = 2.4446
        Za = 0.5792
        Zc = 0.6939
        Zb = 0.7986

        if position_name == 'OMP' or position_name == 'OMPS':
            position_center = np.where(common_variable('psi', paths) == common_variable('psi_min', paths))
            Zlin = common_variable('z', paths)[0, (position_center[1])[0]]
            Ri1 = np.linspace(min(common_variable('Rwall', paths)), max(common_variable('Rwall', paths)), 1000)
            Zi1 = np.ones(1000) * Zlin

            psi1 = interp2linear(common_variable('r', paths), common_variable('z', paths),
                                 common_variable('psi', paths), Ri1, Zi1)
            # psi_contour = RectBivariateSpline(common_variable('r[:, 1], common_variable('z[1],
            #                                   common_variable('psi)
            # psi1 = psi_contour(Ri1, Zi1)[:, 0]

            for ind in range(1000, 1, -1):
                if psi1[ind - 1] < common_variable('psicore', paths):
                    break

            if position_name == 'OMP':
                position_type = 'LINE'
                Ri_start = Ri1[ind]
                Ri_end = max(common_variable('Rwall', paths))
                Zi_start = Zlin
                Zi_end = Zlin
                R = np.linspace(Ri_start, Ri_end, N_point)
                Z = np.linspace(Zi_start, Zi_end, N_point)
            elif position_name == 'OMPS':
                position_type = 'POINT'

                Ri = np.linspace(Ri1[ind], max(common_variable('Rwall', paths)), 2000)
                Zi = np.ones(2000) * Zlin
                psii = interp2linear(common_variable('r', paths), common_variable('z', paths),
                                     common_variable('psi', paths), Ri, Zi)

                # psii = psi_contour(Ri, Zi)[:, 0]
                Rsep = np.interp(common_variable('PSIsep', paths), psii, Ri)
                R = np.array([Rsep])
                Z = np.array([Zlin])

                input_positions = {'position_name': 'OMP'}
                output_positions = PT_fun_positions(input_positions, paths)
                R_extend = output_positions['list_r']
                Z_extend = output_positions['list_z']

            else:
                raise Exception('position_name wrong !!!')

        elif position_name == 'outerOMPS':
            position_type = 'LINE'
            input_positions_omps = {'position_name': 'OMPS'}
            list_r_omps = PT_fun_positions(input_positions_omps, paths)['list_r']
            list_z_omps = PT_fun_positions(input_positions_omps, paths)['list_z']

            input_positions_omps = {'position_name': 'OMP'}
            list_r_omp = PT_fun_positions(input_positions_omps, paths)['list_r']
            R = np.linspace(list_r_omps, list_r_omp.max(), N_point)
            Z = np.linspace(list_z_omps, list_z_omps, N_point)

        elif position_name == 'LIT':
            position_type = 'LINE'
            Ri_start = R1
            Ri_end = R2
            Zi_start = Z1
            Zi_end = Z2
            R = np.linspace(Ri_start, Ri_end, N_point)
            Z = np.linspace(Zi_start, Zi_end, N_point)
        elif position_name == 'LOT':
            position_type = 'LINE'
            Ri_start = R2
            Ri_end = R3
            Zi_start = Z2
            Zi_end = Z3
            R = np.linspace(Ri_start, Ri_end, N_point)
            Z = np.linspace(Zi_start, Zi_end, N_point)
        elif position_name == 'UIT':
            position_type = 'LINE'
            Ri_start = Rc
            Ri_end = Ra
            Zi_start = Zc
            Zi_end = Za
            R = np.linspace(Ri_start, Ri_end, N_point)
            Z = np.linspace(Zi_start, Zi_end, N_point)
        elif position_name == 'UOT':
            position_type = 'LINE'
            Ri_start = Rc
            Ri_end = Rb
            Zi_start = Zc
            Zi_end = Zb
            R = np.linspace(Ri_start, Ri_end, N_point)
            Z = np.linspace(Zi_start, Zi_end, N_point)
        elif position_name == 'LISP':  # inner strike point
            position_type = 'POINT'
            R_sep_inner = 2.135164624933934
            R = [R_sep_inner]
            Z = np.interp(R, R_DIVL, Z_DIVL)
        elif position_name == 'LOSP':  # outer strike point
            position_type = 'POINT'
            R_sep_outer = 2.241377118263560
            R = [R_sep_outer]
            Z = np.interp(R, R_DIVL, Z_DIVL)
        elif position_name == 'LLP01':  # second point on outer target
            position_type = 'POINT'
            R = [2.253]
            Z = np.interp(R, R_DIVL, Z_DIVL)
        elif position_name == 'LLP02':  # third point on outer target
            position_type = 'POINT'
            R = [2.266]
            Z = np.interp(R, R_DIVL, Z_DIVL)
        else:
            raise Exception('No position information in WEST !!!')

    elif control_setup.Trace_mode == 'PRO':

        if 'point' in position_name and get_variable_from_module(control_setup, f"{position_name}_name"):
            position_type = 'POINT'
            R = get_variable_from_module(control_setup, f"{position_name}_r")
            Z = get_variable_from_module(control_setup, f"{position_name}_z")
            R_extend = np.linspace(R - 0.015, R + 0.015, N_point)
            Z_extend = np.linspace(Z, Z, N_point)
        elif 'line' in position_name and get_variable_from_module(control_setup, f"{position_name}_name"):
            position_type = 'LINE'
            Ri_start = get_variable_from_module(control_setup, f"{position_name}_r_start")
            Zi_start = get_variable_from_module(control_setup, f"{position_name}_z_start")
            Ri_end = get_variable_from_module(control_setup, f"{position_name}_r_end")
            Zi_end = get_variable_from_module(control_setup, f"{position_name}_z_end")
            N_point = get_variable_from_module(control_setup, f"{position_name}_num")
            R = np.linspace(Ri_start, Ri_end, N_point)
            Z = np.linspace(Zi_start, Zi_end, N_point)
        else:
            raise Exception('No position information for PRO !!!')
    else:
        raise Exception('Trace_mode information wrong !!!')

    output = {'list_r': R, 'list_z': Z, 'position_type': position_type}
    if len(R_extend):
        output['R_extend'] = R_extend
        output['Z_extend'] = Z_extend
    return output


def PT_fun_matrix_para(input, paths):
    para_path = input['para_path']
    para_scalling_factor = input['para_scalling_factor']
    matrix_type = input['matrix_type']
    File_Plasma_Final = h5py.File(paths.plasma_final_file_path, 'r')

    if matrix_type == 'tri':
        para_value_normalised = File_Plasma_Final['/triangles' + para_path]
        # print(type(para_scalling_factor))
        # print(type(para_value_normalised))
        para_value = para_value_normalised * para_scalling_factor
        # print(type(para_value))
        matrix_para = np.tile(para_value, (3, 1))
        matrix_para = matrix_para.T
    elif matrix_type == 'quad':
        X_quad_all = np.zeros(shape=(1, 4))
        for k in range(common_variable('nzones', paths)):
            index_zone = k + 1
            zone_data = File_Plasma_Final['/zone' + str(index_zone) + para_path]
            zone_data_used = zone_data[2:-2, 2:-2]
            X_vector = zone_data_used.flatten()
            X_quad = np.tile(X_vector, (4, 1))
            X_quad = X_quad.T
            X_quad_all = np.append(X_quad_all, X_quad, axis=0)

        X_quad_all = np.delete(X_quad_all, 0, axis=0)
        matrix_para = X_quad_all * para_scalling_factor
    else:
        raise Exception('matrix_type not found !!!')
    File_Plasma_Final.close()
    output = {'matrix_para': matrix_para}
    return output


def get_s_int(long, short):
    number_str = long.replace(short, '')
    return number_str


def read_SymbList(paths):
    line_needed = find_param_data(paths.param_raptorx_file_path, 'SymbList =',
                                  'Symbols of chemical elements, inside quotes and delimited by comas')
    head, sep, tail = line_needed.partition('!')
    head, sep, tail = head.partition('=')
    tail = tail.replace("'", "")
    tail = tail.replace(' ', '')

    return tail


def PT_fun_matrix_para_define(input, paths):
    para_name = input['para_name']
    matrix_type = input['matrix_type']
    if para_name == 'ne' or para_name == 'ni':
        if para_name == 'ne':
            para_path = '/spec0/n'
        elif para_name == 'ni':
            para_path = '/spec1/n'
        else:
            raise Exception('matrix_type not found !!!')

        para_scalling_factor = common_variable('n0', paths)
        input_matrix_para = {'matrix_type': matrix_type, 'para_path': para_path,
                             'para_scalling_factor': para_scalling_factor}
        output_matrix_para = PT_fun_matrix_para(input_matrix_para, paths)
        output = {'matrix_para': output_matrix_para['matrix_para']}

    elif para_name == 'Te' or para_name == 'Ti':
        if para_name == 'Te':
            para_path = '/spec0/T'
        elif para_name == 'Ti':
            para_path = '/spec1/T'
        else:
            raise Exception('matrix_type not found !!!')

        para_scalling_factor = common_variable('T0', paths)
        input_matrix_para = {'matrix_type': matrix_type, 'para_path': para_path,
                             'para_scalling_factor': para_scalling_factor}
        output_matrix_para = PT_fun_matrix_para(input_matrix_para, paths)
        output = {'matrix_para': output_matrix_para['matrix_para']}

    elif any(key in para_name for key in ['Prad_s', 'Prad_N_At_s', 'Prad_N_Mol_s', 'Prad_N_Pls_s', 'Prad_N_Ti_s']):
        n0 = common_variable('n0', paths)
        T0 = common_variable('T0', paths)
        constant_e = common_variable('constant_e', paths)
        tau0 = common_variable('tau0', paths)

        if matrix_type == 'tri':
            para_scalling_factor = n0 * T0 * constant_e / tau0
            path_templates = {
                'Prad_s': '/spec{}/PradPlasma',
                'Prad_N_At_s': '/Nspec{}/PradAt',
                'Prad_N_Mol_s': '/Nspec{}/PradMol',
                'Prad_N_Pls_s': '/Nspec{}/PradPls',
                'Prad_N_Ti_s': '/Nspec{}/PradTi',
            }

            # Find the matching key and construct the para_path
            for key, template in path_templates.items():
                if key in para_name:
                    snumber_str = get_s_int(para_name, key)
                    para_path = template.format(snumber_str)
                    break
            else:
                raise Exception('para_name not found !!!')

            input_matrix_para = {'matrix_type': matrix_type, 'para_path': para_path,
                                 'para_scalling_factor': para_scalling_factor}
            output_matrix_para = PT_fun_matrix_para(input_matrix_para, paths)
            output = {'matrix_para': np.absolute(output_matrix_para['matrix_para'])}

        elif matrix_type == 'quad':
            if 'Prad_s' in para_name:
                para_scalling_factor = n0 * T0 * constant_e / tau0
                path_templates = {
                    'Prad_s': '/spec{}/radiation',
                }
                # Find the matching key and construct the para_path
                for key, template in path_templates.items():
                    if key in para_name:
                        snumber_str = get_s_int(para_name, key)
                        para_path = template.format(snumber_str)
                        break
                else:
                    raise Exception('para_name not found !!!')
                input_matrix_para = {'matrix_type': matrix_type, 'para_path': para_path,
                                     'para_scalling_factor': para_scalling_factor}
                output_matrix_para = PT_fun_matrix_para(input_matrix_para, paths)
                output = {'matrix_para': np.absolute(output_matrix_para['matrix_para'])}
            else:
                output_matrix_quad_rzchi = PT_fun_matrix_quad_rzchi(paths)

                # input_matrix_para_define = {'para_name': str(para_name), 'matrix_type': 'tri'}
                para_tri_all = PT_fun_matrix_general(para_name=str(para_name), matrix_type='tri', paths=paths)

                para_quad = griddata_classic(para_tri_all.r, para_tri_all.z,
                                             para_tri_all.value, output_matrix_quad_rzchi['matrix_r'],
                                             output_matrix_quad_rzchi['matrix_z'], method='linear')

                output = {'matrix_para': np.absolute(para_quad)}

        else:
            raise Exception('matrix_type not found !!!')

    elif para_name == "Prad_total":
        nelts = get_nelts(paths)
        nions = get_nions(paths)

        prad_value_total = 0
        # for s_number in range(1, nelts + 1):
        #     for key in ['Prad_N_At_s', 'Prad_N_Mol_s', 'Prad_N_Pls_s', 'Prad_N_Ti_s']:
        #         para_value_all = PT_fun_matrix_general(para_name=f"{key}{s_number}", matrix_type=matrix_type,
        #                                                paths=paths)
        #         prad_value_total = prad_value_total + para_value_all.value

        for s_number in range(1, nelts + 1):
            for key in ['Prad_N_At_s', 'Prad_N_Mol_s', 'Prad_N_Pls_s', 'Prad_N_Ti_s']:
                try:
                    para_value_all = PT_fun_matrix_general(para_name=f"{key}{s_number}", matrix_type=matrix_type,
                                                           paths=paths)
                    prad_value_total += para_value_all.value
                except Exception as e:
                    print(f"Failed to read {key}{s_number}: {e}")
                    continue

        for s_number in range(1, nions + 1):
            for key in ['Prad_s']:
                para_value_all = PT_fun_matrix_general(para_name=f"{key}{s_number}", matrix_type=matrix_type,
                                                       paths=paths)
                prad_value_total = prad_value_total + para_value_all.value

        output = {'matrix_para': prad_value_total}

    else:
        raise Exception('para_name not found !!!')

    return output


def PT_fun_matrix_quad_rzchi(paths):
    File_mesh_raptorx = h5py.File(paths.mesh_raptorx_file_path, 'r')
    rho0 = common_variable('rho0_mesh', paths)
    chiM_quad_all = np.zeros(shape=(1, 4))
    R_quad_all = np.zeros(shape=(1, 4))
    Z_quad_all = np.zeros(shape=(1, 4))
    for k in range(common_variable('nzones', paths)):
        index_zone = k + 1
        Rcorner = File_mesh_raptorx['/zone' + str(index_zone) + '/Rcorners'][:, :, 0] * rho0
        Zcorner = File_mesh_raptorx['/zone' + str(index_zone) + '/Zcorners'][:, :, 0] * rho0
        chi = File_mesh_raptorx['/zone' + str(index_zone) + '/chi'][:, :, 0]
        (m, p) = chi.shape
        chiNaN = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                if chi[i, j] == 1:
                    chiNaN[i, j] = 'nan'

        chiM = chiNaN[2:-2, 2:-2]
        chiM_matrix = np.squeeze(chiM)
        chiM_vector = chiM_matrix.flatten()
        chiM_quad = np.tile(chiM_vector, (4, 1))
        chiM_quad = chiM_quad.T

        chiM_quad_all = np.append(chiM_quad_all, chiM_quad, axis=0)

        R1_matrix = Rcorner[0:-1, 0:-1]
        R2_matrix = Rcorner[1:, 0:-1]
        R3_matrix = Rcorner[1:, 1:]
        R4_matrix = Rcorner[0:-1, 1:]

        Z1_matrix = Zcorner[0:-1, 0:-1]
        Z2_matrix = Zcorner[1:, 0:-1]
        Z3_matrix = Zcorner[1:, 1:]
        Z4_matrix = Zcorner[0:-1, 1:]

        R1_vector = R1_matrix.flatten()
        R2_vector = R2_matrix.flatten()
        R3_vector = R3_matrix.flatten()
        R4_vector = R4_matrix.flatten()

        Z1_vector = Z1_matrix.flatten()
        Z2_vector = Z2_matrix.flatten()
        Z3_vector = Z3_matrix.flatten()
        Z4_vector = Z4_matrix.flatten()

        R_quad = np.vstack((R1_vector, R4_vector, R3_vector, R2_vector))
        Z_quad = np.vstack((Z1_vector, Z4_vector, Z3_vector, Z2_vector))

        R_quad_all = np.append(R_quad_all, R_quad.T, axis=0)
        Z_quad_all = np.append(Z_quad_all, Z_quad.T, axis=0)

    File_mesh_raptorx.close()
    R_quad_all = np.delete(R_quad_all, 0, axis=0)
    Z_quad_all = np.delete(Z_quad_all, 0, axis=0)
    chiM_quad_all = np.delete(chiM_quad_all, 0, axis=0)
    output = {'matrix_r': R_quad_all, 'matrix_z': Z_quad_all, 'matrix_chi': chiM_quad_all}

    return output


def PT_fun_matrix_general(para_name, matrix_type, paths):
    # para_name = input['para_name']
    # matrix_type = input['matrix_type']

    input_matrix_para_define = {'para_name': para_name, 'matrix_type': matrix_type}
    output_matrix_para_define = PT_fun_matrix_para_define(input_matrix_para_define, paths)
    if matrix_type == 'tri':
        matrix_r = common_variable('Rtri', paths)
        matrix_z = common_variable('Ztri', paths)
        matrix_r = matrix_r.T
        matrix_z = matrix_z.T
        matrix_para = output_matrix_para_define['matrix_para']

    elif matrix_type == 'quad':
        output_matrix_quad_rzchi = PT_fun_matrix_quad_rzchi(paths)
        matrix_r = output_matrix_quad_rzchi['matrix_r']
        matrix_z = output_matrix_quad_rzchi['matrix_z']
        matrix_chi = output_matrix_quad_rzchi['matrix_chi']
        matrix_para = output_matrix_para_define['matrix_para'] + matrix_chi
    else:
        raise Exception('matrix_type not found !!!')
    # output = {'matrix_r': matrix_r, 'matrix_z': matrix_z, 'matrix_para': matrix_para}hh

    return Matrix(r=matrix_r, z=matrix_z, value=matrix_para)
    # return output


def read_and_save_index(input, paths):
    index_label = input['index_label']
    file_path = os.path.join(paths.position_index_folder_path, str(f'{index_label}.csv'))

    if input['read_or_save'] == 'save':
        if not os.path.exists(paths.position_index_folder_path):
            os.makedirs(paths.position_index_folder_path)

        index_value = input['index_value']
        df = pd.DataFrame({'index_value': index_value})
        df.to_csv(file_path, mode='w', index=None)

    elif input['read_or_save'] == 'read':
        df = pd.read_csv(file_path)
        index_value = df['index_value'].values
        return index_value

    elif input['read_or_save'] == 'exist':
        if os.path.exists(file_path):
            return True
        else:
            return False
    else:

        raise Exception('wrong read_or_save information !!!')


def PT_fun_index_curve_intersect_polygeons(input):
    matrix_r = input['matrix_r']
    matrix_z = input['matrix_z']
    list_r = input['list_r']
    list_z = input['list_z']

    in_mark = []
    for index in range(len(matrix_r)):
        # index = k + 1
        r_poly = matrix_r[index, :].T
        z_poly = matrix_z[index, :].T
        r_poly = np.append(r_poly, r_poly[0])
        z_poly = np.append(z_poly, z_poly[0])
        # r_poly = r_poly.append(r_poly[0])
        # z_poly = z_poly.append(z_poly[0])
        x, y = intersection(list_r, list_z, r_poly, z_poly)
        if len(x):
            in_mark = np.append(in_mark, 0)
        else:
            in_mark = np.append(in_mark, 1)
    index_used = np.where(in_mark == 0)[0]
    output = {'index_used': index_used}
    return output


def PT_fun_curve_intersect_polygeons(input, paths):
    index_label = input['index_label']
    para_name = input['para_name']
    matrix_type = input['matrix_type']
    # input_matrix_general = {'para_name': para_name, 'matrix_type': matrix_type}
    output_matrix_general = PT_fun_matrix_general(para_name=para_name, matrix_type=matrix_type, paths=paths)

    if read_and_save_index({'read_or_save': 'exist', 'index_label': index_label}, paths):
        index_used = read_and_save_index({'read_or_save': 'read', 'index_label': index_label}, paths)
    else:

        list_r = input['list_r']
        list_z = input['list_z']

        input_intersect = {'matrix_r': output_matrix_general.r,
                           'matrix_z': output_matrix_general.z,
                           'list_r': list_r, 'list_z': list_z}
        output_intersect = PT_fun_index_curve_intersect_polygeons(input_intersect)
        index_used = output_intersect['index_used']
        read_and_save_index({'read_or_save': 'save', 'index_value': index_used, 'index_label': index_label}, paths)

    matrix_r = output_matrix_general.r[index_used, :]
    matrix_z = output_matrix_general.z[index_used, :]
    matrix_para = output_matrix_general.value[index_used, :]

    colum1 = np.mean(matrix_r, axis=1)
    colum2 = np.mean(matrix_r, axis=1)
    colum3 = np.mean(matrix_z, axis=1)
    colum4 = np.mean(matrix_para, axis=1)

    matrix = np.vstack((colum1, colum2, colum3, colum4))
    matrix = matrix.T
    matrix_sort = matrix[np.argsort(matrix[:, 0])]

    list_r = matrix_sort[:, 1]
    list_z = matrix_sort[:, 2]
    list_para = matrix_sort[:, 3]

    output = {'list_r': list_r, 'list_z': list_z, 'list_para': list_para}
    return output


def PT_fun_wall_general(input, paths):
    para_name = input['para_name']
    rknots = common_variable('Rknots', paths)
    zknots = common_variable('Zknots', paths)
    triknots = common_variable('triKnots', paths)
    triwall = common_variable('triwall', paths)
    triface = common_variable('triface', paths)
    trisequence = common_variable('trisequence', paths)

    R_list = []
    Z_list = []
    surf_list = []

    for index in range(len(triface)):
        value_triface = triface[index]
        if value_triface == 1:
            R1 = rknots[triknots[0, triwall[index] - 1] - 1]
            Z1 = zknots[triknots[0, triwall[index] - 1] - 1]
            R2 = rknots[triknots[1, triwall[index] - 1] - 1]
            Z2 = zknots[triknots[1, triwall[index] - 1] - 1]
        elif value_triface == 2:
            R1 = rknots[triknots[1, triwall[index] - 1] - 1]
            Z1 = zknots[triknots[1, triwall[index] - 1] - 1]
            R2 = rknots[triknots[2, triwall[index] - 1] - 1]
            Z2 = zknots[triknots[2, triwall[index] - 1] - 1]
        else:
            R1 = rknots[triknots[2, triwall[index] - 1] - 1]
            Z1 = zknots[triknots[2, triwall[index] - 1] - 1]
            R2 = rknots[triknots[0, triwall[index] - 1] - 1]
            Z2 = zknots[triknots[0, triwall[index] - 1] - 1]

        surf = np.sqrt(np.square(R1 - R2) + np.square(Z1 - Z2)) * np.pi * (R1 + R2)

        R_list = np.append(R_list, R1)
        Z_list = np.append(Z_list, Z1)
        surf_list = np.append(surf_list, surf)

    wall_r = []
    wall_z = []
    wall_surf = []
    for index2 in range(len(trisequence)):
        r = R_list[trisequence[index2] - 1]
        z = Z_list[trisequence[index2] - 1]
        surf_value = surf_list[trisequence[index2] - 1]

        wall_r = np.append(wall_r, r)
        wall_z = np.append(wall_z, z)
        wall_surf = np.append(wall_surf, surf_value)

    File_Plasma_Final = h5py.File(paths.plasma_final_file_path, 'r')
    if para_name == 'ne':
        n_0 = File_Plasma_Final['/walls/wall1/spec0/n'][0]
        n_0 = n_0 * common_variable('n0', paths)
        wall_para = n_0
    elif para_name == 'ni':
        n_1 = File_Plasma_Final['/walls/wall1/spec1/n'][0]
        n_1 = n_1 * common_variable('n0', paths)
        wall_para = n_1
    elif para_name == 'Te':
        T_0 = File_Plasma_Final['/walls/wall1/spec0/T'][0]
        T_0 = T_0 * common_variable('T0', paths)
        wall_para = T_0
    elif para_name == 'Ti':
        T_1 = File_Plasma_Final['/walls/wall1/spec1/T'][0]
        T_1 = T_1 * common_variable('T0', paths)
        wall_para = T_1

    elif para_name == 'Ge':
        G_0 = File_Plasma_Final['/walls/wall1/spec0/G'][0]
        G_0 = G_0 * common_variable('n0', paths) * common_variable('c0', paths)
        wall_para = G_0

    elif para_name == 'Gi':
        G_1 = File_Plasma_Final['/walls/wall1/spec1/G'][0]
        G_1 = G_1 * common_variable('n0', paths) * common_variable('c0', paths)
        wall_para = G_1
    elif para_name == 'qperp':
        fluxE_0 = File_Plasma_Final['/walls/wall1/spec0/fluxE'][0]
        fluxE_1 = File_Plasma_Final['/walls/wall1/spec1/fluxE'][0]

        q_0 = fluxE_0 / wall_surf
        q_1 = fluxE_1 / wall_surf
        wall_para = q_0 + q_1
    else:
        raise Exception('wrong para_name information !!!')

    File_Plasma_Final.close()
    output = {'wall_para': wall_para, 'wall_r': wall_r, 'wall_z': wall_z}
    return output


def PT_fun_index_curve_intersect_wall(input, paths):
    index_label = input['index_label']

    if read_and_save_index({'read_or_save': 'exist', 'index_label': index_label}, paths):
        index_list = read_and_save_index({'read_or_save': 'read', 'index_label': index_label}, paths)

    else:
        R_wall = input['list_r_wall']
        Z_wall = input['list_z_wall']

        R_curve = input['list_r_curve']
        Z_curve = input['list_z_curve']

        R_curve = np.linspace(R_curve[0], R_curve[-1], 2000)
        Z_curve = np.linspace(Z_curve[0], Z_curve[-1], 2000)

        min_index_list = []
        for index in range(len(R_curve)):
            distance_list = np.sqrt(np.square(R_wall - R_curve[index]) + np.square(Z_wall - Z_curve[index]))

            min_index = np.where(distance_list == min(distance_list))[0]

            min_index_list = np.append(min_index_list, min_index)

        index_wall_temp = [x[0] for x in groupby(min_index_list)]

        index_wall_new = np.array(index_wall_temp)

        index_list = index_wall_new.astype(int)

        read_and_save_index({'read_or_save': 'save', 'index_value': index_list, 'index_label': index_label}, paths)

    output = {'index_wall': index_list}
    return output


def PT_fun_data(input, paths):
    para_name = input['para_name']
    matrix_type = input['matrix_type']
    position_name = input['position_name']
    location = input['location']
    data_treatment = input['data_treatment']

    input_positions = {'position_name': position_name}
    # print(input_positions)
    output_positions = PT_fun_positions(input_positions, paths)
    position_type = output_positions['position_type']

    index_label = location + '_' + matrix_type + '_' + position_name

    if location == 'INNER':
        if position_type == 'POINT':
            input_intersect = {'para_name': para_name, 'matrix_type': matrix_type,
                               'list_r': output_positions['R_extend'], 'list_z': output_positions['Z_extend'],
                               'index_label': index_label}

            output_intersect = PT_fun_curve_intersect_polygeons(input_intersect, paths)

            para_value = np.interp(output_positions['list_r'], output_intersect['list_r'],
                                   output_intersect['list_para'])
            list_para = para_value
            list_r = output_positions['list_r']
            list_z = output_positions['list_z']

        elif position_type == 'LINE':
            input_intersect = {'para_name': para_name, 'matrix_type': matrix_type,
                               'list_r': output_positions['list_r'], 'list_z': output_positions['list_z'],
                               'index_label': index_label}
            output_intersect = PT_fun_curve_intersect_polygeons(input_intersect, paths)

            list_para = output_intersect['list_para']
            list_r = output_intersect['list_r']
            list_z = output_intersect['list_z']

        else:
            raise Exception('wrong position_type information !!!')
    elif location == 'WALL':
        if para_name != 'qpara':
            input_wall_general = {'para_name': para_name}
            output_wall_general = PT_fun_wall_general(input_wall_general, paths)

            input_index_curve_intersect_wall = {'list_r_wall': output_wall_general['wall_r'],
                                                'list_z_wall': output_wall_general['wall_z'],
                                                'list_r_curve': output_positions['list_r'],
                                                'list_z_curve': output_positions['list_z'],
                                                'index_label': index_label}

            output_index_curve_intersect_wall = PT_fun_index_curve_intersect_wall(input_index_curve_intersect_wall,
                                                                                  paths)
            index_wall = output_index_curve_intersect_wall['index_wall']

            list_para = output_wall_general['wall_para'][index_wall]
            list_r = output_wall_general['wall_r'][index_wall]
            list_z = output_wall_general['wall_z'][index_wall]
        else:
            input_data = {'para_name': 'qperp', 'matrix_type': matrix_type, 'position_name': position_name,
                          'location': 'WALL', 'data_treatment': 'NONE'}

            output_data_qperp = PT_fun_data(input_data, paths)

            vec_nr = 1
            vec_nz = (output_positions['list_r'][-1] - output_positions['list_r'][0]) / (
                    output_positions['list_z'][0] - output_positions['list_z'][-1])
            vec_nphi = 0

            matrix_r = common_variable('r', paths)
            matrix_z = common_variable('z', paths)

            list_r = output_data_qperp['list_r']
            list_z = output_data_qperp['list_z']

            bphi_mag_list = interp2linear(matrix_r, matrix_z, common_variable('bphi_mag', paths), list_r, list_z)

            br_mag_list = interp2linear(matrix_r, matrix_z, common_variable('br_mag', paths), list_r, list_z)

            bz_mag_list = interp2linear(matrix_r, matrix_z, common_variable('bz_mag', paths), list_r, list_z)

            A = np.arcsin(bphi_mag_list * vec_nphi + br_mag_list * vec_nr + bz_mag_list * vec_nz)
            B = np.sqrt(np.square(bphi_mag_list) + np.square(br_mag_list) + np.square(bz_mag_list))
            C = np.sqrt(1 + np.square(vec_nz))
            angle_list = A / (B * C)

            print(angle_list)

            list_para = output_data_qperp['data_result'] / np.sin(angle_list)
    else:
        raise Exception('wrong location information !!!')

    if data_treatment == 'FLOAT':
        data_result = float(list_para)
    elif data_treatment == 'MAXABS':
        data_result = np.absolute(list_para)
        data_result = np.max(data_result)
    elif data_treatment == 'MEANABS':
        data_result = np.absolute(list_para)
        data_result = np.mean(data_result)
    elif data_treatment == 'NONE':
        data_result = list_para
    else:
        raise Exception('data_treatment error !!!')

    output = {'list_r': list_r, 'list_z': list_z, 'list_para': list_para, 'data_result': data_result}
    return output


def Prad_total_int_lot(paths):
    # input_matrix_para_define = {'para_name': 'Prad_total', 'matrix_type': 'tri'}
    output_general = PT_fun_matrix_general(para_name='Prad_total', matrix_type='tri', paths=paths)
    Prad_total_matrix_r = output_general.r
    Prad_total_matrix_z = output_general.z
    Prad_total_matrix_data = output_general.value

    control_setup = load_module(paths.control_setup_file_path)

    if control_setup.Trace_mode == 'WEST':  # 56420
        list_r = np.array([2.20716611522153, 2.35740000000000, 2.39262398812739, 2.24239010334891, 2.20716611522153])
        list_z = np.array(
            [-0.699957746729085, -0.760489358356196, -0.673037819962054, -0.612506208334943, -0.699957746729085])
    elif control_setup.Trace_mode == 'TCV':
        list_r = np.array([0.624, 0.77078, 0.9534, 0.9534, 0.624, 0.624])
        list_z = np.array([-0.555, -0.311709, -0.311709, -0.75, -0.75, -0.555])
    else:
        raise Exception('Trace_mode wrong !!!')

    index_0 = boolean_points_inout_polygon(Prad_total_matrix_r, Prad_total_matrix_z, list_r, list_z, side="in")

    index_inside = []

    for index in range(len(Prad_total_matrix_r)):

        if index_0[index, 0]:
            index_inside = np.append(index_inside, int(index))

    #####################

    # p1 = Polygon([(list_r[0], list_z[0]), (list_r[1], list_z[1]), (list_r[2], list_z[2]),(list_r[3], list_z[3])])
    #
    # index_inside = []
    #
    # for index in range(len(Prad_total_matrix_r)):
    #     r_poly = Prad_total_matrix_r[index, :].T
    #     z_poly = Prad_total_matrix_z[index, :].T
    #
    #     p2 = Polygon([(r_poly[0], z_poly[0]), (r_poly[1], z_poly[1]), (r_poly[2], z_poly[2]),(r_poly[3], z_poly[3])])
    #
    #     if p1.intersects(p2):
    #         index_inside = np.append(index_inside, int(index))

    ##############

    # print(Prad_total_matrix_r)
    # print(index_inside.astype(int))

    index_inside = index_inside.astype(int)

    Prad_total_matrix_r_sel = Prad_total_matrix_r[index_inside, :]
    Prad_total_matrix_z_sel = Prad_total_matrix_z[index_inside, :]
    Prad_total_matrix_data_sel = Prad_total_matrix_data[index_inside, :]

    # index_nan = np.isnan(Prad_total_matrix_data_sel[:,1])

    index_final = []

    for index2 in range(len(Prad_total_matrix_r_sel)):
        if np.isnan(Prad_total_matrix_data_sel[index2, 1]):
            pass
        else:
            index_final = np.append(index_final, index2)

    index_final = index_final.astype(int)

    Prad_total_matrix_r_final = Prad_total_matrix_r_sel[index_final, :]
    Prad_total_matrix_z_final = Prad_total_matrix_z_sel[index_final, :]
    Prad_total_matrix_data_final = Prad_total_matrix_data_sel[index_final, :]

    output = polygons_integral_volume(Prad_total_matrix_r_final, Prad_total_matrix_z_final,
                                      Prad_total_matrix_data_final)

    return output


def def_ratio_rad_cond_zero(paths):
    Prad_value = Prad_total_int_lot(paths)
    control_setup = load_module(paths.control_setup_file_path)
    if control_setup.Trace_mode == 'WEST':  # 56420

        input_fun_data = {'para_name': 'Te', 'matrix_type': 'quad', 'position_name': 'OMPS', 'location': 'INNER',
                          'data_treatment': 'FLOAT'}
        r_general = 2.46834
        a_general = 0.42812
        q95_general = 4.0976
        lambda_q = 14.45 * 0.001

    elif control_setup.Trace_mode == 'TCV':

        input_fun_data = {'para_name': 'Te', 'matrix_type': 'quad', 'position_name': 'POINT2', 'location': 'INNER',
                          'data_treatment': 'FLOAT'}
        r_general = 0.9
        a_general = 0.1924
        q95_general = 4.2466
        lambda_q = 3.6 * 0.001

    else:
        raise Exception('Trace_mode wrong !!!')

    output = PT_fun_data(input_fun_data, paths)
    OMPS_Te_FLOAT = output['data_result']

    output = 7 * r_general * np.power(q95_general, 2) * Prad_value / (
            4 * a_general * 2000 * np.power(OMPS_Te_FLOAT, 3.5) * lambda_q)

    return output


def count_number_of_line_and_point(file_path, name_started):
    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Extract all variable names
    variable_names = re.findall(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', file_content, re.MULTILINE)

    # Count the number of unique 'line' variables
    name_variables = {name.split('_')[0] for name in variable_names if name.startswith(f'{name_started}')}
    print(name_variables)
    number_count = len(name_variables)

    return number_count


def get_data(paths):
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    print('#################### DP_Data_Grabbing.py ####################')
    print('Start processing ...')

    #################### Grabbing Data ####################
    file_exist = os.path.exists(paths.trace_data_file_path)

    ## iteration number
    if file_exist:
        df = pd.read_csv(paths.trace_data_file_path)
        iteration_number = int(df['iteration_number'].iloc[-1] + 1)
    else:
        iteration_number = 1

    ## time
    time_simu = Get_time(paths)

    ## puffrate
    puff_list_dict, number_of_puff_not_used = read_puff_rate(paths)

    ## input power
    # total_input_power = read_total_input_power()

    Pin_e, Pin_i = read_detail_input_power(paths)

    total_input_power = Pin_e + Pin_i

    ## creat dataframe
    dataframe = pd.DataFrame(
        {'iteration_number': iteration_number, 'time': time_simu, 'total_input_power': total_input_power,
         'Pin_e': Pin_e, 'Pin_i': Pin_i}, index=[0])

    control_setup = load_module(paths.control_setup_file_path)

    active_puff_name = 'puff_rate' + str(int(control_setup.Puff_FD_index))
    dataframe['puff_rate'] = puff_list_dict[active_puff_name]

    for key in puff_list_dict.keys():
        dataframe[key] = puff_list_dict[key]

    if str(control_setup.Trace_mode) == 'WEST':

        # other parameters
        input = {'para_name': 'ne', 'matrix_type': 'quad', 'position_name': 'OMPS', 'location': 'INNER',
                 'data_treatment': 'FLOAT'}
        output = PT_fun_data(input, paths)
        OMPS_ne_FLOAT = output['data_result']

        input = {'para_name': 'Te', 'matrix_type': 'quad', 'position_name': 'OMPS', 'location': 'INNER',
                 'data_treatment': 'FLOAT'}
        output = PT_fun_data(input, paths)
        OMPS_Te_FLOAT = output['data_result']

        input = {'para_name': 'ne', 'matrix_type': 'quad', 'position_name': 'LOT', 'location': 'WALL',
                 'data_treatment': 'MAXABS'}
        output = PT_fun_data(input, paths)
        LOT_ne_MAXABS = output['data_result']

        input = {'para_name': 'Te', 'matrix_type': 'quad', 'position_name': 'LOT', 'location': 'WALL',
                 'data_treatment': 'MAXABS'}
        output = PT_fun_data(input, paths)
        LOT_Te_MAXABS = output['data_result']

        input = {'para_name': 'Ge', 'matrix_type': 'quad', 'position_name': 'LOT', 'location': 'WALL',
                 'data_treatment': 'MAXABS'}
        output = PT_fun_data(input, paths)
        LOT_Ge_MAXABS = output['data_result']

        input = {'para_name': 'qpara', 'matrix_type': 'quad', 'position_name': 'LOT', 'location': 'WALL',
                 'data_treatment': 'MAXABS'}
        output = PT_fun_data(input, paths)
        LOT_Qe_MAXABS = output['data_result']

        input = {'para_name': 'Te', 'matrix_type': 'quad', 'position_name': 'LOSP', 'location': 'WALL',
                 'data_treatment': 'FLOAT'}
        output = PT_fun_data(input, paths)
        LOSP_Te_FLOAT = output['data_result']

        LSN_Radiator_distance = radiator_distance(paths)
        ratio_rad_cond_zero = def_ratio_rad_cond_zero(paths)

        dataframe['OMPS_ne_FLOAT'] = OMPS_ne_FLOAT
        dataframe['OMPS_Te_FLOAT'] = OMPS_Te_FLOAT
        dataframe['LOT_ne_MAXABS'] = LOT_ne_MAXABS
        dataframe['LOT_Te_MAXABS'] = LOT_Te_MAXABS
        dataframe['LOT_Ge_MAXABS'] = LOT_Ge_MAXABS
        dataframe['LOT_Qe_MAXABS'] = LOT_Qe_MAXABS
        dataframe['LOSP_Te_FLOAT'] = LOSP_Te_FLOAT
        dataframe['LSN_radiator_distance'] = LSN_Radiator_distance
        dataframe['ratio_rad_cond_zero'] = ratio_rad_cond_zero

    elif str(control_setup.Trace_mode) == 'PRO':
        Number_of_points = count_number_of_line_and_point("auto_run_S3XE_control_setup.py", "point")
        Number_of_lines = count_number_of_line_and_point("auto_run_S3XE_control_setup.py", "line")

        if Number_of_points != 0:
            for index_point in range(Number_of_points):
                index_number = str(int(index_point + 1))
                index_name = f"point{index_number}"
                position_name = get_variable_from_module(control_setup, f"{index_name}_name")
                para_name = get_variable_from_module(control_setup, f"{index_name}_para_name")
                para_type = get_variable_from_module(control_setup, f"{index_name}_data_treatment")
                para_location = get_variable_from_module(control_setup, f"{index_name}_location")
                para_matrix = get_variable_from_module(control_setup, f"{index_name}_matrix")
                full_name = position_name + '_' + para_name + '_' + para_type

                input_fun_data = {'para_name': para_name, 'matrix_type': para_matrix, 'position_name': index_name,
                                  'location': para_location, 'data_treatment': para_type}
                output = PT_fun_data(input_fun_data, paths)
                dataframe[full_name] = output['data_result']

        if Number_of_lines != 0:
            for index_line in range(Number_of_lines):
                index_number = str(int(index_line + 1))
                index_name = f"line{index_number}"
                position_name = get_variable_from_module(control_setup, f"{index_name}_name")
                para_name = get_variable_from_module(control_setup, f"{index_name}_para_name")
                para_type = get_variable_from_module(control_setup, f"{index_name}_data_treatment")
                para_location = get_variable_from_module(control_setup, f"{index_name}_location")
                para_matrix = get_variable_from_module(control_setup, f"{index_name}_matrix")
                full_name = position_name + '_' + para_name + '_' + para_type

                input_fun_data = {'para_name': para_name, 'matrix_type': para_matrix, 'position_name': index_name,
                                  'location': para_location, 'data_treatment': para_type}

                output = PT_fun_data(input_fun_data, paths)
                dataframe[full_name] = output['data_result']

        if control_setup.Trace_mode == 'TCV':
            ratio_rad_cond_zero = def_ratio_rad_cond_zero(paths)
            dataframe['ratio_rad_cond_zero'] = ratio_rad_cond_zero

    elif control_setup.Trace_mode == 'BASIC':
        pass

    else:
        raise Exception('No Trace_mode information !!!')

    if not os.path.exists(paths.trace_data_file_path):
        dataframe.to_csv(paths.trace_data_file_path, mode='a', index=None)
    else:
        df_old = pd.read_csv(paths.trace_data_file_path)
        list_old = list(df_old)
        list_new = list(dataframe)

        flag_df = False
        ind_df = 0
        for para_name_df in list_new:
            ind_df += 1
            if para_name_df not in list_old:
                flag_df = True
                df_old.insert(ind_df - 1, para_name_df, 0)  # add new column
                # df_old[para_name_df] = 0

        list_old_2 = list(df_old)
        for para_name_df2 in list_old_2:
            if para_name_df2 not in list_new:
                flag_df = True
                df_old.drop(columns=para_name_df2)  # delete column

        if flag_df:
            df_old.to_csv(paths.trace_data_file_path, index=None)

        dataframe.to_csv(paths.trace_data_file_path, mode='a', header=False, index=None)

    print('Processing complete')
    return


def read_target_value_map(time_value, paths):
    df = pd.read_csv(paths.target_value_map_file_path)
    list_time = df['list_time'].values
    list_target_value = df['list_target_value'].values

    if time_value >= min(list_time) and time_value <= max(list_time):
        target_value = np.interp(time_value, list_time, list_target_value)
    else:
        raise Exception('No good time !!!')

    return target_value


# def job_termination_and_mark(paths, marker, exit_code=211):
#     running_info_dict = load_dict_from_file(paths.running_info_file_path)
#
#     update_dict_to_file({
#         'slurm_iteration_number': 0,
#         'slurm_job_id': np.nan,
#         'running_state': 'quick_stop',
#     }, paths.running_info_file_path)
#
#     if check_file_exists(paths.jobid_file_path):
#         run_command(paths.controlled_code_folder_path, f"rm -r auto_run_jobid.txt")
#
#     setup_module = load_module(running_info_dict['setup_file_path'])
#     if setup_module.change_folder_name_after_finished == 'on':
#         add_suffix_to_folder(paths.controlled_code_folder_path, marker)
#
#     exit(exit_code)

# perform_quick_stop(running_info_dict['slurm_job_id'], paths, marker, running_info_dict['setup_file_path'])


# def perform_quick_stop(jobid_str, paths, marker, setup_file_path):
# update_dict_to_file({
#     'slurm_iteration_number': 0,
#     'slurm_job_id': np.nan,
#     'running_state': 'quick_stop',
# }, paths.running_info_file_path)
#
# if check_file_exists(paths.jobid_file_path):
#     run_command(paths.controlled_code_folder_path, f"rm -r auto_run_jobid.txt")

# print("Job stopped successfully")

# setup_module = load_module(setup_file_path)
#
# if setup_module.change_folder_name_after_finished == 'on':
#     add_suffix_to_folder(paths.controlled_code_folder_path, marker)

# exit(1)

# run_command(paths.controlled_code_folder_path, f"scancel {jobid_str}")


def update_target_value(paths, run_info_dict):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    time_sim = df_DP_Trace_Data['time'].iloc[-1]

    control_setup = load_module(paths.control_setup_file_path)

    # control_setup.cache_limit

    # variables_setup = read_setup_simple('DP_setup.txt')
    # df = pd.read_csv(paths.target_value_map_file_path)

    if control_setup.map_t_difference == None:

        time_map_0 = control_setup.target_map_t_list[0]
        time_difference = time_map_0 - time_sim
        modify_variable_in_pyfile(paths.control_setup_file_path, "map_t_difference", time_difference)
        # renew_setup('DP_setup.txt', 'map_t_difference', str(time_difference))
    else:
        time_difference = control_setup.map_t_difference

    time_good = time_sim + time_difference

    time_map_end = control_setup.target_map_t_list[-1]

    if time_good > time_map_end:
        print('>>>>>> Calculation reach the final step in target map <<<<<<')
        # run_command(paths.controlled_code_folder_path, "python3 auto_run_code/running_control.py slurm_quick_stop $script_name")
        # job_termination_and_mark(paths, '_MaxTarget')
        job_termination_and_mark(job_running_mode=run_info_dict['job_running_mode'],
                                 controlled_code_folder_path=run_info_dict['controlled_code_folder_path'],
                                 running_info_file_path=run_info_dict['running_info_file_path'],
                                 jobid_file_path=run_info_dict['jobid_file_path'],
                                 decision_of_loop='MaxTarget',
                                 change_folder_name_after_finished=run_info_dict['change_folder_name_after_finished'],
                                 marker='MaxTarget',
                                 exit_code=211)

    # target_value_realtime = read_target_value_map(time_good, paths)
    # df = pd.read_csv(paths.target_value_map_file_path)
    list_time = control_setup.target_map_t_list
    list_target_value = control_setup.target_map_v_list

    if time_good >= min(list_time) and time_good <= max(list_time):
        target_value_realtime = np.interp(time_good, list_time, list_target_value)
    else:
        raise Exception('No good time !!!')

    # if target_value_realtime < 1.4e19:
    #     target_value_realtime = 1.4e19

    target_value_realtime_str = '{:e}'.format(target_value_realtime)

    modify_variable_in_pyfile(paths.control_setup_file_path, "Target_value", target_value_realtime_str)


# def read_gas_puff_map(time_value, paths):
#     puff_list_dict_not_used, number_of_puff = read_puff_rate(paths)
#
#     df = pd.read_csv(paths.gas_puff_map_file_path)
#     list_time = df['list_time'].values
#
#     puff_list_dict = {}
#
#     for index in range(number_of_puff):
#         index_real = index + 1
#         var_name = 'puff_rate_' + str(index_real)
#         list_puff_value = df[var_name].values
#         if time_value >= min(list_time) and time_value <= max(list_time):
#             puff_value = np.interp(time_value, list_time, list_puff_value)
#         else:
#             raise Exception('No good time !!!')
#
#         puff_list_dict[var_name] = puff_value
#
#     return puff_list_dict


def update_gas_puff_map(paths, run_info_dict):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    time_sim = df_DP_Trace_Data['time'].iloc[-1]

    control_setup = load_module(paths.control_setup_file_path)

    # variables_setup = read_setup_simple('DP_setup.txt')
    # df = pd.read_csv(paths.gas_puff_map_file_path)

    if control_setup.gas_map_dt is None:
        time_map_0 = control_setup.gas_map_t_list[0]
        time_difference = time_map_0 - time_sim
        modify_variable_in_pyfile(paths.control_setup_file_path, "gas_map_dt", time_difference)
    else:
        time_difference = control_setup.gas_map_dt

    time_good = time_sim + time_difference

    # time_map_end = df['list_time'].iloc[-1]

    time_map_end = control_setup.gas_map_t_list[-1]

    if time_good > time_map_end:
        print('>>>>>> Calculation reach the final step in target map <<<<<<')

        job_termination_and_mark(job_running_mode=run_info_dict['job_running_mode'],
                                 controlled_code_folder_path=run_info_dict['controlled_code_folder_path'],
                                 running_info_file_path=run_info_dict['running_info_file_path'],
                                 jobid_file_path=run_info_dict['jobid_file_path'],
                                 decision_of_loop='MaxTarget',
                                 change_folder_name_after_finished=run_info_dict['change_folder_name_after_finished'],
                                 marker='MaxTarget',
                                 exit_code=211)

    # puff_list_dict = read_gas_puff_map(time_good, paths)
    puff_list_dict_not_used, number_of_puff = read_puff_rate(paths)

    # df = pd.read_csv(paths.gas_puff_map_file_path)
    # list_time = df['list_time'].values
    list_time = control_setup.gas_map_t_list

    # puff_list_dict = {}
    puff_list = []

    for index in range(number_of_puff):
        index_real = index + 1
        var_name = 'puff_rate_' + str(index_real)
        list_puff_value = get_variable_from_module(control_setup, f"gas_map_puff{index_real}_list")
        # list_puff_value = df[var_name].values
        if min(list_time) <= time_good <= max(list_time):
            puff_value = np.interp(time_good, list_time, list_puff_value)
        else:
            raise Exception('No good time !!!')

        # puff_list_dict[var_name] = puff_value

        if puff_value >= 0:
            puff_value_used = puff_value
        else:
            puff_value_used = 0

        # puff_list = []
        # for key in puff_list_dict.keys():
        #     puff_value = float(puff_list_dict[key])
        #     if puff_value >= 0:
        #         puff_value_used = puff_value
        #     else:
        #         puff_value_used = 0
        #
        puff_value_used_str = '{:e}'.format(puff_value_used)

        puff_list.append(puff_value_used_str)

    puff_list_str = ', '.join(puff_list)

    read_modify_variable_in_eirene_file(paths.eirene_coupling_file_path, "puff_rate", puff_list_str)


# def read_input_power_map(time_value, paths):
#     # puff_list_dict_not_used, number_of_puff = read_puff_rate()
#
#     df = pd.read_csv(paths.input_power_map_file_path)
#     list_time = df['list_time'].values
#
#     num_items = df.shape[1] - 1
#
#     power_list_dict = {}
#
#     for index in range(num_items):
#         index_real = index + 1
#         var_name = 'input_power_' + str(index_real)
#         list_input_power = df[var_name].values
#         if time_value >= min(list_time) and time_value <= max(list_time):
#             input_power_value = np.interp(time_value, list_time, list_input_power)
#         else:
#             raise Exception('No good time !!!')
#
#         power_list_dict[var_name] = input_power_value
#
#     return power_list_dict


def update_input_power_map(paths, run_info_dict):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    time_sim = df_DP_Trace_Data['time'].iloc[-1]

    control_setup = load_module(paths.control_setup_file_path)

    if control_setup.power_map_dt is None:
        time_map_0 = control_setup.power_map_t_list[0]
        time_difference = time_map_0 - time_sim
        modify_variable_in_pyfile(paths.control_setup_file_path, "power_map_dt", time_difference)

    else:
        time_difference = control_setup.power_map_t_list

    time_good = time_sim + time_difference

    time_map_end = control_setup.power_map_t_list[-1]

    if time_good > time_map_end:
        print('>>>>>> Calculation reach the final step in target map <<<<<<')
        job_termination_and_mark(job_running_mode=run_info_dict['job_running_mode'],
                                 controlled_code_folder_path=run_info_dict['controlled_code_folder_path'],
                                 running_info_file_path=run_info_dict['running_info_file_path'],
                                 jobid_file_path=run_info_dict['jobid_file_path'],
                                 decision_of_loop='MaxTarget',
                                 change_folder_name_after_finished=run_info_dict['change_folder_name_after_finished'],
                                 marker='MaxTarget',
                                 exit_code=211)

    list_time = control_setup.power_map_t_list

    pin_list_dict_not_used, number_of_pin = read_input_power(paths)

    # power_list_dict = {}

    puff_list = []

    for index in range(number_of_pin):
        index_real = index + 1
        # var_name = 'input_power_' + str(index_real)

        list_input_power = get_variable_from_module(control_setup, f"power_map_in{index_real}_list")
        # list_input_power = df[var_name].values
        if min(list_time) <= time_good <= max(list_time):
            input_power_value = np.interp(time_good, list_time, list_input_power)
        else:
            raise Exception('Not good time !!!')

        if input_power_value >= 0:
            input_power_used = input_power_value
        else:
            input_power_used = 0

        puff_value_used_str = '{:e}'.format(input_power_used)

        puff_list.append(puff_value_used_str)

    #     # power_list_dict[var_name] = input_power_value
    #
    # puff_list = []
    # for key in power_list_dict.keys():
    #     puff_value = float(power_list_dict[key])
    #     if puff_value >= 0:
    #         puff_value_used = puff_value
    #     else:
    #         puff_value_used = 0
    #
    #     puff_value_used_str = '{:e}'.format(puff_value_used)
    #
    #     puff_list.append(puff_value_used_str)

    tbc_list_str = ', '.join(puff_list)

    read_modify_variable_in_para_file(paths.control_setup_file_path, "TBC", tbc_list_str)


def name_analyse(para_name, description):
    if para_name == 'LSN_radiator_distance':
        result = str('rad')
    elif para_name == 'ratio_rad_cond_zero':
        result = str('rd')
    else:
        name_composition = para_name.split('_')
        if description == 'para_name':
            result = str(name_composition[1])
        elif description == 'position':
            result = str(name_composition[0])
        else:
            raise Exception('No description information !!!')
    return result


def standard_deviation(paths):
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)

    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)
    target_para = str(control_setup.target_para)
    number_of_samples = int(control_setup.number_of_samples)
    iteration_number = int(df_DP_Trace_Data['iteration_number'].iloc[-1])
    target_value = float(control_setup.Target_value)

    if iteration_number < number_of_samples:
        standard_deviation_value = target_value / 400
    else:
        time_list = df_DP_Trace_Data['time'].iloc[int(-1 * number_of_samples):]
        target_sample_list = df_DP_Trace_Data[target_para].iloc[int(-1 * number_of_samples):]
        fit_result = np.polyfit(time_list, target_sample_list, 4)
        fit_equation = np.poly1d(fit_result)
        fit_value_list = fit_equation(time_list)
        error_list = target_sample_list - fit_value_list
        standard_deviation_value = np.std(error_list, ddof=1)

    return float(standard_deviation_value)


def amplitude(paths):
    standard_deviation_value = standard_deviation(paths)
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)
    amplitude_factor = float(control_setup.amplitude_factor)
    amplitude_value = standard_deviation_value * amplitude_factor
    return amplitude_value


def feed_back_function(dataframe, paths):
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)
    # Target_value = float(dataframe['Target_value'])
    Target_value = float(dataframe['Target_value'].iloc[0])

    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    target_para = str(control_setup.target_para)  # define the target parameter
    K_p = float(control_setup.K_p)  # proportional gain
    K_i = float(control_setup.K_i)  # integral gain
    K_d = float(control_setup.K_d)  # derivative gain
    max_puff = float(control_setup.max_puff)  # maximum puff rate
    min_puff = float(control_setup.min_puff)  # minimum puff rate
    rollover_check = control_setup.rollover_check  # rollover checking mode
    rollover_Ge_puff = float(control_setup.rollover_Ge_puff)  # puff value when rollover happens
    max_Ge_exp = float(control_setup.max_Ge_exp)  # peak Ge value when rollover happens

    if control_setup.integral_bg_puff is None:
        old_integral_bg_term = dataframe['old_puff_rate'].iloc[0]
    else:
        old_integral_bg_term = float(control_setup.integral_bg_puff)

    # Initialization
    rollover_switch = str(control_setup.rollover_switch)
    ind_difference = 0
    time_difference = 0
    max_Ge = max_Ge_exp
    quick_fall_back = 'off'

    # judging if it's the first time running the feed back
    file_exist = os.path.isfile(paths.trace_feedback_file_path)
    if file_exist:
        state_mark = 'not_first_run'
    else:
        state_mark = 'first_run'

    # calculate error and kn
    df_setpoint = df_DP_Trace_Data[target_para]
    target_para_key_name = name_analyse(target_para, 'para_name')
    # if target_para == 'LOT_Ge_MAXABS':
    if target_para_key_name == 'Ge':
        # to check the rollover state
        if rollover_check == 'NONE':  # active control of attach state
            rollover_switch = 'NONE'
        else:
            file_exist = os.path.isfile(paths.trace_feedback_file_path)
            if file_exist:
                df_DP_Trace_output = pd.read_csv(paths.trace_feedback_file_path)
                rollover_switch = df_DP_Trace_output['rollover_switch'].iloc[-1]
                max_Ge = df_DP_Trace_output['max_Ge'].iloc[-1]

            if rollover_switch == 'off':
                ind_max = df_setpoint.idxmax()
                ind_latest = df_setpoint.shape[0] - 1
                time_difference = df_DP_Trace_Data['time'].iloc[-1] - df_DP_Trace_Data['time'].iloc[ind_max]
                ind_difference = ind_latest - ind_max
                if ind_difference >= 6:
                    rollover_switch = 'on'
                max_Ge = df_setpoint.max()
        # give response
        if rollover_switch == 'off':
            max_Ge_final = max(max_Ge_exp, max_Ge)
            Target_value = 2.0 * max_Ge_final - Target_value
            K_n = -4.0 / 3.6e22 * 2e17
        elif rollover_switch == 'on':
            K_n = 4.0 / 3.6e22 * 2e17
            min_puff = rollover_Ge_puff
        elif rollover_switch == 'NONE':
            K_n = -4.0 / 3.6e22 * 2e17
        else:
            raise Exception('No rollover_switch defined !!!')
    else:
        if target_para_key_name == 'ne':
            K_n = -1.0  # for ne
        elif target_para_key_name == 'Te':
            K_n = 2e17  # for Te
        elif target_para_key_name == 'Qe':
            K_n = 1e5  # for Te
        elif target_para_key_name == 'rad':
            K_n = -2e20  # for Te
        elif target_para_key_name == 'rd':  # ratio of detachment
            K_n = -1.2e18
        else:
            raise Exception('No target defined !!!')

    Error = float(df_setpoint.iloc[-1]) - Target_value

    # proportional_term
    proportional_term = K_n * K_p * Error

    # integral_term
    ki_kn_error = K_i * K_n * Error
    old_puff_rate = dataframe['old_puff_rate'].iloc[0]
    noise_amplitude = amplitude(paths)
    if ki_kn_error < 0:
        if old_puff_rate <= 0:
            ki_kn_error = 0
        elif abs(Error) > abs(noise_amplitude):
            ki_kn_error = 10 * ki_kn_error
            quick_fall_back = 'on'

    if state_mark == 'first_run':
        integral_bg_term = old_integral_bg_term
        deta_t = df_DP_Trace_Data['time'].iloc[-1] - df_DP_Trace_Data['time'].iloc[-2]
    elif state_mark == 'not_first_run':
        df_DP_Trace_output = pd.read_csv(paths.trace_feedback_file_path)
        deta_t = df_DP_Trace_Data['time'].iloc[-1] - df_DP_Trace_output['time'].iloc[-1]
        old_ki_kn_error = df_DP_Trace_output['ki_kn_error'].iloc[-1]
        integral_bg_term = old_integral_bg_term + integrate.trapz([old_ki_kn_error, ki_kn_error], dx=deta_t)
    else:
        raise Exception('state_mark error !!!')

    # derivative_term
    deta_setpoint = df_setpoint.iloc[-1] - df_setpoint.iloc[-2]
    if deta_t == 0:
        derivative_term = 0
    else:
        derivative_term = K_n * K_d * deta_setpoint / deta_t

    # new_puff_cal = proportional_term + integral_term + derivative_term + cst_puff
    integral_bg_puff = max(integral_bg_term, 0)
    modify_variable_in_pyfile(paths.control_setup_file_path, "integral_bg_puff", integral_bg_puff)

    new_puff_cal = proportional_term + derivative_term + integral_bg_term
    new_puff = max(min(max_puff, new_puff_cal), min_puff)

    if state_mark == 'first_run' and new_puff == 0:
        ki_kn_error = 0

    dataframe['Target_value'] = Target_value
    dataframe['Error'] = Error
    dataframe['noise_amplitude'] = noise_amplitude
    dataframe['ki_kn_error'] = ki_kn_error
    dataframe['proportional_term'] = proportional_term
    dataframe['integral_bg_term'] = integral_bg_term
    dataframe['derivative_term'] = derivative_term
    dataframe['new_puff_cal'] = new_puff_cal
    dataframe['new_puff'] = new_puff
    dataframe['rollover_switch'] = rollover_switch
    dataframe['max_Ge'] = max_Ge
    dataframe['ind_difference'] = ind_difference
    dataframe['time_difference'] = time_difference
    dataframe['K_n'] = K_n
    dataframe['quick_fall_back'] = quick_fall_back

    return new_puff, dataframe


def write_puff_rate(puff_applied, paths):
    puff_applied = float(puff_applied)
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)
    Lock_ratio = control_setup.Lock_ratio
    Puff_FD_index = control_setup.Puff_FD_index

    puff_list_dict, number_of_puff_not_used = read_puff_rate(paths)
    puff_list_dict_new = {}
    if Lock_ratio == 'on':
        old_puff_rate = read_active_puff(paths)
        ratio = puff_applied / old_puff_rate
        for key in puff_list_dict.keys():
            puff_list_dict_new[key] = puff_list_dict[key] * ratio
    elif Lock_ratio == 'off':
        for key in puff_list_dict.keys():
            if key == 'puff_rate' + str(int(Puff_FD_index)):
                puff_list_dict_new[key] = puff_applied
            else:
                puff_list_dict_new[key] = puff_list_dict[key]

    puff_list = []
    for key in puff_list_dict_new.keys():
        puff_list.append(str(puff_list_dict_new[key]))
    puff_list_str = ', '.join(puff_list)

    read_modify_variable_in_eirene_file(paths.eirene_coupling_file_path, "puff_rate", puff_list_str)


def update(paths):
    print('#################### DP_Update_GasPuff_FD.py ####################')
    print('Start processing ...')

    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)

    active_feedback = str(control_setup.active_feedback)

    if active_feedback == 'on':
        print('Feedback activated')
        ################ parse Data grabbing file ################
        df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
        old_puff_rate = read_active_puff(paths)

        # old_puff_rate = df_DP_Trace_Data['puff_rate'].iloc[-1]
        iteration_number = df_DP_Trace_Data['iteration_number'].iloc[-1]
        time_simu = df_DP_Trace_Data['time'].iloc[-1]

        ################ decide output ################
        target_para = str(control_setup.target_para)  # define the target parameter, MBIS

        # possible to control Target_value from a target list as function of iteration or time
        Target_value = float(control_setup.Target_value)  # target value for feedback, MBIS
        max_puff = float(control_setup.max_puff)  # maximum puff rate, MBIS
        min_puff = float(control_setup.min_puff)  # minimum puff rate, MBIS

        # # time delay setting
        time_delay = float(control_setup.time_delay)  # time delay of gas puff, MBIS
        # # time buffer setting
        t_buffer = float(control_setup.t_buffer)  # buffer time, MBIS

        ################ data processing ################
        ind_feedback_wait = 3

        if float(iteration_number) < ind_feedback_wait:
            new_puff = float(old_puff_rate)
        else:
            ####### special target ######
            if t_buffer > 0:
                time_start = df_DP_Trace_Data['time'].iloc[ind_feedback_wait - 2]
                value_start = df_DP_Trace_Data[target_para].iloc[ind_feedback_wait - 1]

                if float(time_simu) < (t_buffer + time_start):
                    Target_value = (Target_value - value_start) * (
                            float(time_simu) - time_start) / t_buffer + value_start
            #############################
            dataframe = pd.DataFrame(
                {'iteration_number': iteration_number, 'time': time_simu, 'Target_para': target_para,
                 'Target_value': Target_value, 'old_puff_rate': old_puff_rate,
                 'min_puff': min_puff, 'max_puff': max_puff}, index=[0])

            new_puff, dataframe = feed_back_function(dataframe, paths)

            ################# record output #################
            if not os.path.exists(paths.trace_feedback_file_path):
                dataframe.to_csv(paths.trace_feedback_file_path, mode='a', index=None)
            else:
                dataframe.to_csv(paths.trace_feedback_file_path, mode='a', header=False, index=None)

        ################ apply time delay #######################
        if time_delay == 0:
            puff_applied = new_puff
        else:
            file_exist = os.path.isfile(paths.trace_feedback_file_path)
            if file_exist:
                df_DP_Trace_output = pd.read_csv(paths.trace_feedback_file_path)
                List_time_output = df_DP_Trace_output['time'].values
                List_puff_output = df_DP_Trace_output['new_puff'].values
                if List_time_output[-1] - List_time_output[0] >= time_delay:
                    time_past = List_time_output[-1] - time_delay
                    puff_applied = np.interp(time_past, List_time_output, List_puff_output)
                else:
                    puff_applied = float(old_puff_rate)
            else:
                puff_applied = float(old_puff_rate)

        ################ modify setup in simulation #######################
        write_puff_rate(puff_applied, paths)

    else:
        print('Feedback deactivated')

    print('Processing complete')


def profile_intersection(input):
    list_r_sim = input['list_r_sim']
    list_para_sim = input['list_para_sim']
    list_r_target = input['list_r_target']

    list_r_start = max(min(list_r_sim), min(list_r_target))
    list_r_end = min(max(list_r_sim), max(list_r_target))

    index = np.where((list_r_start <= list_r_sim) & (list_r_sim <= list_r_end))

    list_r_sim_intersection = list_r_sim[index]
    list_para_sim_intersection = list_para_sim[index]

    return list_r_sim_intersection, list_para_sim_intersection


def profile_gradient(list_r, list_para):
    list_grad = []
    for index in range(len(list_r)):
        # if index == len(list_r) - 1:
        #     grad_result = (list_para[-2] - list_para[-1]) / (list_r[-2] - list_r[-1])
        # else:
        #     grad_result = (list_para[index] - list_para[index + 1]) / (list_r[index] - list_r[index + 1])

        if index == 0:
            grad_result = (list_para[0] - list_para[1]) / (list_r[0] - list_r[1])
        elif index == len(list_r) - 1:
            grad_result = (list_para[-1] - list_para[-2]) / (list_r[-1] - list_r[-2])
        else:
            distance = min(abs(list_r[index - 1] - list_r[index]), abs(list_r[index + 1] - list_r[index]))
            r_a = list_r[index] - distance
            r_b = list_r[index] + distance
            para_a = np.interp(r_a, list_r, list_para)
            para_b = np.interp(r_b, list_r, list_para)
            grad_result = (para_a - para_b) / (r_a - r_b)
        list_grad = np.append(list_grad, grad_result)
    # for index in range(len(list_r)):
    #     if index == 0:
    #         grad_result = (list_para[0] - list_para[1]) / (list_r[0] - list_r[1])
    #     elif index == len(list_r) - 1:
    #         grad_result = (list_para[-1] - list_para[-2]) / (list_r[-1] - list_r[-2])
    #     else:
    #         distance = min(abs(list_r[index - 1] - list_r[index]), abs(list_r[index + 1] - list_r[index]))
    #         r_a = list_r[index] - distance
    #         r_b = list_r[index] + distance
    #         para_a = np.interp(r_a, list_r, list_para)
    #         para_b = np.interp(r_b, list_r, list_para)
    #         grad_result = (para_a - para_b) / (r_a - r_b)
    #     list_grad = np.append(list_grad, grad_result)

    return list_grad


def profile_error(input):
    list_r_sim_intersection = input['list_r_sim_intersection']
    list_para_sim_intersection = input['list_para_sim_intersection']
    list_r_target = input['list_r_target']
    list_para_target = input['list_para_target']

    output_gradient_sim_intersection = profile_gradient(list_r_sim_intersection, list_para_sim_intersection)

    list_para_target_interp_by_sim_intersection = np.interp(list_r_sim_intersection, list_r_target, list_para_target)

    output_gradient_target = profile_gradient(list_r_sim_intersection, list_para_target_interp_by_sim_intersection)

    list_gradient_error = output_gradient_sim_intersection - output_gradient_target

    return list_gradient_error


def delete_nan(original_list_x, original_list_y):
    list_x = []
    list_y = []
    for index in range(len(original_list_x)):
        if np.isnan(original_list_x[index]) or np.isnan(original_list_y[index]):
            pass
        else:
            list_x = np.append(list_x, original_list_x[index])
            list_y = np.append(list_y, original_list_y[index])

    return list_x, list_y


def list_r_and_para_error(para_name, paths):
    input = {'para_name': para_name, 'matrix_type': 'quad', 'position_name': 'OMP', 'location': 'INNER',
             'data_treatment': 'NONE'}
    output = PT_fun_data(input, paths)
    list_r = output['list_r']
    list_para = output['data_result']

    df = pd.read_csv(paths.profile_feedback_file_path)
    list_r_target = df['list_r'].values

    if para_name == 'ne':
        list_para_target = df['list_ne'].values
    elif para_name == 'Te':
        list_para_target = df['list_te'].values
    else:
        raise Exception('para_name error !!!')

    input_intersection = {'list_r_sim': list_r, 'list_para_sim': list_para, 'list_r_target': list_r_target}

    list_r_sim_intersection, list_para_sim_intersection = profile_intersection(input_intersection)

    input_error = {'list_r_sim_intersection': list_r_sim_intersection,
                   'list_para_sim_intersection': list_para_sim_intersection, 'list_r_target': list_r_target,
                   'list_para_target': list_para_target}

    list_para_error = profile_error(input_error)

    list_r_sim_intersection, list_para_error = delete_nan(list_r_sim_intersection, list_para_error)

    return list_r_sim_intersection, list_para_error


def read_or_save_pd(input, paths):
    file_name = input['file_name']
    if input['operation'] == 'read':
        df = pd.read_csv(file_name)

        time = df['time'].iloc[-1]
        result = []
        for index in range(df['number_index'].iloc[-1]):
            index_number = index + 1
            value = df['index' + str(index_number)].iloc[-1]
            result = np.append(result, value)

        return time, result
    elif input['operation'] == 'save' or input['operation'] == 'renew':
        df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
        time_new = df_DP_Trace_Data['time'].iloc[-1]
        iteration_number = df_DP_Trace_Data['iteration_number'].iloc[-1]
        number_index = len(input['list'])
        dataframe = pd.DataFrame({'iteration_number': iteration_number, 'time': time_new, 'number_index': number_index},
                                 index=[0])

        for index in range(number_index):
            dataframe['index' + str(index + 1)] = input['list'][index]

        if input['operation'] == 'save':
            if not os.path.exists(file_name):
                dataframe.to_csv(file_name, mode='a', index=None)
            else:
                dataframe.to_csv(file_name, mode='a', header=False, index=None)
        elif input['operation'] == 'renew':
            dataframe.to_csv(file_name, mode='w', index=None)


def profile_feedback_function(para_name, paths):
    list_r_sim_intersection, list_para_error = list_r_and_para_error(para_name, paths)

    if para_name == 'ne':
        list_para_error_normalized = list_para_error / 3e21
        defaut_diff = 0.3
    elif para_name == 'Te':
        list_para_error_normalized = list_para_error / 1e4
        defaut_diff = 1
    else:
        raise Exception('para_name error !!!')

    # print(list_para_error_normalized)
    # variables = DP_Setup_RW.read_setup_simple('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)

    kp = -1 * float(control_setup.diff_fd_kp)
    ki = -1 * float(control_setup.diff_fd_ki)

    list_para_proportional_term = kp * list_para_error_normalized
    ki_error = ki * list_para_error_normalized

    trace_ki_error_specified_file_path = replace_filename_part(paths.trace_ki_error_general_file_path, "para",
                                                               str(para_name))

    diff_bg_specified_file_path = replace_filename_part(paths.diff_bg_general_file_path, "para",
                                                        str(para_name))

    trace_diff_specified_file_path = replace_filename_part(paths.trace_diff_general_file_path, "para",
                                                           str(para_name))

    file_exist_ki_error = os.path.isfile(trace_ki_error_specified_file_path)
    file_exist_diff_bg = os.path.isfile(diff_bg_specified_file_path)

    if file_exist_ki_error:
        input = {'operation': 'read', 'file_name': trace_ki_error_specified_file_path}
        time_old, list_ki_error_old = read_or_save_pd(input, paths)

        input = {'operation': 'read', 'file_name': diff_bg_specified_file_path}
        time_old_not_used, list_diff_bg_old = read_or_save_pd(input, paths)

        input = {'operation': 'read', 'file_name': trace_diff_specified_file_path}
        time_old, list_diff_old = read_or_save_pd(input, paths)

        for index_diff in range(len(list_diff_old)):  # control the integral item to be 0 with too small diff value
            if list_diff_old[index_diff] < 0.005 and ki_error[index_diff] < 0:
                ki_error[index_diff] = 0

        df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
        time_new = df_DP_Trace_Data['time'].iloc[-1]

        deta_t = time_new - time_old

        list_integral_bg_term = []
        for index in range(len(list_ki_error_old)):
            integral_bg_term = list_diff_bg_old[index] + integrate.trapz([list_ki_error_old[index], ki_error[index]],
                                                                         dx=deta_t)
            list_integral_bg_term = np.append(list_integral_bg_term, integral_bg_term)
    elif file_exist_diff_bg:
        input = {'operation': 'read', 'file_name': diff_bg_specified_file_path}
        time_old_not_used, list_diff_bg_old = read_or_save_pd(input, paths)
        list_integral_bg_term = list_diff_bg_old
    else:
        list_integral_bg_term = [defaut_diff] * len(ki_error)

    list_diff_new = list_para_proportional_term + list_integral_bg_term

    input = {'operation': 'save', 'list': list_diff_new, 'file_name': trace_diff_specified_file_path}
    read_or_save_pd(input, paths)

    input = {'operation': 'save', 'list': ki_error, 'file_name': trace_ki_error_specified_file_path}
    read_or_save_pd(input, paths)

    input = {'operation': 'renew', 'list': list_integral_bg_term, 'file_name': diff_bg_specified_file_path}
    read_or_save_pd(input, paths)

    return list_r_sim_intersection, list_diff_new


def write_diff_h5(directory, value, paths):
    f = h5py.File(paths.diffusion_file_path, 'a')
    f[directory] = value
    f.close()


def list_filter(list_long, list_index):
    list_short = []
    for index in list_index:
        list_short.append(list_long[index])

    return list_short


def PT_fun_rz_to_npsi(Rcenter, Zcenter, paths):
    psi1 = interp2linear(common_variable('r', paths), common_variable('z', paths), common_variable('psi', paths),
                         Rcenter, Zcenter)
    NPSIcenter = (psi1 - common_variable('psi_min', paths)) / (
            common_variable('PSIsep', paths) - common_variable('psi_min', paths))
    output = {'NPSIcenter': NPSIcenter}
    return output


def PT_fun_the_zones_in_PFR(paths):
    File_mesh = h5py.File(paths.mesh_file_path, 'r')
    neighbors = File_mesh['/neighbors'][:]

    zone_PFR = []

    for k in range(neighbors.shape[1]):
        if neighbors[0, k] == -3:
            zone_PFR = np.append(zone_PFR, neighbors[5, k])

    output = zone_PFR.astype(int)

    return output


def read_number_of_species(paths):
    line_needed = find_param_data(paths.param_raptorx_file_path, 'Nelts =', 'Number of chemical elements in the plasma')
    head, sep, tail = line_needed.partition('!')
    # head = head.replace(',', ' ')
    head = head.split()
    number = head[2]

    return int(number)


def PT_fun_zones_quad_rz(zone_number, paths):
    File_mesh_raptorx = h5py.File(paths.mesh_raptorx_file_path, 'r')
    rho0 = common_variable('rho0_mesh', paths)
    Rcorner = File_mesh_raptorx['/zone' + str(zone_number) + '/Rcorners'][:, :, 0] * rho0
    Zcorner = File_mesh_raptorx['/zone' + str(zone_number) + '/Zcorners'][:, :, 0] * rho0
    File_mesh_raptorx.close()

    R1_matrix = Rcorner[0:-1, 0:-1]
    R2_matrix = Rcorner[1:, 0:-1]
    R3_matrix = Rcorner[1:, 1:]
    R4_matrix = Rcorner[0:-1, 1:]

    Z1_matrix = Zcorner[0:-1, 0:-1]
    Z2_matrix = Zcorner[1:, 0:-1]
    Z3_matrix = Zcorner[1:, 1:]
    Z4_matrix = Zcorner[0:-1, 1:]

    Rcenter = (R1_matrix + R2_matrix + R3_matrix + R4_matrix) / 4
    Zcenter = (Z1_matrix + Z2_matrix + Z3_matrix + Z4_matrix) / 4

    output = {'Rcorner': Rcorner, 'Zcorner': Zcorner, 'Rcenter': Rcenter, 'Zcenter': Zcenter}

    return output


def generate_nonlinear_matrix(matrix_reference, gain, bias, type1, type2):
    # B = np.ones((matrix_reference.shape[0], matrix_reference.shape[1]))
    B = np.array(matrix_reference, copy=True)
    if type1 == 1:
        A = B
    elif type1 == 2:
        A = flip90_left(B)
    elif type1 == 3:
        A = flip180(B)
    elif type1 == 4:
        A = flip90_right(B)
    else:
        raise Exception('wrong type1')

    if type2 == 1:
        pass
    elif type2 == 2:
        A = flipR(A)
    else:
        raise Exception('wrong type2')

    # bias :0-1
    slop = (A.shape[1] - 1) / (A.shape[0] - 1) * bias

    for index_line in range(A.shape[0]):
        limit_col = index_line * slop
        for index_col in range(A.shape[1]):
            # number_col = index_col + 1
            if index_col == 0:
                A[index_line, index_col] = (gain - 1) / (A.shape[0] - 1) * index_line + 1

            elif limit_col >= index_col > 0:
                A[index_line, index_col] = ((gain - 1) / (A.shape[0] - 1) * index_line) / limit_col * (
                        limit_col - index_col) + 1
            else:
                A[index_line, index_col] = 1

    if type1 == 1:
        A = A
    elif type1 == 2:
        A = flip90_right(A)
    elif type1 == 3:
        A = flip180(A)
    elif type1 == 4:
        A = flip90_left(A)
    else:
        raise Exception('wrong type1')

    if type2 == 1:
        pass
    elif type2 == 2:
        A = flipR(A)
    else:
        raise Exception('wrong type2')

    return A


def generate_diff_map(list_r, list_D, list_chie, paths):
    list_r = list_r.tolist()
    list_D = list_D.tolist()
    list_chie = list_chie.tolist()

    input = {'position_name': 'OMPS'}
    output = PT_fun_positions(input, paths)
    r_omps = float(output['list_r'])
    z_omps = float(output['list_z'])

    position_center = np.where(common_variable('psi', paths) == common_variable('psi_min', paths))
    R0 = common_variable('r', paths)[(position_center[1])[0], 0]
    Z0 = common_variable('z', paths)[0, (position_center[1])[0]]

    list_r_abs = [i + r_omps for i in list_r]

    # limit the max value
    r_max = common_variable('r', paths).max()

    list_r_extend = [R0] + list_r_abs + [r_max]
    list_D_extend = [list_D[0]] + list_D + [list_D[-1]]
    list_chie_extend = [list_chie[0]] + list_chie + [list_chie[-1]]

    index_useful = []
    for index_list in range(len(list_r_extend)):
        if list_r_extend[index_list] >= R0 and list_r_extend[index_list] <= r_max:
            index_useful.append(index_list)

    list_r_entire = list_filter(list_r_extend, index_useful)
    list_D_entire = list_filter(list_D_extend, index_useful)
    list_chie_entire = list_filter(list_chie_extend, index_useful)

    list_r_fine = np.linspace(R0, r_max, 1000)
    list_r_fine = list_r_fine.tolist()
    list_r_fine.extend(list_r_entire)
    list_r_full = list(set(list_r_fine))
    list_r_full.sort()

    list_D_full = np.interp(list_r_full, list_r_entire, list_D_entire)
    list_chie_full = np.interp(list_r_full, list_r_entire, list_chie_entire)
    list_z_full = [Z0] * len(list_r_full)

    # variables = DP_Setup_RW.read_setup_simple('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)

    if str(control_setup.ballooning) == 'on':
        b_exponent = float(control_setup.b_exponent)
        Btotal_matrix = np.power(
            (np.power(common_variable('br_mag', paths), 2) + np.power(common_variable('bz_mag', paths),
                                                                      2) + np.power(
                common_variable('bphi_mag', paths), 2)), 0.5)
        B0 = interp2linear(common_variable('r', paths), common_variable('z', paths), Btotal_matrix, R0, Z0)

        if str(control_setup.omp_diff_fix) == 'on':
            list_B_full = interp2linear(common_variable('r', paths), common_variable('z', paths), Btotal_matrix,
                                        list_r_full,
                                        list_z_full)

            list_scal = B0 / np.array(list_B_full)
            list_scal_new = np.power(list_scal, b_exponent)

            list_D_full = np.array(list_D_full) / list_scal_new
            list_chie_full = np.array(list_chie_full) / list_scal_new

    if str(control_setup.div_diff_Pneu) == 'on':
        # input_matrix_general = {'para_name': 'Pneu_total', 'matrix_type': 'tri'}
        output_matrix_general = PT_fun_matrix_general(para_name='Pneu_total', matrix_type='tri', paths=paths)
        matrix_r = output_matrix_general.r
        matrix_z = output_matrix_general.z
        matrix_pneu = output_matrix_general.value
        max_pneu = matrix_pneu.max()

        input_fun_data = {'para_name': 'Pneu_total', 'matrix_type': 'tri', 'position_name': 'outerOMPS',
                          'location': 'INNER', 'data_treatment': 'MEANABS'}
        mean_pneu_omp = PT_fun_data(input_fun_data, paths)['data_result']
        pneu0 = mean_pneu_omp * float(control_setup.div_Pneu_min_scal)

    # list_Z_full = [z_omps] * len(list_r_full)
    list_npsi_full = PT_fun_rz_to_npsi(list_r_full, list_z_full, paths)['NPSIcenter']
    list_chi_full = np.array(list_chie_full, copy=True)

    if str(control_setup.chi_special) == 'on':
        index_npsi = np.argwhere(list_npsi_full < 1)
        list_chi_full[index_npsi] = list_chie_full[index_npsi] * float(control_setup.chi_chie_ratio)

    para_list = {}
    para_list['chie'] = list_chie_full
    para_list['chi'] = list_chi_full
    para_list['D'] = list_D_full
    para_list['nu'] = np.array([float(control_setup.diff_nu)] * len(list_D_full))

    # list_npsi_abs = list_npsi_abs.tolist()
    #
    #
    # list_npsi_full = [0] + list_npsi_abs + [1000] #set a big enough number

    list_zones_in_PFR = PT_fun_the_zones_in_PFR(paths)

    number_of_species = read_number_of_species(paths)

    write_diff_h5('Version', ['V20220520'], paths)

    write_diff_h5('Nmaps', [number_of_species], paths)

    for key_name in para_list.keys():

        list_para_full = para_list[key_name].tolist()
        # list_para_full = [list_para_abs[0]] + list_para_abs + [list_para_abs[-1]]
        for k in range(common_variable('nzones', paths)):
            index_zone = k + 1
            if index_zone in list_zones_in_PFR:
                output_zones_quad_rz = PT_fun_zones_quad_rz(index_zone, paths)
                Rcorner = output_zones_quad_rz['Rcorner']
                Zcorner = output_zones_quad_rz['Zcorner']
                Rcenter = output_zones_quad_rz['Rcenter']
                Zcenter = output_zones_quad_rz['Zcenter']

                NPSIcorner = PT_fun_rz_to_npsi(Rcorner, Zcorner, paths)['NPSIcenter']
                NPSIcenter = PT_fun_rz_to_npsi(Rcenter, Zcenter, paths)['NPSIcenter']

                # print(NPSIcorner.max(), len(list_npsi_full), len(list_para_full))

                para_PFR = np.interp(NPSIcorner.max(), list_npsi_full, list_para_full)
                matrix_para_center = np.ones((NPSIcenter.shape[0], NPSIcenter.shape[1])) * para_PFR
            else:
                output_zones_quad_rz = PT_fun_zones_quad_rz(index_zone, paths)
                Rcenter = output_zones_quad_rz['Rcenter']
                Zcenter = output_zones_quad_rz['Zcenter']

                NPSIcenter = PT_fun_rz_to_npsi(Rcenter, Zcenter, paths)['NPSIcenter']

                matrix_para_center = np.interp(NPSIcenter, list_npsi_full, list_para_full)

            ########################## ballooning #############################
            if str(control_setup.ballooning) == 'on' and key_name != 'nu':
                output_zones_quad_rz = PT_fun_zones_quad_rz(index_zone, paths)
                Rcenter = output_zones_quad_rz['Rcenter']
                Zcenter = output_zones_quad_rz['Zcenter']
                list_B_center = interp2linear(common_variable('r', paths), common_variable('z', paths), Btotal_matrix,
                                              Rcenter,
                                              Zcenter)
                matrix_scal = B0 / np.array(list_B_center)
                matrix_scal_new = np.power(matrix_scal, b_exponent)
                matrix_para_center = matrix_scal_new * matrix_para_center

            if str(control_setup.div_diff_Pneu) == 'on' and key_name != 'nu':
                output_zones_quad_rz = PT_fun_zones_quad_rz(index_zone, paths)
                Rcenter = output_zones_quad_rz['Rcenter']
                Zcenter = output_zones_quad_rz['Zcenter']

                Coordinate_RZ = np.vstack((matrix_r.flatten(), matrix_z.flatten()))
                Coordinate_RZ_T = Coordinate_RZ.T
                matrix_pneu = matrix_pneu.flatten()

                Pneu_center = griddata(Coordinate_RZ_T, matrix_pneu, (Rcenter, Zcenter), method='linear')

                k_center = Pneu_center / pneu0
                pneu_gain = float(control_setup.div_Pneu_gain)
                max_k_center = max_pneu / pneu0

                k_matrix = (k_center * (pneu_gain * max_k_center - 1) + max_k_center * (1 - pneu_gain)) / (
                        max_k_center - 1)  # keep k in (1, pneu_gain * max_k_center)

                k_matrix_new = k_matrix.copy()
                k_matrix_new[k_matrix_new < 1] = 1
                k_matrix_new[np.isnan(k_matrix_new)] = 1

                matrix_para_center = matrix_para_center * k_matrix_new

            elif str(control_setup.div_diff_enhance) == 'on' and key_name != 'nu':
                ########################## special start #############################
                if index_zone in [5, 1, 2, 8]:
                    gain = float(control_setup.div_gain)
                    bias = float(control_setup.div_bias)
                    if index_zone == 8:
                        type1 = 2
                        type2 = 2
                    elif index_zone == 2:
                        type1 = 4
                        type2 = 1
                    elif index_zone == 1:
                        type1 = 4
                        type2 = 2
                    elif index_zone == 5:
                        type1 = 2
                        type2 = 1
                    else:
                        raise Exception('wrong index_zone')

                    nonlinear_matrix = generate_nonlinear_matrix(matrix_para_center, gain, bias, type1, type2)

                    matrix_para_center = matrix_para_center * nonlinear_matrix

                ########################## special end #############################
                # if key_name == 'chie' and index_zone == 8:
                #     test1 = matrix_para_center
                #     test2 = nonlinear_matrix
                #     # test3 = matrix_para_center_1

            # ########################## ballooning #############################
            # if str(variables['ballooning']) == 'on' and key_name != 'nu':
            #     output_zones_quad_rz = PT_fun_zones_quad_rz(index_zone)
            #     Rcenter = output_zones_quad_rz['Rcenter']
            #     Zcenter = output_zones_quad_rz['Zcenter']
            #     list_B_center = interp2linear(common_variable('r'), common_variable('z'), Btotal_matrix, Rcenter, Zcenter)
            #     matrix_scal = B0 / np.array(list_B_center)
            #     matrix_scal_new = np.power(matrix_scal, b_exponent)
            #     matrix_para_center = matrix_scal_new * matrix_para_center

            ########################## reshape #############################
            matrix_para_center = np.reshape(matrix_para_center,
                                            (matrix_para_center.shape[0], matrix_para_center.shape[1], 1))

            if key_name == 'chie':
                write_diff_h5('zone' + str(index_zone) + '/chie', matrix_para_center, paths)
            else:
                for ind in range(number_of_species):
                    write_diff_h5('zone' + str(index_zone) + '/spec' + str(ind + 1) + '/' + key_name,
                                  matrix_para_center, paths)


def update_diff_map(paths):
    list_r_sim_intersection, list_D = profile_feedback_function('ne', paths)

    input = {'position_name': 'OMPS'}
    output = PT_fun_positions(input, paths)
    r_omps = float(output['list_r'])

    list_r_sim_intersection_relative = list_r_sim_intersection - r_omps

    input = {'operation': 'renew', 'list': list_r_sim_intersection_relative,
             'file_name': paths.profile_feedback_list_r_file_path}
    read_or_save_pd(input, paths)

    input = {'operation': 'renew', 'list': list_D, 'file_name': paths.profile_feedback_list_d_file_path}
    read_or_save_pd(input, paths)

    # variables = read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)

    if control_setup.given_profile == 'n_Te':
        list_r_sim_intersection, list_chie = profile_feedback_function('Te', paths)

        input = {'operation': 'renew', 'list': list_chie, 'file_name': paths.profile_feedback_list_chie_file_path}
        read_or_save_pd(input, paths)

    else:
        list_chie = list_D / 0.3 * 1

    generate_diff_map(list_r_sim_intersection_relative, list_D, list_chie, paths)

    run_command(paths.controlled_code_folder_path, f"cp {paths.diffusion_file_path} {paths.mesh_folder_path}")
    run_command(paths.controlled_code_folder_path, f"mv {paths.diffusion_file_path} {paths.rundir_folder_path}")


def Restart_check(paths):
    print('################ check restart  #######################')
    print('Restart checking...')
    if read_modify_variable_in_para_file(paths.param_raptorx_file_path, "restart") == 0:
        print('First iteration finished, switch to restart mode...')
        read_modify_variable_in_para_file(paths.param_raptorx_file_path, "restart", new_value="1")
    else:
        print('Already in restart mode, no modification.')

    print('checking complete')


def standard_deviation_converged(paths):
    standard_deviation_value = standard_deviation(paths)
    control_setup = load_module(paths.control_setup_file_path)
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    target_std_factor = float(control_setup.target_std_factor)
    standard_deviation_converged_value = standard_deviation_value * target_std_factor
    return standard_deviation_converged_value


def Convergence_check(paths):
    # variables = DP_Setup_RW.read_setup('DP_setup.txt')
    control_setup = load_module(paths.control_setup_file_path)
    df_DP_Trace_Data = pd.read_csv(paths.trace_data_file_path)
    number_of_samples = int(control_setup.number_of_samples)
    active_feedback = str(control_setup.active_feedback)
    target_para = str(control_setup.target_para)

    time_simu = float(df_DP_Trace_Data['time'].iloc[-1])

    puff_rate = read_active_puff(paths)

    error_mean_factor = float(control_setup.error_mean_factor)

    iteration_number = int(df_DP_Trace_Data['iteration_number'].iloc[-1])
    last_sample_value = float(df_DP_Trace_Data[target_para].iloc[-1])

    if iteration_number >= number_of_samples:
        target_sample_list = df_DP_Trace_Data[target_para].iloc[int(-1 * number_of_samples):]
        if active_feedback == 'on':
            target_value = float(control_setup.Target_value)
        else:
            target_value = np.mean(target_sample_list)
    else:
        target_value = 0

    target_error_mean_converged = abs(target_value / error_mean_factor)
    target_value_std_converged = standard_deviation_converged(paths)
    volatility_of_last_sample_converged = float(control_setup.last_sample_vol)

    if iteration_number >= number_of_samples:
        puff_rate_list = df_DP_Trace_Data['puff_rate'].iloc[int(-1 * number_of_samples):]
        target_sample_list = df_DP_Trace_Data[target_para].iloc[int(-1 * number_of_samples):]
        puff_rate_mean = np.mean(puff_rate_list)
        target_value_std = np.std(target_sample_list, ddof=1)
        target_error_mean = abs(np.mean(target_sample_list) - target_value)
        sample_range = float(abs(max(target_sample_list) - min(target_sample_list)))
        volatility_of_last_sample = float(abs(last_sample_value - target_value)) / sample_range

        convergence_marker = 0
        if target_value_std <= target_value_std_converged:
            if target_error_mean <= target_error_mean_converged:
                convergence_marker = 1

    else:
        puff_rate_mean = puff_rate
        target_value_std = target_value_std_converged * 2
        target_error_mean = target_error_mean_converged * 2
        volatility_of_last_sample = volatility_of_last_sample_converged * 2
        convergence_marker = 0

    noise_amplitude = amplitude(paths)

    dataframe = pd.DataFrame({'iteration_number': iteration_number,
                              'time': time_simu,
                              'noise_amplitude': noise_amplitude,
                              'puff_rate': puff_rate,
                              'puff_rate_mean': puff_rate_mean,
                              'last_sample_value': last_sample_value,
                              'target_value': target_value,
                              'target_value_std_converged': target_value_std_converged,
                              'target_error_mean_converged': target_error_mean_converged,
                              'Volatility_of_last_sample_converged': volatility_of_last_sample_converged,
                              'target_value_std': target_value_std,
                              'target_error_mean': target_error_mean,
                              'Volatility_of_last_sample': volatility_of_last_sample,
                              'convergence_marker': convergence_marker,
                              'iteration_marker': 1}, index=[0])

    if not os.path.exists(paths.convergence_checker_file_path):
        dataframe.to_csv(paths.convergence_checker_file_path, mode='a', index=None)
    else:
        dataframe.to_csv(paths.convergence_checker_file_path, mode='a', header=False, index=None)

    df_DP_convergence_checker = pd.read_csv(paths.convergence_checker_file_path)

    number_analyse = int(control_setup.number_matched)

    if len(df_DP_convergence_checker) >= number_analyse:
        convergence_marker_list = df_DP_convergence_checker['convergence_marker'].iloc[-int(number_analyse):]
        total_value = sum(convergence_marker_list)
    else:
        total_value = 0

    iteration_marker_list = df_DP_convergence_checker['iteration_marker'].iloc[:]
    total_iteration_times = sum(iteration_marker_list)

    if total_value == number_analyse and volatility_of_last_sample <= volatility_of_last_sample_converged and total_iteration_times >= (
            number_of_samples + number_analyse):
        result = 'converged'
    else:
        result = 'not_converged'
    return result


if __name__ == "__main__":
    pass
