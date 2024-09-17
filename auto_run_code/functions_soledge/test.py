# # from functions_soledge.functions_soledge import read_modify_variable_in_para_file
# #
# # hh = read_modify_variable_in_para_file("param_raptorXfg.txt", "iniPerturb", object())
# #
# # print(hh)
# #
# # # read_modify_variable_in_para_file("param_raptorXfg.txt", "iniPerturb", None)
#
#
# from functions_soledge import count_number_of_line_and_point
#
# print(count_number_of_line_and_point("auto_run_S3XE_control_setup.py", "point"))
from functions_soledge import PT_fun_data, PathSummary

paths = PathSummary("D:\\Academic_Research\\My_Work\\Simulation\\SOLEDGE_3X\\4_Control_Scripts")

input = {'para_name': 'ne', 'matrix_type': 'quad', 'position_name': 'OMPS', 'location': 'INNER',
                 'data_treatment': 'FLOAT'}
output = PT_fun_data(input, paths)
OMPS_ne_FLOAT = output['data_result']