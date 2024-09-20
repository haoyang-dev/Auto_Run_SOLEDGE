
run_dir_folder_path = 'D:\Academic_Research\My_Work\Simulation\SOLEDGE_3X\S3XE_Control_Scripts\Development\run_dir';
plasma_file_path = 'D:\Academic_Research\My_Work\Simulation\SOLEDGE_3X\S3XE_Control_Scripts\Development\run_dir\plasmaRestart.h5';

paths = py.functions_soledge.function2_intermediate.Path_PP(run_dir_folder_path, plasma_file_path);

output_general = py.functions_soledge.function2_intermediate.PT_fun_matrix_general("Prad_total", "tri", paths);

% Extract the results from the output
Prad_total_matrix_r = output_general.r;
Prad_total_matrix_z = output_general.z;
Prad_total_matrix_data = output_general.value;

% Convert Python objects to MATLAB arrays if necessary
Prad_total_matrix_r = double(Prad_total_matrix_r);
Prad_total_matrix_z = double(Prad_total_matrix_z);
Prad_total_matrix_data = double(Prad_total_matrix_data);

patch(Prad_total_matrix_r',Prad_total_matrix_z',Prad_total_matrix_data');
shading flat
axis equal

