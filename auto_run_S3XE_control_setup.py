import numpy as np
################ Data Extraction ################
Trace_mode        = 'WEST'                # BASIC / PRO / WEST
num_interval      = 1                     # Save Plasma.h5 file every fixed number of first loop: [int]
cache_limit       = 10                    # Save only the files of last few loops, if negative, save all. [int]

################ Para Feedback control ################
active_feedback   = 'off'                  # on/off (can only be actived under Trace_mode: WEST or PRO)
control_method    = 'PID'                 # PID
target_para       = 'OMPS_ne_FLOAT'       # OMPS_ne_FLOAT or LOT_Te_MAXABS or LOT_Ge_MAXABS or LSN_radiator_distance (LSN_radiator_distance only supproted in WEST)
Target_value      = 2.47e19

Puff_FD_index     = 1                     # define the index of gas puff that used to control
Lock_ratio        = 'off'                 # on / off Lock gas composition ratio

K_p               = 2300                  # proportional gain, used by both methods
K_i               = 4.0e4                 # integral gain
K_d               = 0                     # derivative gain, used by both methods
integral_bg_puff  = None                  # integral_term + puff background, if None, the gas puff rate will be used for the first run
max_puff          = 5.0e22
min_puff          = 0
time_delay        = 0                     # time delay of gas puff in s
t_buffer          = 0                     # buffer time
amplitude_factor  = 2.5                   # noise_amplitude_value = noise_standard_deviation_value * amplitude_factor

rollover_switch   = 'off'                 # on / off
rollover_check    = 'BYGE'                # rollover checking mode, NONE or BYGE
rollover_Ge_puff  = 2.5e21                # puff value when rollover happens
rollover_ne       = 2.28e19               # upstream density when rollover happens
max_Ge_exp        = 1.11e24               # peak Ge value when rollover happens

target_map        = 'off'                 # on/off
map_t_difference  = None                  # Tmap - Tsim
target_map_t_list = np.array([0, 1, 2, 3, 4, 5])
target_map_v_list = np.array([0, 1, 2, 3, 4, 5])*1e19

################ Profile feedback ################
profile_feedback  = 'off'                 # on/off
diff_fd_kp        = 0                     # safe to be 0
diff_fd_ki        = 0                     # 10 maybe the best
given_profile     = 'n'                   # n / n_Te
diff_nu           = 0.3                   # H-mode 0.2, L-mode 0.3

chi_special       = 'off'                 # chi_i inside sep = chie * chi_chie_ratio
chi_chie_ratio    = 2                     #  the ratio inside sep

ballooning        = 'off'                 # on / off
b_exponent        = 1                     # 0-5
omp_diff_fix      = 'on'                  # on / off

div_diff_Pneu     = 'off'                 # div_diff_Pneu and div_diff_enhance can use only one at the same time
div_Pneu_min_scal = 5                     # Set Pneu0 = mean(Pneu,outeromps) * div_Pneu_min_scal
div_Pneu_gain     = 1                     # k = Pneu/Pneu0, D = (k*(gain*max(k)-1)+max(k)*(1-gain))/(max(k) - 1)*D0, D = D0 when Pneu <= Pneu0

div_diff_enhance  = 'off'                 # on / off, TCV H-mode only
div_gain          = 20
div_bias          = 1.5

################ Gas puff control ################
gas_puff_map      = 'off'                 # on/off, when turning on this function, set active_feedback  = off
gas_map_dt        = None                  # Tmap - Tsim
gas_map_t_list    = np.array([0, 1, 2, 3, 4, 5])
gas_map_puff1_list= np.array([0, 1, 2, 3, 4, 5])*1e21  # puff rate
gas_map_puff2_list= None                               # if second puff index is available, it needs to be defined

################ input power control ################
input_power_map   = 'off'                 # on/off, when turning on this function, set active_feedback  = off
power_map_dt      = None                  # Tmap - Tsim
power_map_t_list  = np.array([0, 1, 2, 3, 4, 5])
power_map_in1_list= np.array([0, 1, 2, 3, 4, 5])*1e5  # electron input power [W]
power_map_in2_list= np.array([0, 1, 2, 3, 4, 5])*1e5  # ion input power [W]
power_map_in3_list= np.array([0, 0, 0, 0, 0, 0])*1e5  # if impurities exits, it needs to be defined

################ Convergence ################
check_convergence = 'off'                 # on/off
number_of_samples = 20
number_matched    = 6
error_mean_factor = 1e4                   # target_error_mean_converged = target_value/error_mean_factor (2e3 or 1e4)
target_std_factor = 1.2                   # standard_deviation_converged_value = noise_standard_deviation_value * target_std_factor
last_sample_vol   = 0.25                  # the ratio of the error of the last sample to the fluctuation of all the samples, 0 - 0.5

############### Self define position for PRO mode ################
point1_name           = 'OMPS'
point1_para_name      = 'ne'
point1_data_treatment = 'FLOAT'
point1_location       = 'INNER'
point1_matrix         = 'quad'
point1_r              = 2.96913318
point1_z              = 0.01507538

line1_name            = 'LOT'
line1_para_name       = 'ne'
line1_data_treatment  = 'MAXABS'
line1_location        = 'WALL'
line1_matrix          = 'quad'
line1_r_start         = 2.1919
line1_z_start         = -0.6938
line1_r_end           = 2.443147993067367
line1_z_end           = -0.795232879591286
line1_num             = 50
