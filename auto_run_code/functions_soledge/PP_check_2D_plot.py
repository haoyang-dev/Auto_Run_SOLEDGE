import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from auto_run_code.functions_soledge.function2_intermediate import PathSummary, PT_fun_matrix_general, Prad_total_int_lot

# user define, read
paths = PathSummary('D:\\Academic_Research\\My_Work\\Simulation\\SOLEDGE_3X\\S3XE_Control_Scripts\\Development')
matrix_type = 'tri'

# input_matrix_para_define = {'para_name': 'Prad_total', 'matrix_type': 'tri'}
output_matrix = PT_fun_matrix_general(para_name='Prad_total', matrix_type='tri', paths=paths) #angle_relative_to_LOT
Prad_total_matrix_r = output_matrix.r
Prad_total_matrix_z = output_matrix.z
Prad_total_matrix_data = output_matrix.value

# Create a figure and axis
# fig, ax = plt.subplots()

x = Prad_total_matrix_r
y = Prad_total_matrix_z
z = Prad_total_matrix_data

# Create a list of polygons with their corresponding colors
polygons = []
colors = []

if matrix_type == 'quad':
    number_define = 4
elif matrix_type == 'tri':
    number_define = 3
else:
    raise ValueError('matrix_type must be either quad or tri')

for i in range(len(x)):
    # Create a polygon for each quad cell
    poly = [(x[i, j], y[i, j]) for j in range(number_define)]
    polygons.append(poly)

    # Calculate a representative color value for the cell, e.g., mean of the z-values
    color_value = np.mean(z[i])
    colors.append(color_value)

# Create a figure and axis
fig, ax = plt.subplots()

# Create a PolyCollection from the list of polygons
collection = PolyCollection(polygons, array=np.array(colors), cmap='viridis', edgecolor='none')

# Add the collection to the axis
ax.add_collection(collection)

# Auto-scale the axis to fit the content
ax.autoscale_view()

# Add a colorbar
fig.colorbar(collection, ax=ax, label='Z-value')

# Set axis labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# fig.savefig('high_res_plot.png', dpi=600, bbox_inches='tight')


# Show the plot
plt.show()


print(Prad_total_int_lot(paths))


# output = peak_value_in_field(para_name='Prad_total', matrix_type='tri', paths=paths)
#
# ax.plot(output.r, output.z, 'r+', label='R vs Z')
#
# # ax.legend()
#
# plt.show()
#
# print(radiator_distance(paths))

# input = {'para_name': 'qpara', 'matrix_type': 'quad', 'position_name': 'LOT', 'location': 'WALL',
#                  'data_treatment': 'MAXABS'}
# output = PT_fun_data(input, paths)
#
# print(output['data_result'])