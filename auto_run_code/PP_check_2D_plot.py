import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from functions_soledge.function2_intermediate import PT_fun_matrix_general, Path_PP

run_dir_folder_path = 'D:\\Academic_Research\\My_Work\\Simulation\\SOLEDGE_3X\\S3XE_Control_Scripts\\Development\\run_dir'
plasma_file_path = 'D:\\Academic_Research\\My_Work\\Simulation\\SOLEDGE_3X\\S3XE_Control_Scripts\\Development\\run_dir\\plasmaRestart.h5'

paths = Path_PP(run_dir_folder_path, plasma_file_path)
matrix_type = 'tri'

output_matrix = PT_fun_matrix_general(para_name='Prad_total', matrix_type='tri', paths=paths) #angle_relative_to_LOT
Prad_total_matrix_r = output_matrix.r
Prad_total_matrix_z = output_matrix.z
Prad_total_matrix_data = output_matrix.value

x = Prad_total_matrix_r
y = Prad_total_matrix_z
z = Prad_total_matrix_data

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
ax.set_aspect('equal')

# Add a colorbar
fig.colorbar(collection, ax=ax, label='Prad')

# Set axis labels
ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')

plt.show()
