from stl_parser import read_binary_stl
from geometry_ops import (bounding_box, surface_area, check_mesh_integrity, generate_slices, generate_contours)
from gcode_generator import generate_gcode

triangles = read_binary_stl("xyzCalibration_cube.stl")
# triangles = read_binary_stl("3D BENCHY V2.STL")
print("Bounding Box:", bounding_box(triangles))
print("Surface Area:", surface_area(triangles))
print("Mesh Integrity:", check_mesh_integrity(triangles))

bounding_info = bounding_box(triangles)
layer_height = 0.2
z_slices = generate_slices(bounding_info[0][2], bounding_info[1][2], layer_height)
contours = generate_contours(triangles, z_slices)
# For debugging
# for z, contour in contours.items():
#     print(f"Contour at Z={z}:")
#     for segment in contour:
#         print(segment)

gcode_list = generate_gcode(contours, infill_density=100)

with open("output.gcode", "w") as file:
    for line in gcode_list:
        file.write(line + "\n")
