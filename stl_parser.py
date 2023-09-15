import struct
from collections import namedtuple

Triangle = namedtuple('Triangle', ['normal', 'v1', 'v2', 'v3'])

def read_binary_stl(filename):
    with open(filename, 'rb') as file:
        header = file.read(80)
        triangle_count = struct.unpack('<I', file.read(4))[0]
        triangles = []

        for _ in range(triangle_count):
            normal = struct.unpack('<fff', file.read(12))
            vertex1 = struct.unpack('<fff', file.read(12))
            vertex2 = struct.unpack('<fff', file.read(12))
            vertex3 = struct.unpack('<fff', file.read(12))
            min_z_triangle = min(vertex1[2], vertex2[2], vertex3[2])
            max_z_triangle = max(vertex1[2], vertex2[2], vertex3[2])


            attr_byte_count = struct.unpack('<H', file.read(2))
            triangle = Triangle(normal, vertex1, vertex2, vertex3)
            triangles.append((triangle, (min_z_triangle, max_z_triangle)))

    return triangles