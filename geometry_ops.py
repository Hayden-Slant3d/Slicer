import numpy as np
from pykdtree.kdtree import KDTree

def bounding_box(triangles):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for triangle_data in triangles:
        triangle, _ = triangle_data
        for v in [triangle.v1, triangle.v2, triangle.v3]:
            x, y, z = v
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def surface_area(triangles):
    total_area = 0
    for triangle_data in triangles:
        triangle, _ = triangle_data
        v1, v2, v3 = np.array(triangle.v1), np.array(triangle.v2), np.array(triangle.v3)
        total_area += np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2
    return total_area

def get_triangle_data(triangle):
    min_x = min(triangle.v1[0], triangle.v2[0], triangle.v3[0])
    max_x = max(triangle.v1[0], triangle.v2[0], triangle.v3[0])
    
    min_y = min(triangle.v1[1], triangle.v2[1], triangle.v3[1])
    max_y = max(triangle.v1[1], triangle.v2[1], triangle.v3[1])
    
    min_z = min(triangle.v1[2], triangle.v2[2], triangle.v3[2])
    max_z = max(triangle.v1[2], triangle.v2[2], triangle.v3[2])
    
    bbox = (min_x, max_x, min_y, max_y, min_z, max_z)
    return triangle, bbox


def build_kdtree(triangles):
    centroids = []
    for triangle_data in triangles:
        triangle, _ = triangle_data
        centroid = np.mean([triangle.v1, triangle.v2, triangle.v3], axis=0)
        centroids.append(centroid)
    return KDTree(np.array(centroids))


def triangles_overlap(triangle1, triangle2):
    bbox1 = get_triangle_data(triangle1)[1]
    bbox2 = get_triangle_data(triangle2)[1]
    
    if bbox1[1] < bbox2[0] or bbox1[0] > bbox2[1] or bbox1[3] < bbox2[2] or bbox1[2] > bbox2[3] or bbox1[5] < bbox2[4] or bbox1[4] > bbox2[5]:
        return False

    # More detailed overlap check can be added if necessary
    return True

def has_nearby_overlaps(kdtree, triangle, triangles):
    centroid = np.mean([triangle.v1, triangle.v2, triangle.v3], axis=0)
    
    # Get indices of neighboring triangles
    _, indices = kdtree.query(centroid.reshape(1, -1))

    for index in indices:
        candidate = triangles[index]
        if triangles_overlap(triangle, candidate[0]):  # Now calling the detailed overlap check function
            return True
    return False


def check_mesh_integrity(triangles):
    kdtree = build_kdtree(triangles)
    edges = set()
    for triangle_data in triangles:
        triangle, _ = triangle_data
        for vertex_pair in [(triangle.v1, triangle.v2), (triangle.v2, triangle.v3), (triangle.v1, triangle.v3)]:
            edge = tuple(sorted(vertex_pair))
            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)

    if len(edges) != 0:
        return False  # Unpaired edges detected

    for i, triangle_data in enumerate(triangles):
        triangle, _ = triangle_data
        if has_nearby_overlaps(kdtree, triangle, triangles):  # Using KDTree-based overlap check
            return False  # Overlapping triangles detected
    return True

def generate_slices(min_z, max_z, layer_height):
    z_values = np.arange(min_z, max_z, layer_height)
    return z_values

def triangle_plane_intersection(triangle_data, z):
    triangle, (min_z_triangle, max_z_triangle) = triangle_data

    # If z is outside the range, immediately return an empty list
    if not (min_z_triangle <= z <= max_z_triangle):
        return []

    intersections = []

    for i in range(3):
        vertices = [triangle.v1, triangle.v2, triangle.v3]
        point1 = vertices[i]
        point2 = vertices[(i + 1) % 3]


        # Check if edge intersects the plane
        if min(point1[2], point2[2]) <= z <= max(point1[2], point2[2]) and point1[2] != point2[2]:
            ratio = (z - point1[2]) / (point2[2] - point1[2])
            x = point1[0] + ratio * (point2[0] - point1[0])
            y = point1[1] + ratio * (point2[1] - point1[1])
            intersections.append((x, y))

    return intersections

def generate_contours(triangles, z_values):
    contours = {}
    for z in z_values:
        contour = []
        for triangle_data in triangles:
            intersections = triangle_plane_intersection(triangle_data, z)
            if len(intersections) == 2:
                contour.append(intersections)
        contours[z] = contour

    return contours

def generate_grid_infill(contour, density=20, line_distance=2, rotation_angle=0):
    # Calculate the bounding box for the contour
    min_x = min(pt[0] for seg in contour for pt in seg)
    max_x = max(pt[0] for seg in contour for pt in seg)
    min_y = min(pt[1] for seg in contour for pt in seg)
    max_y = max(pt[1] for seg in contour for pt in seg)

    # Adjust the line distance based on the density
    adjusted_distance = line_distance / (density / 100)

    # Generate grid lines
    vertical_lines = [((x, min_y), (x, max_y)) for x in np.arange(min_x, max_x, adjusted_distance)]
    horizontal_lines = [((min_x, y), (max_x, y)) for y in np.arange(min_y, max_y, adjusted_distance)]

    # Rotate grid lines if rotation_angle is provided
    if rotation_angle != 0:
        cos_angle = np.cos(np.radians(rotation_angle))
        sin_angle = np.sin(np.radians(rotation_angle))

        rotate = lambda p: (
            cos_angle * (p[0] - min_x) - sin_angle * (p[1] - min_y) + min_x,
            sin_angle * (p[0] - min_x) + cos_angle * (p[1] - min_y) + min_y
        )

        vertical_lines = [(rotate(start), rotate(end)) for start, end in vertical_lines]
        horizontal_lines = [(rotate(start), rotate(end)) for start, end in horizontal_lines]

    # Clip the grid lines to the contour
    vertical_lines_clipped = clip_lines_to_contour(vertical_lines, contour)
    horizontal_lines_clipped = clip_lines_to_contour(horizontal_lines, contour)

    return vertical_lines_clipped + horizontal_lines_clipped


def intersection_of_two_lines(line1, line2):
    # Lines are defined as ((x1, y1), (x2, y2))
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Compute determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if det == 0:  # Lines are parallel and won't intersect
        return None

    # Compute the intersection point using Cramer's rule
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    
    # Now, we need to check if the intersection point is within the segments
    if (min(x1, x2) <= px <= max(x1, x2) and
        min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and
        min(y3, y4) <= py <= max(y3, y4)):
        return px, py
    else:
        return None

def clip_lines_to_contour(lines, contour):
    clipped = []
    for line in lines:
        intersections = [intersection_of_two_lines(line, segment) for segment in contour]
        valid_points = [point for point in intersections if point is not None]
        
        if len(valid_points) == 2:
            clipped.append(tuple(valid_points))
    return clipped

