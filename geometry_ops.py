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

def rotate_point(p, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = p

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def generate_grid_infill(contour, density=20, max_line_distance=10, min_line_distance=1, rotation_angle=0):
    # Ensure density is between 0 and 100
    density = max(0, min(100, density))

    if density == 0:
        return []

    # Calculate the adjusted line distance based on the density
    adjusted_distance = ((100 - density) * max_line_distance + density * min_line_distance) / 100

    # Calculate the bounding box for the contour
    min_x = min(pt[0] for seg in contour for pt in seg)
    max_x = max(pt[0] for seg in contour for pt in seg)
    min_y = min(pt[1] for seg in contour for pt in seg)
    max_y = max(pt[1] for seg in contour for pt in seg)

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

# Cohen-Sutherland line clipping
OUT_LEFT, OUT_RIGHT, OUT_BOTTOM, OUT_TOP = 1, 2, 4, 8

def compute_out_code(x, y, xmin, ymin, xmax, ymax):
    code = 0
    if x < xmin:
        code |= OUT_LEFT
    elif x > xmax:
        code |= OUT_RIGHT
    if y < ymin:
        code |= OUT_BOTTOM
    elif y > ymax:
        code |= OUT_TOP
    return code

def clip_line_to_bbox(line, bbox):
    x0, y0 = line[0]
    x1, y1 = line[1]
    xmin, ymin, xmax, ymax = bbox

    outcode0 = compute_out_code(x0, y0, xmin, ymin, xmax, ymax)
    outcode1 = compute_out_code(x1, y1, xmin, ymin, xmax, ymax)
    accept = False

    while True:
        if not (outcode0 | outcode1):
            accept = True
            break
        elif outcode0 & outcode1:
            break
        else:
            outcode_out = outcode0 if outcode0 else outcode1
            if outcode_out & OUT_TOP:
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif outcode_out & OUT_BOTTOM:
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif outcode_out & OUT_RIGHT:
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            elif outcode_out & OUT_LEFT:
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin

            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_out_code(x0, y0, xmin, ymin, xmax, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_out_code(x1, y1, xmin, ymin, xmax, ymax)

    return ((x0, y0), (x1, y1)) if accept else None


def clip_lines_to_contour(lines, contour):
    bbox = (min(pt[0] for seg in contour for pt in seg),
            min(pt[1] for seg in contour for pt in seg),
            max(pt[0] for seg in contour for pt in seg),
            max(pt[1] for seg in contour for pt in seg))

    clipped = []
    for line in lines:
        clipped_line = clip_line_to_bbox(line, bbox)
        if clipped_line:
            clipped.append(clipped_line)

    return clipped

