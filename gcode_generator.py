import numpy as np
from geometry_ops import (generate_grid_infill)
def generate_gcode_header():
    gcode = []
    gcode.append("; BEGIN HEADER")
    gcode.append("M190 S60 ; set bed temperature and wait for it to be reached")
    gcode.append("M104 S200 ; set extruder temperature")
    gcode.append("M109 S200 ; wait for extruder temperature")
    gcode.append("G21 ; set units to millimeters")
    gcode.append("G90 ; use absolute coordinates")
    gcode.append("G92 E0 ; zero the extruder")
    gcode.append("G28 ; home all axes")
    gcode.append("; END HEADER\n")
    return gcode

def generate_gcode_footer():
    gcode = []
    gcode.append("; BEGIN FOOTER")
    gcode.append("M104 S0 ; turn off extruder temperature")
    gcode.append("M140 S0 ; turn off bed temperature")
    gcode.append("G28 ; home all axes")
    gcode.append("M84 ; disable motors")
    gcode.append("; END FOOTER\n")
    return gcode


def layer_to_gcode(z, contour, feed_rate=1500, extrusion_rate=0.05):
    gcode = []
    gcode.append(f"; LAYER {z}")
    
    for segment in contour:
        # Check if the segment has two points. If not, print a warning and skip it.
        if len(segment) != 2:
            print("Problematic segment:", segment)
            continue  # Skip processing this segment
        
        start, end = segment
        
        # Move to the start of the segment without extruding
        gcode.append(f"G0 X{start[0]:.2f} Y{start[1]:.2f} Z{z:.2f} F{feed_rate}")

        # Compute the length of the segment for extrusion
        length = np.linalg.norm(np.array(end) - np.array(start))
        extrusion = length * extrusion_rate

        # Move to the end of the segment while extruding
        gcode.append(f"G1 X{end[0]:.2f} Y{end[1]:.2f} E{extrusion:.2f} F{feed_rate}")
    
    gcode.append("\n")
    return gcode


def generate_gcode(contours, infill_density=20):
    gcode = generate_gcode_header()

    for z, contour in contours.items():
        gcode.extend(layer_to_gcode(z, contour))

        # Add infill
        infill = generate_grid_infill(contour, density=infill_density)
        gcode.extend(layer_to_gcode(z, infill))

    gcode.extend(generate_gcode_footer())
    
    return gcode
