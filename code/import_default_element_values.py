#!/usr/bin/env python
# coding: utf-8

def import_default_element_values():
    element_color_map = {}
    element_radius_map = {}
    with open("../data/element_default_values.txt", "r") as element_default_value_database:
        lines =element_default_value_database.readlines()
        for line in lines:
            # remove comments and break apart by spaces
            line = line.split("#")[0].split()
            if line:
                atomic_number = int(line[0])
                atomic_symbol = line[1]
                atomic_radius = float(line[2])
                van_der_Waals_radius = float(line[3])
                ionic_radius = float(line[4])
                default_color = (float(line[5]), float(line[6]), float(line[7]))
                element_radius_map[atomic_symbol] = atomic_radius
                element_color_map[atomic_symbol] = default_color
    return element_color_map, element_radius_map