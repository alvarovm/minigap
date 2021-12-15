#!/usr/bin/env python
# coding: utf-8

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Rydberg, kJ, Hartree, mol, kcal
import numpy as np


def assign_calc(molecule, calc_type):    
    if calc_type == "EMT":
        molecule.calc = EMT()
    elif calc_type == "LJ":
        molecule.calc = LennardJones()
    elif calc_type == "Morse":
        molecule.calc = MorsePotential()
    else:
        print("This function does not recognize '{}' as a currently supported Atoms calculator type".format(calc_type))
        return
    return molecule

def assign_precalculated_energy(structs, energy_keyword):
    if isinstance(structs, Atoms):
        structs = [structs]
    elif not isinstance(structs[0], Atoms):
        raise TypeError("'structs' argument should be an Atoms object or iterable of Atoms objects")
    
    for i in range(len(structs)):
        struct = structs[i]
        if energy_keyword in struct.info:
            energy=struct.info[energy_keyword]
        else:
            raise KeyError("Could not find the '{}' field in the Atoms.info dictionary for structure #{}. Double check your input file.".format(energy_keyword, i))
        struct.calc = SinglePointCalculator(struct, energy=energy)
    return structs

def convert_energy(energy, units_in, units_out):
    # relative to 1 eV
    # using colby.edu/chemistry/PChem/Hartree.html where ASE didn't have it
    conversion_factors = {
        "eV": 1.0,
        "Ha": 1 / Hartree,
        "J": 1000/kJ,
        "kJ/mol": mol/kJ,
        "kcal/mol": mol/kcal,
        "Ry": 1/Rydberg,
        "K": 11604.588577015096
    }
    energy = np.array(energy)
    unrecognized_units = []
    if units_in != units_out:
        if units_in not in conversion_factors:
            unrecognized_units.append(units_in)
        if units_out not in conversion_factors:
            unrecognized_units.append(units_out)
        if len(unrecognized_units):
            units_error = "Do not recognize energy units: {}. Recognized energy units are: {}".format(unrecognized_units, list(conversion_factors.keys()))
            raise KeyError(units_error)
    
        energy = energy / conversion_factors[units_in] * conversion_factors[units_out]
    return energy
    
def convert_force(force, units_in, units_out):
    # relative to 1 eV/Å
    # using greif.geo.berkeley.edu/~driver/conversions.html as reference
    conversion_factors = {
        "eV/ang": 1.0,
        "Ry/Bohr": 1 / 25.71104309541616,
        "Ha/Bohr": 1 / 51.42208619083232,
        "N": 1e13/kJ
    }
    force = np.array(force)
    unrecognized_units = []
    if units_in != units_out:
        if units_in not in conversion_factors:
            unrecognized_units.append(units_in)
        if units_out not in conversion_factors:
            unrecognized_units.append(units_out)
        if len(unrecognized_units):
            units_error = "Do not recognize force units: {}. Recognized force units are: {}".format(unrecognized_units, list(conversion_factors.keys()))
            raise KeyError(units_error)
    
        force = force / conversion_factors[units_in] * conversion_factors[units_out]
    return force