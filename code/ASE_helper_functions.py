#!/usr/bin/env python
# coding: utf-8

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.calculators.singlepoint import SinglePointCalculator


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

def assign_energy(structs, energy_keyword):
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