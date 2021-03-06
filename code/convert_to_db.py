#!/usr/bin/env python3

import argparse
import os.path as path
import os
import ase.db
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

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

parser = argparse.ArgumentParser()
parser.add_argument('-efb', '--existing_file_behavior', type=str, default="skip", choices = ["skip", "overwrite", "append"], help='Specifies what to do when a requested database file already exists. This flag needs to come before all filenames.')
parser.add_argument('-aek', '--alt_energy_keyword', type=str, help='If you have energies stored using a keyword other than "energy", specify that here. This flag needs to come before all filenames.')
parser.add_argument('filenames', nargs=argparse.REMAINDER, help="List of filenames to convert. Filenames must be last argument of this command.")
cmdline_args = parser.parse_args()

script_path = path.dirname(path.realpath(__file__))
data_path = path.dirname(script_path) + "/data"
relpath = path.relpath(data_path)

i=1
for filename_in in cmdline_args.filenames:
    if len(cmdline_args.filenames) > 1:
        print("--{}--".format(i))
        i += 1
    if "/" not in filename_in:
        filename_in = data_path + "/" + filename_in

    filename_out = relpath + "/" + filename_in.split("/")[-1].split(".")[0] + ".db"

    print("Attempt to create a ASE database at {} from {}".format(filename_out, filename_in))
    
    if path.exists(filename_out) and cmdline_args.existing_file_behavior == "skip":
        print("Skipping: {} already exists. Will not overwrite {} or append. Will not create any database from {}. \
              \n          To overwrite existing database, rerun this script with '--existing_file_behavior overwrite'. \
              \n          To append to existing database, rerun this script with '--existing_file_behavior append'.".format( filename_out, filename_out, filename_in) )        
    else:
        if path.exists(filename_out) and cmdline_args.existing_file_behavior == "append":
            print("{} already exists. Appending new data from {} to existing database.".format(filename_out, filename_in))
        elif path.exists(filename_out) and cmdline_args.existing_file_behavior == "overwrite":
            print("{} already exists. Deleting pre-existing database and replacing with new database from {}.".format(filename_out, filename_in))
            os.remove(filename_out)
        print("Importing data from {}".format(filename_in), end=" ... ")
        Atoms_objects = ase.io.read(filename_in, ":")
        print("cmdline_args.alt_energy_keyword =", cmdline_args.alt_energy_keyword)
        if cmdline_args.alt_energy_keyword:
            print("Assigning energies using the keyword '{}'".format(cmdline_args.alt_energy_keyword))
            Atoms_objects = assign_precalculated_energy(Atoms_objects, cmdline_args.alt_energy_keyword)
        print("Exporting data to {}".format(filename_out))
        db = ase.db.connect(filename_out)
        for Atoms_object in Atoms_objects:
            db.write(Atoms_object)
        print("Success! {} created.".format(filename_out))
