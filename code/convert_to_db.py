#!/usr/bin/env python3

import argparse
import os.path as path
import os
import ase.db
import ase.io

parser = argparse.ArgumentParser()
parser.add_argument('-efb', '--existing_file_behavior', type=str, default="skip", choices = ["skip", "overwrite", "append"], help='Specifies what to do when a requested database file already exists. This flag needs to come before all filenames.')
parser.add_argument('filenames', nargs=argparse.REMAINDER, help="List of filenames to convert")
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
        print("Exporting data to {}".format(filename_out))
        db = ase.db.connect(filename_out)
        for Atoms_object in Atoms_objects:
            db.write(Atoms_object)
        print("Success! {} created.".format(filename_out))