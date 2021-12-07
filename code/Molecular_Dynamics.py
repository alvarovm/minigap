#!/usr/bin/env python
# coding: utf-8

from ase import Atoms
import numpy.random as rand
from ase.optimize import BFGS
from ase.optimize import MDMin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.io import read
import matplotlib.pyplot as plt
import os
import numpy as np
import base64

# --------
import sys
sys.path.append('../code')
from ASE_helpers import *
from general_helpers import *
# --------

def make_diatomic(element = 'N', verbose=False, bond_length=1.1, calc_type="EMT"):
    atom = Atoms(element)
    molecule = Atoms('2'+ element, [(0., 0., -bond_length/2), (0., 0., bond_length/2)])
    
    atom = assign_calc(atom, calc_type)
    molecule = assign_calc(molecule, calc_type)

    if verbose:
        e_atom = atom.get_potential_energy()
        e_molecule = molecule.get_potential_energy()
        e_atomization = e_molecule - 2 * e_atom
        
        element_names = {"N": "Nitrogen", "O": "Oxygen", "H": "Hydrogen"}
        
        element_full = (element_names[element] if element in element_names else element)
        
        print("According to {} calculator:".format(calc_type))
        print('%s atom energy: %5.2f eV' % (element_full, e_atom))
        print('%s molecule energy: %5.2f eV' % (element_full, e_molecule))
        print('Atomization energy: %5.2f eV' % -e_atomization)
    
    return molecule




def make_optimized_diatomic(element = "N", optimizer_type="MDMin", fmax=0.0001, verbose=False, bond_length=1.1, 
                            optimize_step=.02, calc_type="EMT",
                            traj_directory = "DEFAULT_DIRECTORY", traj_filename = "DEFAULT_FILENAME", parent_directory="../", return_type="final"):
    if return_type not in ("final", "history", "filename"):
        return_type_error="Do not understand return_type value '{}'. \
        \nUse 'final' for the final optimized molecule; use 'history' for a list of all molecules in optimization; \
        \nor use 'filename' for the location where this trajectory was saved.".format(return_type)
        raise ValueError(return_type_error)
    
    molecule = make_diatomic(element=element, bond_length=bond_length, verbose=verbose, calc_type=calc_type)
        
    if traj_directory not in [None, False]:
        if traj_directory == "DEFAULT_DIRECTORY":
            traj_directory = parent_directory + "/data/"
        traj_filename = traj_filename.replace("DEFAULT_FILENAME", element + "2_" + optimizer_type + "_" + calc_type + ".traj")
        traj_filename = traj_directory + traj_filename
        traj_filename = find_unique_filename(traj_filename, identifier_type="random_string", verbose=verbose)        
    else:
        traj_filename=None
        if return_type == "filename":
            return_type_confusion_error = "Cannot use return_type = 'filename' if printing to file is turned off.\
            \nTo print to a file in the default '/data/' directory, leave your traj_directory argument blank. \
            \nAlternatively, provide a directory to traj_directory or change return_type."
            raise ValueError(return_type_confusion_error)
            
        
    if optimizer_type == "MDMin":
        optimizer = MDMin(molecule, trajectory=traj_filename, logfile=None, dt= optimize_step)
    elif optimizer_type == "BFGS":
        optimizer = BFGS(molecule, trajectory=traj_filename, logfile=None)
    else:
        print("This function does not currently support the optimizer type '{}'".format(optimizer_type))
        return
    
    optimizer.run(fmax=fmax)
    
    if return_type == "filename":
        return traj_filename
    elif return_type == "final":
        return molecule
    elif return_type == "history":
        return read(traj_filename, index=':')


    
    

def print_md_progress(molecule, i, is_diatomic=False):
    epot = molecule.get_potential_energy() / len(molecule)
    ekin = molecule.get_kinetic_energy() / len(molecule)
    if is_diatomic:
        print('Step %2.0f: Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK) Etot = %.3feV' 
              % (i, epot, ekin, ekin / (2.5 * units.kB), epot + ekin))
    else:
        print('Step %2.0f: Energy per atom: Epot = %.3feV  Ekin = %.3feV  Etot = %.3feV' 
              % (i, epot, ekin,  epot + ekin))

def generate_md_traj(structure=None, from_diatomic=False, element = "N", nsteps=10, md_type="VelocityVerlet", time_step=1, bond_length=1.1,
                     temperature=300, verbose = False, print_step_size = 10, calc_type="EMT", preoptimize=True, return_traj_file = False,
                     md_seed=1, jitter=0, plot_energies="default", traj_directory = None, parent_directory="../"):
    
    rand.seed(md_seed)
    if structure is None and from_diatomic == False:
        print("Must provide a structure from which to start a trajectory or specify that you want to generate a trajectory of diatomic molecules.")
        return
    elif from_diatomic == True:
        if preoptimize:
            molecule = make_optimized_diatomic(element=element, verbose=verbose, bond_length=bond_length, calc_type=calc_type,
                                               traj_directory= traj_directory, traj_filename = "for_MD_DEFAULT_FILENAME")
        else:
            molecule = make_diatomic(element = element, verbose=verbose, bond_length=bond_length, calc_type=calc_type)
        chemical_formula = "{}2".format(element)
        if verbose:
            print("Now generating MD trajectory of {} {}₂ molecules at {:.0f} K using {} dynamics and the {} calculator".format(nsteps, element, temperature, md_type, calc_type))
    elif structure is not None:
        molecule = structure
        chemical_formula = molecule.get_chemical_formula()
        molecule = assign_calc(molecule, calc_type)
        # Replace this with the built-in ASE method ideally
        if jitter:
            jittered_positions = molecule.positions + np.random.normal(0, jitter, molecule.positions.shape )
            molecule.set_positions(jittered_positions)

        if verbose:
            print("Now generating MD trajectory of {} {} structures at {:.0f} K using {} dynamics and the {} calculator".format(nsteps, chemical_formula, temperature, md_type, calc_type ))
    else:
        print("Did not understand instructions for generating trajectory.")
        return
    
    # This could be generalized in the future to work in a directory outside of minigap,
    # but for now it needs to have a sister directory called 'data' or have its directly explicitly provided
    if traj_directory in [True, "", None, False]:
        traj_directory = parent_directory + "/data/"
    traj_filename = traj_directory + chemical_formula + "_" + md_type + ".traj"
    traj_filename = find_unique_filename(traj_filename, identifier_type="random_string", verbose=verbose)        

        
    MaxwellBoltzmannDistribution(molecule, temperature_K=temperature)# * (2.5 * units.kB))
    
    if md_type == "VelocityVerlet":
        md = VelocityVerlet(molecule, time_step * units.fs,
                            trajectory=traj_filename, logfile=None)#log_filename)
    elif md_type == "Berendsen":
        md = NVTBerendsen(molecule, time_step * units.fs, taut = time_step * units.fs, temperature_K = temperature,
                            trajectory=traj_filename, logfile=None)#log_filename)
    else:
        print("This function does not currently support the molecular dynamics type '{}'".format(md_type))
        return
    # It is usually most convenient to pass the total size of the trajectory you want which is the number of steps + 1
    # nsteps, the generate_md_traj parameter, refers to the total trajectory size whereas nsteps, the local generate_md_traj variable, refers to the number of md steps
    nsteps -= 1
 
    try:
        if verbose:
            step_i = 0
            remaining_steps = nsteps
            while step_i <= nsteps - print_step_size:
                md.run(print_step_size)
                step_i += print_step_size
                print_md_progress(molecule, step_i, from_diatomic)
            if step_i < nsteps:
                md.run(nsteps - step_i)
                step_i = nsteps
                print_md_progress(molecule, step_i, from_diatomic)
        else:
            md.run(nsteps)
    except NotImplementedError as err:
        acceptable_calcs = ["Morse", "LJ", "EMT"]
        acceptable_calcs.remove(calc_type)
        print(str(err) + ". Other md calculator options are: ", acceptable_calcs)
        return "MD failed"
    
    if return_traj_file:
        return traj_filename
    else:
        atoms_traj_list = read(traj_filename,index=':')

        if plot_energies == "on" or (verbose and plot_energies != "off"):
            if from_diatomic:
                plt.plot([atoms.get_kinetic_energy()/len(atoms)/(2.5*units.kB) for atoms in atoms_traj_list])
            else:
                plt.plot([atoms.get_kinetic_energy()/len(atoms) for atoms in atoms_traj_list])

        return atoms_traj_list