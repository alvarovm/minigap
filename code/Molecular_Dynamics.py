#!/usr/bin/env python
# coding: utf-8

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS
from ase.optimize import MDMin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.io import read
import matplotlib.pyplot as plt
import os
import base64

def generate_unique_id():
    return base64.b64encode(os.urandom(64)).decode().replace("/", "").replace("+", "")[:5]

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
        
        print('%s atom energy: %5.2f eV' % (element_full, e_atom))
        print('%s molecule energy: %5.2f eV' % (element_full, e_molecule))
        print('Atomization energy: %5.2f eV' % -e_atomization)
    
    return molecule




def make_optimized_diatomic(element = "N", optimizer_type="MDMin", fmax=0.0001, verbose=False, bond_length=1.1, 
                            optimize_step=.02, calc_type="EMT", return_traj_file = False):
    molecule = make_diatomic(element=element, bond_length=bond_length, verbose=verbose, calc_type=calc_type)
        
    if return_traj_file:
        fid = generate_unique_id()
        traj_filename = "../data/" + element + "_" + optimizer_type + "_" + fid + ".traj"
    else:
        traj_filename=None
    
    if optimizer_type == "MDMin":
        optimizer = MDMin(molecule, trajectory=traj_filename, logfile=None, dt= optimize_step)
    elif optimizer_type == "BFGS":
        optimizer = BFGS(molecule, trajectory=traj_filename, logfile=None)
    else:
        print("This function does not currently support the optimizer type '{}'".format(optimizer_type))
        return
    
    optimizer.run(fmax=fmax)
    
    if return_traj_file:
        return traj_filename
    else:
        return molecule


    
    

def print_md_progress(molecule, i):
    epot = molecule.get_potential_energy() / len(molecule)
    ekin = molecule.get_kinetic_energy() / len(molecule)
    print('Step %2.0f: Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK) Etot = %.3feV' 
          % (i, epot, ekin, ekin / (2.5 * units.kB), epot + ekin))

def generate_md_traj(structure=None, from_diatomic=False, element = "N", nsteps=10, md_type="VelocityVerlet", time_step=1, bond_length=1.1,
                           temperature=300, verbose = False, print_step_size = 10, calc_type="EMT", preoptimize=True, return_traj_file = False):
    if structure is None and from_diatomic == False:
        print("Must provide a structure from which to start a trajectory or specify that you want to generate a trajectory of diatomic molecules.")
        return
    elif from_diatomic == True:
        if preoptimize:
            molecule = make_optimized_diatomic(element=element, verbose=verbose, bond_length=bond_length, calc_type=calc_type)
        else:
            molecule = make_diatomic(element = element, verbose=verbose, bond_length=bond_length, calc_type=calc_type)
        chemical_formula = "{}2".format(element)
        if verbose:
            print("Now generating MD trajectory of {} {}₂ molecules at {:.0f} K using {} dynamics".format(nsteps, element, temperature, md_type ))
    elif structure is not None:
        molecule = structure
        chemical_formula = molecule.get_chemical_formula()
        molecule = assign_calc(molecule, calc_type)
        if verbose:
            print("Now generating MD trajectory of {} {} structures at {:.0f} K using {} dynamics".format(nsteps, chemical_formula, temperature, md_type ))
    else:
        print("Did not understand instructions for generating trajectory.")
        return
    
    fid = generate_unique_id()

    traj_filename = "../data/" + chemical_formula + "_" + md_type + "_" + fid + ".traj"
    
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
    
    if verbose:
        step_i = 0
        remaining_steps = nsteps
        while step_i <= nsteps - print_step_size:
            md.run(print_step_size)
            step_i += print_step_size
            print_md_progress(molecule, step_i)
        if step_i < nsteps:
            md.run(nsteps - step_i)
            step_i = nsteps
            print_md_progress(molecule, step_i)
    else:
        md.run(nsteps)
    
    if return_traj_file:
        return traj_filename
    else:
        atoms_traj_list = read(traj_filename,index=':')

        if verbose:
            plt.plot([atoms.get_kinetic_energy()/len(atoms)/(2.5*units.kB) for atoms in atoms_traj_list])

        return atoms_traj_list