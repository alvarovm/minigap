#!/usr/bin/env python
# coding: utf-8

from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np

def get_dscribe_descriptors(atoms_list, species = [], rcut = 3.2, nmax = 5, lmax = 5,
                            is_global=False, return_derivatives = False, positions = [],
                            rbf_type="gto", is_periodic=False, attach=False, smear=1.0, n_jobs = 1):
    atoms_list = [atoms_list] if isinstance(atoms_list, Atoms) else atoms_list
    species = species if len(species) else np.unique(sum([atoms_list_i.get_chemical_symbols() for atoms_list_i in atoms_list], []))
    averaging_keyword = "outer" if is_global else "off"

    soap = SOAP(average=averaging_keyword,  species=species, periodic=is_periodic, rcut=rcut, nmax=nmax, lmax=lmax, rbf=rbf_type, sigma=smear)
    if return_derivatives:
        if len(positions):
            return soap.derivatives(atoms_list, positions = positions, method="auto", attach=attach, n_jobs = n_jobs)
        else:
            #positions = [atoms.get_positions() for atoms in atoms_list]
            return soap.derivatives(atoms_list, method="auto", attach=attach, n_jobs = n_jobs)
            
    else:
        return soap.create(atoms_list)