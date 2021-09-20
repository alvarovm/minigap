#!/usr/bin/env python
# coding: utf-8

from dscribe.descriptors import SOAP
from ase import Atoms

def get_dscribe_descriptors(atoms_list, species = ["N"], rcut = 3.2, nmax = 5, lmax = 5, is_global=True, return_derivatives = False, positions = [],
                            rbf_type="gto", is_periodic=False, attach=False):
    atoms_list = ([atoms_list] if isinstance(atoms_list, Atoms) else atoms_list)
    averaging_keyword = "outer" if is_global else "off"

    soap = SOAP(average=averaging_keyword,  species=species, periodic=is_periodic, rcut=rcut, nmax=nmax, lmax=lmax, rbf=rbf_type)
    if return_derivatives:
        if len(positions):
            return soap.derivatives(atoms_list, positions = positions, method="auto", attach=attach)
        else:
            #positions = [atoms.get_positions() for atoms in atoms_list]
            return soap.derivatives(atoms_list, method="auto", attach=attach)
            
    else:
        return soap.create(atoms_list)