#!/usr/bin/env python
# coding: utf-8

from dscribe.descriptors import SOAP

def get_dscribe_descriptors(atoms_list, species = ["N"], rcut = 3.2, nmax = 5, lmax = 5, is_global=True, return_derivatives = False, positions = []):
    atoms_list = ([atoms_list] if isinstance(atoms_list, Atoms) else atoms_list)
    averaging_keyword = "outer" if is_global else "off"
    if not len(positions):
        positions = [atoms.get_positions() for atoms in atoms_list]
    soap = SOAP(average=averaging_keyword,  species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax)
    if return_derivatives:
        return soap.derivatives(atoms_list, positions = positions, method="auto")
    else:
        return soap.create(atoms_list)