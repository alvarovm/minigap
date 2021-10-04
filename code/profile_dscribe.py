from Generate_Descriptors import get_dscribe_descriptors
from ase.io import read
import numpy as np




# I only use these settings if I am training on diatomic structures
# ------------------------------------------------------------------------
use_diatomics = False
if use_diatomics:
    my_element = "O"
    my_temp=300
    my_calc = "EMT"
    my_md_seed = 1

# Number of structures    
# ------------------------------------------------------------------------
my_n = 1

# SOAP parameters
# ------------------------------------------------------------------------
get_local_descriptors = True
my_rcut = 5
my_nmax = 8
my_lmax = 8
attach_SOAP_center = True
is_periodic = False
use_forces =True

StructureList = read("../data/ManyGraphenes_unzipped.extxyz", ":{}".format(my_n))


dscribe_output = get_dscribe_descriptors(
                                            StructureList,
                                            species=np.unique(StructureList[0].get_chemical_symbols()), 
                                            attach=attach_SOAP_center, 
                                            is_periodic = is_periodic,
                                            is_global= not get_local_descriptors,
                                            return_derivatives= use_forces, 
                                            nmax=my_nmax, 
                                            lmax=my_lmax,
                                            rcut=my_rcut
                                            )
