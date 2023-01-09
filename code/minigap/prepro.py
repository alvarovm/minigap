import numpy as np
from ase.units import Rydberg, kJ, Hartree, mol, kcal
from .logging import logger

# -------- WHAT
no_forces_string = "Not Using Forces"


def CompileStructureList(settings, miniGAP_parent_directory, output_directory="../results/"):
    struct_list = []
    if settings.structure_file in (None, "None", ""):
        print('hello')
#         if structure_settings.chemical_formula in (None, "None", ""):
#             # Diatomic molecule in MD trajectory
#             struct_list = generate_MD_traj_for_miniGAP(structure=None, from_diatomic=True, md_settings=settings, 
#                                                    miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )
#             # This simpler dataset may be useful for debugging purposes
#             # Diatomic molecule with evenly spaced bond lengths
#             # struct_list = [make_diatomic(element = structure_settings.diatomic_element, verbose=False, bond_length=L, \
#             #                              calc_type=structure_settings.md_energy_calculator) for L in np.linspace(.6,1.8, structure_settings.n_total)]

#         else:
#             # Compile dataset by picking a molecule from the g2 collection and performing MD on it
#             # ASE creates this g2 collection 'lazily' so we must call each member before we can get a complete list of names
#             g2_list = [m for m in g2]
#             del g2_list
#             if settings.chemical_formula in g2._names:
#                 struct_list = generate_MD_traj_for_miniGAP(structure=molecule(settings.chemical_formula), from_diatomic=False, 
#                                                        md_settings=settings, miniGAP_parent_directory=miniGAP_parent_directory,
#                                                        traj_directory=output_directory )
#             else: 
#                 # https://aip.scitation.org/doi/10.1063/1.473182
#                 # https://wiki.fysik.dtu.dk/ase/ase/build/build.html#molecules
#                 print("ASE does not recognize {}. Please choose from the g2 collection: {}".format(structure_settings.chemical_formula, g2._names))
#                 if not in_notebook:
#                     exit()
    else:
        print('hello2')
#         if structure_settings.molecular_dynamics:
#             # Compile dataset by picking a structure from the a provided structure fiel and performing MD on it
#             starter_struct = import_structs_for_miniGAP( settings, in_notebook, miniGAP_parent_directory, by_indices=True)[0]
#             struct_list = generate_MD_traj_for_miniGAP(structure=starter_struct, from_diatomic=False, md_settings=structure_settings, 
#                                                    miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )
#         else:
            # Import dataset directly from file
    struct_list = import_structs(settings, miniGAP_parent_directory)
    return struct_list 

def import_structs(settings, miniGAP_parent_directory = "../", by_indices=False, frac=None):
    
    from os.path import join, isfile, splitext
    from ase.io import read
    
    
    if frac is None:
        if settings.import_fraction is not None:
            frac = settings.import_fraction
    
    StructureList = []
    
    filename = settings.structure_file
     
    indices = [settings.md_index]
    
    tested_filetypes = ("xyz", "extxyz", "extxyz.gz", "xyz.gz", "db", "traj")
    
    if not isfile(filename):

#     if "/" not in filename:
        filename = miniGAP_parent_directory + "/data/" + filename
        #filename = join(join(miniGAP_parent_directory,"data/"), filename)
        if not isfile(filename):
            
#    filetype = "." + filename.split("/")[-1].split(".", 1)[1]
            logger.debug("No structure file {} found.".format(settings.structure_file))
            return StructureList
    
    filetype = splitext(filename)[1].strip('.').lower()
    
    if by_indices:
        print ('no index')
#         for index in indices:
#             StructureList.append(read(filename, index))
#         if settings.verbose:
#             if len(indices) == 1:
#                 logger.debug("Imported structure #{} from {}.".format(indices[0], filename))
#             else:
#                 logger.debug("Imported {} structures from {} with indices: {}".format(len(indices), filename, indices) )
    elif filetype in tested_filetypes:
        if filetype != ".db":
            logger.debug("If you plan to use miniGAP with this dataset more than once, it is recommended that you convert this file to a .db file. \
            \nThe conversion may be slow, but this will save significant time each run of miniGAP thereafter. \
            \nTo perform this conversion, run the command 'code/convert_to_db.py {}' from the minigap parent directory".format(filename))
          
        FullStructureList = read(filename, ":")

        
        n_full_dataset = len(FullStructureList)
        
        if settings.import_fraction is not None:
            if not (settings.import_fraction <= 1 and settings.import_fraction > 0):
                import_frac_err = "Cannot use '{}' as import_fraction. Please choose a number ∈ (0, 1] for import_fraction ".format(settings.import_fraction)
                import_frac_err += "or explicitly instruct miniGAP to import N structures with the settings import_fraction=null and n_total = N ."
                logger.debug(import_frac_err)
                raise ValueError(import_frac_err)
            elif settings.verbose:
                logger.debug("Importing {:.0%} of dataset...".format(settings.import_fraction))
            n_total = int(settings.import_fraction * n_full_dataset)
        else:
            if settings.n_total > n_full_dataset:
                logger.debug("Cannot import {} structures from a dataset containing {} structures. Importing all structures instead.".format(settings.n_total, n_full_dataset))
            n_total = min(settings.n_total, n_full_dataset)
            
        StructureList = FullStructureList[::n_full_dataset//n_total][:n_total]
        
        if settings.verbose:

            logger.debug("Imported {} structures from {}. Structures were taken uniformly from throughout dataset which contains {} total structures.".format(len(StructureList), settings.structure_file, n_full_dataset))

    else:
        logger.debug("Have not yet tested filetype '{}'. However, it is possible ASE can read this filetype. \
        \nIf so, add this filetype to the tested_filetypes list in the import_structs_for_miniGAP function to enable import from {}.".format(filetype, filename))
        
        if not settings.in_notebook:
            exit()

    return StructureList

