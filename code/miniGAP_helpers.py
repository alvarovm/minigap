from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.kernels import SquaredExponential, Polynomial

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skcosmo.sample_selection import CUR as CURSample
from skcosmo.sample_selection import PCovCUR as PCovCURSample
from skcosmo.feature_selection import CUR as CURFeature
from skcosmo.feature_selection import PCovCUR as PCovCURFeature

from ase.io import read
from ase.collections import g2
from ase.build import molecule
import os.path as path
import time

# --------
import sys
sys.path.append('../code')
from Generate_Descriptors import get_dscribe_descriptors
from Molecular_Dynamics import generate_md_traj
from plot_helpers import *
from analysis_helpers import *
from general_helpers import make_unique_directory
from ASE_helpers import *
# --------
no_forces_string = "Not Using Forces"


def make_miniGAP_results_subdirectory(settings, date, miniGAP_parent_directory="../"):
    title = settings.title
    ideal_subdirectory_name = title if title != None else "results"
    if settings.append_date_to_title:
        ideal_subdirectory_name += date
    ideal_directory_name = miniGAP_parent_directory + "results/" + ideal_subdirectory_name
    unique_directory_name = make_unique_directory(ideal_directory_name, identifier_type='counter', verbose=settings.verbose)
    return unique_directory_name



def import_structs_for_miniGAP(import_settings, in_notebook=True, miniGAP_parent_directory = "../", by_indices=False):
    filename = import_settings.structure_file
    verbose = import_settings.verbose 
    indices = [import_settings.md_index]
    
    tested_filetypes = (".xyz", ".extxyz", ".extxyz.gz", ".xyz.gz", ".db", ".traj")
    
    if "/" not in filename:
        filename = miniGAP_parent_directory + "data/" + filename
    filetype = "." + filename.split("/")[-1].split(".", 1)[1]
    if by_indices:
        StructureList = []
        for index in indices:
            StructureList.append(read(filename, index))
        if verbose:
            if len(indices) == 1:
                print("Imported structure #{} from {}.".format(indices[0], filename))
            else:
                print("Imported {} structures from {} with indices: {}".format(len(indices), filename, indices) )
    elif filetype in tested_filetypes:
        if filetype != ".db":
            print("If you plan to use miniGAP with this dataset more than once, it is recommended that you convert this file to a .db file. \
            \nThe conversion may be slow, but this will save significant time each run of miniGAP thereafter. \
            \nTo perform this conversion, run the command 'code/convert_to_db.py {}' from the minigap parent directory".format(filename))
          
        FullStructureList = read(filename, ":")
        n_full_dataset = len(FullStructureList)
        if import_settings.import_fraction is not None:
            if not (import_settings.import_fraction <= 1 and import_settings.import_fraction > 0):
                import_frac_err = "Cannot use '{}' as import_fraction. Please choose a number ∈ (0, 1] for import_fraction ".format(import_settings.import_fraction)
                import_frac_err += "or explicitly instruct miniGAP to import N structures with the settings import_fraction=null and n_total = N ."
                raise ValueError(import_frac_err)
            elif verbose:
                print("Importing {:.0%} of dataset...".format(import_settings.import_fraction), end="")
            n_total = int(import_settings.import_fraction * n_full_dataset)
        else:
            if import_settings.n_total > n_full_dataset:
                print("Cannot import {} structures from a dataset containing {} structures. Importing all structures instead.".format(import_settings.n_total, n_full_dataset))
            n_total = min(import_settings.n_total, n_full_dataset)
        StructureList = FullStructureList[::n_full_dataset//n_total][:n_total]
        if verbose:
            print("Imported {} structures from {}. Structures were taken uniformly from throughout dataset which contains {} total structures.".format(len(StructureList), import_settings.structure_file, n_full_dataset))
            # old code
#             if filetype == ".db":
#                 from ase.db import connect
#                 db = connect(filename)
#                 FullStructureList = []
#                 for row in db.select():
#                     FullStructureList.append(db.get_atoms(id = row.id))            
#             else:
#                 print("Imported the first {} structures from {} as our dataset.\
#                 \nIt would be better to drawn on structures evenly from throughout your full dataset instead of taking only from the beginning.\
#                 \nHowever, it would be prohibitively slow to import all structures from a file other than an ASE database file during every miniGAP run.\
#                 \nThis is partially why conversion to a .db is recommended.".format(n_total, filename))
#                 StructureList = read(filename, ":{}".format(n_total))

    else:
        print("Have not yet tested filetype '{}'. However, it is possible ASE can read this filetype. \
        \nIf so, add this filetype to the tested_filetypes list in the import_structs_for_miniGAP function to enable import from {}.".format(filetype, filename))
        
        if not in_notebook:
            exit()
    return StructureList


def generate_MD_traj_for_miniGAP(structure, from_diatomic, md_settings, miniGAP_parent_directory="../", traj_directory = "../data/"):
    return generate_md_traj(structure=structure, from_diatomic=from_diatomic, preoptimize=False, bond_length=md_settings.diatomic_bond_length, 
                            element=md_settings.diatomic_element, temperature=md_settings.md_temp, nsteps=md_settings.n_total,
                            md_type = md_settings.md_algorithm, calc_type=md_settings.md_energy_calculator, md_seed= md_settings.md_seed, 
                            time_step=md_settings.md_time_step, verbose=md_settings.verbose, print_step_size=md_settings.n_total/10, 
                            plot_energies="off", parent_directory=miniGAP_parent_directory, traj_directory = traj_directory)

def CompileStructureList(structure_settings, in_notebook, miniGAP_parent_directory, output_directory="../results/"):
    if structure_settings.structure_file in (None, "None", ""):
        if structure_settings.chemical_formula in (None, "None", ""):
            # Diatomic molecule in MD trajectory
            struct_list = generate_MD_traj_for_miniGAP(structure=None, from_diatomic=True, md_settings=structure_settings, 
                                                   miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )
            # This simpler dataset may be useful for debugging purposes
            # Diatomic molecule with evenly spaced bond lengths
            # struct_list = [make_diatomic(element = structure_settings.diatomic_element, verbose=False, bond_length=L, \
            #                              calc_type=structure_settings.md_energy_calculator) for L in np.linspace(.6,1.8, structure_settings.n_total)]

        else:
            # Compile dataset by picking a molecule from the g2 collection and performing MD on it
            # ASE creates this g2 collection 'lazily' so we must call each member before we can get a complete list of names
            g2_list = [m for m in g2]
            del g2_list
            if structure_settings.chemical_formula in g2._names:
                struct_list = generate_MD_traj_for_miniGAP(structure=molecule(structure_settings.chemical_formula), from_diatomic=False, 
                                                       md_settings=structure_settings, miniGAP_parent_directory=miniGAP_parent_directory,
                                                       traj_directory=output_directory )
            else: 
                # https://aip.scitation.org/doi/10.1063/1.473182
                # https://wiki.fysik.dtu.dk/ase/ase/build/build.html#molecules
                print("ASE does not recognize {}. Please choose from the g2 collection: {}".format(structure_settings.chemical_formula, g2._names))
                if not in_notebook:
                    exit()
    else:
        if structure_settings.molecular_dynamics:
            # Compile dataset by picking a structure from the a provided structure fiel and performing MD on it
            starter_struct = import_structs_for_miniGAP( structure_settings, in_notebook, miniGAP_parent_directory, by_indices=True)[0]
            struct_list = generate_MD_traj_for_miniGAP(structure=starter_struct, from_diatomic=False, md_settings=structure_settings, 
                                                   miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )
        else:
            # Import dataset directly from file
            struct_list = import_structs_for_miniGAP( structure_settings, in_notebook, miniGAP_parent_directory)
    return struct_list 

def isolated_energy_parser(user_input):
    # returns dictionary from input format
    # e.g. "H:-0.4::Li:-7.5" --> {"H":-0.4, "Li":-7.5}
    return {atom_energy.split(":")[0]:float(atom_energy.split(":")[1]) for atom_energy in user_input.split("::")}

# returns isolated energies in eV
def get_isolated_energy(element, user_provided_isolated_energy_string, user_unit):
        # Calculated with B3LYP+D3 on CRYSTAL17
        default_isolated_energies = {
            "H":  -0.43796573720365,
            "Li": -7.4538249414114,
            "Be": -14.644518869964,
            "B": -24.512608074927,
            "C": -37.768216780161,
            "N": -54.572189501408,
            "O": -75.003393876879,
            "F": -99.672000678934,
            "Na": -162.20736395632,
            "Mg": -199.4784909209,
            "Al": -241.9640517253,
            "Si": -289.26042225551,
            "P": -341.20691893397,
            "S": -398.04415675952,
            "Cl": -460.04770109924,
            "K": -599.77830261799,
            "Ca": -677.27165785863,
            "Sc": -760.34151748052,
            "Ti": -848.8244251979,
            "V": -943.67762714996,
            "Cr": -1044.1147378511,
            "Mn": -1150.8928446101,
            "Co": -1382.3371803164,
            "Ni": -1508.0647901075,
            "Cu": -1640.1713647787,
            "Zn": -1779.2571044599,
            "Ga": -1924.5685189651,
            "Ge": -2076.7379664988,
            "As": -2235.7034565607,
            "Tc": -4205.5805934383
        }
        
        if user_provided_isolated_energy_string is not None:
            user_provided_isolated_energies = isolated_energy_parser(user_provided_isolated_energy_string)
            if element in user_provided_isolated_energies:
                isolated_energy = convert_energy(user_provided_isolated_energies[element], user_unit, "eV")
            elif element in default_isolated_energies:
                isolated_energy = default_isolated_energies[element]
            else:
                isolated_energy = 0
        else:
            isolated_energy = 0  
        return isolated_energy
  

    

def GatherStructureInfo(struct_list, gather_settings ):
    gather_forces = gather_settings.use_forces
    alt_en_kw = gather_settings.alt_energy_keyword
    
    pos_list =  [list(atoms.positions) for atoms in struct_list]
    n_atom_list = np.array([len(struct) for struct in struct_list])
    

    en_list = []
    en_shift_list = []
    frc_list = []
    for struct in struct_list:
        # get average atomic energy for each atom
        # Can we use .get_potential_energies() method instead? 
        en_list_i = [retrieve_ASE_energy(struct, alt_keyword = alt_en_kw)/len(struct) for atom in struct]
        # Convert units if necessary
        # Convert to eV and eV/Å during calculation
        # Because F = -dE/dx, we must use the same x units for the known F values and the predicted F values
        # In particular, I choose eV and eV/Å because ASE positions are always assumed to be in angstroms
        # If you are using an input file which has structural information not in angstroms, forces learned by miniGAP will not be accurate
        en_list_i = convert_energy(en_list_i, gather_settings.input_energy_units, "eV") 
        # Subtract energy of free atoms if specified by user
        isolated_energies = [ get_isolated_energy(atom.symbol, gather_settings.isolated_energies, gather_settings.isolated_energy_units) for atom in struct ]
        en_list_i -= isolated_energies
        en_list.append(en_list_i)
        # Sum of isolated energies of all atoms of each structure
        en_shift_list.append(sum(isolated_energies))
        
        if gather_forces:
            # if "forces" not in StructureList[0].calc.results:
            #     *do something* 
            frc_list_i = struct.get_forces()
            # See note in energy section about unit conversion
            frc_list_i = convert_force(frc_list_i, gather_settings.input_force_units, "eV/ang")
        else:
            frc_list_i = no_forces_string
        frc_list.append(frc_list_i)  
    
    return en_list, frc_list, pos_list, n_atom_list, en_shift_list

def GenerateDescriptorsAndDerivatives(struct_list, nmax, lmax, rcut, smear=0.3, attach=True, is_periodic=False, return_derivatives=True, get_local_descriptors = True):
    
    dscribe_output = get_dscribe_descriptors(
                                            struct_list, 
                                            attach=attach, 
                                            is_periodic = is_periodic,
                                            is_global= not get_local_descriptors,
                                            return_derivatives= return_derivatives, 
                                            nmax=nmax, 
                                            lmax=lmax,
                                            rcut=rcut,
                                            smear=smear
    )

    if return_derivatives:
        dsp_dx_list, sp_list = dscribe_output
        dsp_dx_list = np.moveaxis(dsp_dx_list.diagonal(axis1=1, axis2=2), -1, 1)
    else:
        sp_list = dscribe_output
        dsp_dx_list = [no_forces_string] * len(sp_list)
    
    return dsp_dx_list, sp_list



def PrepareDataForTraining(sp_list, dsp_dx_list, en_list, frc_list, pos_list, nat_list, en_shift_list, prep_settings):
    split_seed = prep_settings.split_seed
    prepare_forces = prep_settings.use_forces
    scale_soaps = prep_settings.scale_soaps
    n_total =  len(nat_list)
    verbose = prep_settings.verbose
    
    # This comment itself needs to be split up now haha
    
    # Split all data into training and test sets.
    # Test sets will not be viewed until training is done
    # Intra-structural information is not reshaped into local atomic information until after splitting
    # This means structures will be fully in the training or test sets, not split between both
    # It also means it will be possible to predict global energies in the test set
    
    if prep_settings.n_train is not None:
        if prep_settings.n_train > n_total:
            print("Cannot use {} structures in the training set because only {} total structures were imported.".format( prep_settings.n_train, n_total ) )
            print("Attempting to choose the training set using train_fraction.")
            test_fract = 1 - prep_settings.train_fraction
        else:
            test_fract = 1 - prep_settings.n_train / n_total
    else:
        test_fract = 1 - prep_settings.train_fraction
    
    train_indices, test_indices  = train_test_split(np.arange(n_total), random_state = split_seed, test_size = test_fract )
    n_train, n_test = len(train_indices), len(test_indices)
    
    if verbose:
        print("Split the {}-structure dataset into a {}-structure training dataset and a {}-structure test dataset.".format(n_total, n_train, n_test) )
        print("The test dataset will be set aside until the model is trained and ready for prediction.")
    
    train_ens = [en_list[i] for i in train_indices]; test_ens = [en_list[i] for i in test_indices]
    train_ens, test_ens = np.concatenate(train_ens).reshape(-1, 1), np.concatenate(test_ens).reshape(-1, 1)

    train_nats, test_nats = nat_list[train_indices], nat_list[test_indices]
    train_struct_bools = np.repeat(np.eye(n_train, dtype=np.float64), train_nats, axis=1)
    test_struct_bools = np.repeat(np.eye(n_test, dtype=np.float64), test_nats, axis=1)
    train_nats, test_nats = np.repeat(train_nats, train_nats).reshape((-1, 1)), np.repeat(test_nats, test_nats).reshape((-1, 1))
    # No need to scale the isolated atomic energies, which won't be used until after the training is done
    train_en_shfts, test_en_shfts = np.array(en_shift_list)[train_indices], np.array(en_shift_list)[test_indices]

    # Scale energies to have zero mean and unit variance.
    # Divide forces by the same scale factor but don't subtract the mean

    ens_scaler = StandardScaler().fit(train_ens)
    train_ens, test_ens = ens_scaler.transform(train_ens), ens_scaler.transform(test_ens)
    # The following line is for the tensorflow code. If it is commented out, it is for the gpflow code
    #train_ens, test_ens = train_ens.flatten(), test_ens.flatten()
    ens_var = train_ens.var()
    
    if prepare_forces:
        train_frcs = [frc_list[i] for i in train_indices]; test_frcs = [frc_list[i] for i in test_indices]
        train_frcs, test_frcs = np.concatenate(train_frcs, axis=0), np.concatenate(test_frcs, axis=0)
        train_frcs, test_frcs = train_frcs / ens_scaler.scale_, test_frcs / ens_scaler.scale_
        frcs_var = train_frcs.var()
            
    # soap section
    train_sps_full, test_sps_full = [sp_list[i] for i in train_indices], [sp_list[i] for i in test_indices]
    train_sps_full, test_sps_full = np.concatenate(train_sps_full), np.concatenate(test_sps_full)
    
    # Scale soaps to have zero mean and unit variance.
    # Divide derivatives by the same scale factor but don't subtract the mean
    
    soap_scaler = StandardScaler().fit(train_sps_full)
    if scale_soaps:
        train_sps_full, test_sps_full = soap_scaler.transform(train_sps_full), soap_scaler.transform(test_sps_full)
    
    if prepare_forces:
        train_dsp_dx, test_dsp_dx = [dsp_dx_list[i] for i in train_indices], [dsp_dx_list[i] for i in test_indices]
        train_dsp_dx, test_dsp_dx = np.concatenate(train_dsp_dx).reshape(-1, 3, dsp_dx_list.shape[-1]), np.concatenate(test_dsp_dx).reshape(-1, 3, dsp_dx_list.shape[-1])
        if scale_soaps:
            train_dsp_dx, test_dsp_dx = train_dsp_dx/soap_scaler.scale_, test_dsp_dx/soap_scaler.scale_
    
    # Convert data to tensorflow tensors where necessary  
    # Maybe it makes sense to wait to do this since I have to convert them back to numpy to split them into validation/training sets anyway
    # (The only one I didn't have to convert is test_sps_full, but I can just do this later if I'm moving the rest to later)
#     train_sps_full = tf.constant(train_sps_full, dtype=np.float64)
#     train_ens = tf.constant(train_ens, dtype=np.float64)
#     test_sps_full = tf.constant(test_sps_full, dtype=np.float64)
#     if prepare_forces:
#         train_frcs = tf.constant(train_frcs, dtype=np.float64)

    if not prepare_forces:
        return train_sps_full, test_sps_full, train_ens, test_ens, train_nats, test_nats, train_en_shfts, test_en_shfts, train_indices, test_indices, train_struct_bools, test_struct_bools, soap_scaler, ens_scaler, ens_var
    else:
        return train_sps_full, test_sps_full, train_ens, test_ens, train_nats, test_nats, train_en_shfts, test_en_shfts, train_indices, test_indices, train_struct_bools, test_struct_bools, soap_scaler, ens_scaler, ens_var, train_dsp_dx, test_dsp_dx, train_frcs, test_frcs, frcs_var



def SparsifySoaps(train_soaps, train_energies= [], test_soaps = [], sparsify_samples=False, n_samples=0, sparsify_features=False, n_features=0, selection_method="CUR", **kwargs):
    acceptible_selection_methods = ["CUR", "FPS", "PCovCUR", "KernelPCovR"]
    implemented_selection_methods = ["CUR", "PCovCUR"]
    
    if type(selection_method) in [tuple, list, np.ndarray]:
        if len(selection_method) == 2:
            selection_methods = selection_method
        else:
            print("Incorrect specificaiton of selection method(s). Supply a string to use a single selection method. \
            \nOr supply a list of 2 strings to specify different selection methods for samples and features, respectively.")
            return
    else:
        selection_methods = [selection_method, selection_method]
    
    unrecognized_selection_methods = []
    for method in selection_methods:
        if method not in acceptible_selection_methods:
            unrecognized_selection_methods.append(method)
    if len(unrecognized_selection_methods) > 0:
        print("Do not recognize selecton method(s): {}\nPlease select from {} ".format(unrecognized_selection_methods, acceptible_selection_methods))
        return
    for method in selection_methods:
        if method not in implemented_selection_methods:
            print("Selection by '{}' has not yet been implemented, but can be easily implemented by modificaiton of this function (SparsifySoaps)".format(method))
            return
    
    if "PCovCUR" in selection_methods and len(train_energies) == 0:
        print("You must provide training energies to use PCovCUR. Terminating")
        return
    if sparsify_features == True and len(test_soaps) == 0:
        print("You must provide test soaps to sparsify features. Terminating")
        return        
    
    default_kwargs = {
        "progress_bar":False,
        "score_threshold":1e-12,
        "full":False,
        "score_tolerance":1e-12,
        "iterative_selection":True,
        "mixing":0.5,
        "n_eigenvectors":1,
        "plot_importances":False
    }
    selection_settings = default_kwargs
    for kw, arg in kwargs.items():
        if kw in default_kwargs.keys():
            selection_settings[kw] = arg
        else:
            print("Do not recognize kwarg '{}'. Valid kwargs include {}.".format(kw, default_kwargs.keys()))
    
    
    
    if sparsify_samples:
        if selection_methods[0] == "PCovCUR":
            sample_selector = PCovCURSample(n_to_select=n_samples, progress_bar=selection_settings["progress_bar"], score_threshold=selection_settings["score_threshold"],
                                            full = selection_settings["full"], k = selection_settings["n_eigenvectors"], mixing = selection_settings["mixing"],
                                            iterative = selection_settings["iterative_selection"], tolerance=selection_settings["score_tolerance"]
                                           )
            sample_selector.fit(train_soaps, y=train_energies)
        elif selection_methods[0] == "CUR":
            sample_selector = CURSample(n_to_select=n_samples, progress_bar=selection_settings["progress_bar"], score_threshold=selection_settings["score_threshold"],
                                            full = selection_settings["full"], k = selection_settings["n_eigenvectors"], 
                                            iterative = selection_settings["iterative_selection"], tolerance=selection_settings["score_tolerance"]
                                           )
            sample_selector.fit(train_soaps)
            print("CUR sample selection")
        
    else:
        sample_selector=lambda x: x
        sample_selector.transform = sample_selector.__call__ 
        
    if sparsify_features:
        if selection_methods[1] == "PCovCUR":
            feature_selector = PCovCURFeature(n_to_select=n_features, progress_bar=selection_settings["progress_bar"], score_threshold=selection_settings["score_threshold"],
                                            full = selection_settings["full"], k = selection_settings["n_eigenvectors"], mixing = selection_settings["mixing"],
                                            iterative = selection_settings["iterative_selection"], tolerance=selection_settings["score_tolerance"]
                                           )
            feature_selector.fit(train_soaps, y=train_energies)
        elif selection_methods[1] == "CUR":
            feature_selector = CURFeature(n_to_select=n_features, progress_bar=selection_settings["progress_bar"], score_threshold=selection_settings["score_threshold"],
                                            full = selection_settings["full"], k = selection_settings["n_eigenvectors"], 
                                            iterative = selection_settings["iterative_selection"], tolerance=selection_settings["score_tolerance"]
                                           )
            feature_selector.fit(train_soaps)
            print("CUR feature selection")
    else:
        feature_selector=lambda x: x
        feature_selector.transform = feature_selector.__call__   

        
    representative_train_soaps = sample_selector.transform(train_soaps)
    
    train_soaps = feature_selector.transform(train_soaps)    
    representative_train_soaps = feature_selector.transform(representative_train_soaps)
    test_soaps = feature_selector.transform(test_soaps)
    
    if selection_settings["plot_importances"]: # and (sparsify_samples or sparsify_features):
    #     feature_scores = feature_selector._compute_pi(train_sps_full, train_ens)
        if sparsify_features:
            feature_scores = feature_selector.score(train_soaps)
    #     sample_scores = sample_selector._compute_pi(train_sps_full, train_ens)
        if sparsify_samples:
            sample_scores = sample_selector.score(train_soaps)

        fig, axs = plt.subplots(ncols = 2, figsize=(12, 5))
        if sparsify_features:
            axs[0].plot(np.sort(feature_scores)[::-1])
            axs[0].set_ylabel("Feature Scores")
        if sparsify_samples:
            axs[1].plot(np.sort(sample_scores)[::-1] )
            axs[1].set_ylabel("Sample Scores")
        for ax in axs:
            ax.set_ylim(bottom=0)
    
    return train_soaps, representative_train_soaps, test_soaps


# Note: I still need to add the Tikhonov regularization term for hyperparameter training loss to make the mse equivalent to negative log likelihood

# @tf.function(autograph=False, experimental_compile=False)
def mse(y_predict, y_true):
    return tf.math.reduce_mean(tf.math.squared_difference(y_predict, y_true))

# @tf.function(autograph=False, experimental_compile=False)
def mse_2factor(y1_predict, y1_true, weight1, y2_predict, y2_true, weight2):
    mse1 = mse(y1_predict, y1_true)
    mse2 = mse(y2_predict, y2_true)*3

    return mse1 * weight1 + mse2 * weight2

#@tf.function(autograph=False, experimental_compile=False)
def train_hyperparams_without_forces(model, valid_soap, valid_energy, optimizer):
    with tf.GradientTape() as tape:
        predict_energy = model.predict_f(valid_soap)[0]
#         tf.print("predict energies = ", predict_energy[:3])
        my_mse = mse(predict_energy, valid_energy)
    gradients = tape.gradient(my_mse, model.trainable_variables)
#     tf.print("gradients = ", gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("TRACING train_hyperparams_without_forces")
    return my_mse

# @tf.function(autograph=False, experimental_compile=False)
def predict_energies_from_weights(c, soaps_old, soaps_new, degree, amplitude):
    k = amplitude * tf.math.pow( tf.tensordot(soaps_old, tf.transpose(soaps_new), axes=1), degree )
    return tf.linalg.matmul(c, k, transpose_a=True)


def pick_kernel(kernel_type, **kwargs):
    default_kwargs = {
        # exponentiated_quadratic kernel hyperparameters
        "amplitude":1,
        "length_scale":1,
        # polynomial kernel hyperparameters
        "amplitude":1,
        "degree":2,
        "degree_trainable":False,
        "relative_offset":0,
        "offset_trainable":False,
        #
        "dtype":"float64",
        "verbose":False,
    }
    
    kernel_settings = default_kwargs
    for kw, arg in kwargs.items():
        if kw in default_kwargs.keys():
            kernel_settings[kw] = arg
        else:
            print("Do not recognize pick_kernel kwarg '{}'. Valid pick_kernel kwargs include {}.".format(kw, default_kwargs.keys()))
    
    if kernel_type == "exponentiated_quadratic":
        # gpflow.readthedocs.io/en/master/gpflow/kernels/index.html#gpflow-kernels-squaredexponential
        # k(r) = amplitude exp{-(x-y)²/(2 length_scale²)}
        # = gpflow.Parameter()
        amplitude = gpflow.Parameter(kernel_settings["amplitude"], dtype=kernel_settings["dtype"], name="kernel_amplitude")
        len_scale = gpflow.Parameter(kernel_settings["length_scale"], dtype=kernel_settings["dtype"], name="kernel_len_scale")
        kernel = SquaredExponential(variance=amplitude, lengthscales=len_scale)
        if kernel_settings["verbose"]:
            print("Using an exponentiated quadratic kernel (aka a squared exponential kernel)." )
        return kernel
    elif kernel_type == "polynomial":
        # gpflow.readthedocs.io/en/master/gpflow/kernels/index.html#gpflow-kernels-polynomial
        # k(x, y) = amplitude (xy + relative_offset)ᵈ
        # k(x, y) = (variance * xy + offset)ᵈ
        # variance = amplitude^(1/d), offset = relative_offset * variance
        
        variance = kernel_settings["amplitude"] ** (1/ kernel_settings["degree"])
        offset = kernel_settings["relative_offset"] * variance
        
        # We cannot make offset identically 0
        # This is because gpflow interprets it as a gpflow.Parameter and transforms it with a logarithm
        # However we usually do not want it so I make its magnitude as smallest as possible by default
        # gpflow.readthedocs.io/en/master/_modules/gpflow/kernels/linears.html#Polynomial
        if not offset:
            offset = np.finfo(kernel_settings["dtype"]).tiny
        variance = gpflow.Parameter(variance, dtype=kernel_settings["dtype"], name="kernel_variance")
        offset = gpflow.Parameter(offset, dtype=kernel_settings["dtype"], name="kernel_offset")
        degree =  gpflow.Parameter(kernel_settings["degree"], dtype=kernel_settings["dtype"], name="kernel_degree")
        kernel = Polynomial(variance=variance, offset=offset, degree=degree)
        gpflow.set_trainable(kernel.offset, kernel_settings["offset_trainable"])
        gpflow.set_trainable(kernel.degree, kernel_settings["degree_trainable"])
        if kernel_settings["verbose"]:
            print("Using a degree {} polynomial kernel.".format(kernel_settings["degree"]) )

        return kernel
    else:
        print("Warning: Do not recognize kernel_type={}".format(kernel_type))


        
def CreateMiniGAPVisualization(struct_list, visualization_settings, output_directory, animation_filename="DEFAULT_FILENAME"):
    from Visualize_Structures import Structure3DAnimation

    save_file = visualization_settings.save_dataset_animation & visualization_settings.make_output_files
    verbose = visualization_settings.verbose
    animation_object = Structure3DAnimation(struct_list)
    animation_html5 = animation_object.Plot()
    if save_file:
        animation_filename = output_directory + animation_filename.replace("DEFAULT_FILENAME", "structure_animation.mp4")
        animation_object.Save(animation_filename)
    return animation_html5


def AnalyzeEnergyResults(true_global_ens, predicted_global_ens, analysis_settings,
                true_local_ens=[], predicted_local_ens=[],
                predicted_stdev=None, n_atoms=10, in_notebook=True, output_directory="../data/",
                ax_height = 5, ax_width=7):
    plot_types = analysis_settings.energy_plots.copy()
    stat_types = analysis_settings.error_stats.copy()
    units = analysis_settings.output_energy_units
    local_units = units + "/Atom"
    verbose=analysis_settings.verbose
    color = analysis_settings.color
    
    local_err_stats, local_err  = compile_error_stats( predicted_local_ens,  true_local_ens, stats_to_calculate = stat_types, verbose=verbose)
    global_err_stats, global_err = compile_error_stats(predicted_global_ens, true_global_ens, stats_to_calculate = stat_types, verbose=verbose)
    global_err_per_atom_stats, global_err_per_atom = compile_error_stats(predicted_global_ens/n_atoms, true_global_ens/n_atoms, stats_to_calculate = stat_types, 
                                                                         verbose=verbose)

    
    available_plot_types = ("predicted_vs_true", "global_log_error_histogram", "global_error_per_atom_histogram", "local_log_error_histogram")
    error_info = {}
    system_label = analysis_settings.title + " Calculation" if analysis_settings.title else ""
    global_ens_range = get_data_bounds(true_global_ens, predicted_global_ens, buffer=0.05)
    exponent_range = get_exponent_range((local_err, global_err, global_err_per_atom ), max_min=-5, min_max=-1)
    global_err_range = get_data_bounds(global_err, buffer=0.05)
    local_err_range = get_data_bounds([local_err, global_err_per_atom], buffer=0.05)
    if analysis_settings.crop_energy_outliers:
        if "predicted_vs_true" in plot_types:
            if verbose:
                print("Checking if outliers exists among energy predictions.")            
            cropped_range = suggest_outlier_cropping_for_true_vs_predicted_plot(true_global_ens, predicted_global_ens, buffer = 0, verbose=verbose)
            # The cropping must reduce the range by at least 25% or it isn't performed
            # If the cropping is small it might just confuse people without making a plot that is any more readable than the original
            if np.ptp(cropped_range) < 0.75 * np.ptp(global_ens_range):
                plot_types.append("predicted_vs_true_cropped")
            else:
                print("Cropping energy plots determined to be unnecessary.")
    fig, axs = plt.subplots(figsize = (len(plot_types) * ax_width, ax_height), ncols = len(plot_types) )

    for i in range(len(plot_types)):
        ax = axs[i]
        plot_type = plot_types[i]
        if plot_type == "predicted_vs_true":
            global_r2 = global_err_stats["r2"] if "r2" in stat_types else None
            linear_fit_params = (global_err_stats["m"], global_err_stats["b"]) if "linfit" in stat_types else None
            linear_fit_params = linear_fit_params if linear_fit_params != (None, None) else None
            plot_predicted_vs_true(predicted_global_ens, true_global_ens, ax, variable_label = "Global Energy", units=units, system_label = system_label, r2=global_r2, 
                                   stderr=predicted_stdev, linear_fit_params = linear_fit_params, x_range=global_ens_range, y_range=global_ens_range, color=color, ms=5)
        elif plot_type == "predicted_vs_true_cropped":
            global_r2 = global_err_stats["r2"] if "r2" in stat_types else None
            plot_predicted_vs_true(predicted_global_ens, true_global_ens, ax, variable_label = "Global Energy", units=units, system_label = system_label, r2=global_r2, 
                                   stderr=predicted_stdev, linear_fit_params = linear_fit_params, x_range=cropped_range, y_range=cropped_range, color=color, ms=5)
        elif plot_type == "global_log_error_histogram":
            variable_label="Global Energy"
            if variable_label not in error_info:
                error_info[variable_label]=dict(Units=units, **global_err_stats)
            plot_error_log_histogram(global_err, ax, variable_label = "{} Prediction".format(variable_label), units=units, system_label = system_label, exponent_range = exponent_range,
                             color=color)
        elif plot_type == "global_log_error_per_atom_histogram":
            variable_label="Global Energy/Atom"
            if variable_label not in error_info:
                error_info[variable_label]=dict(Units=local_units, **global_err_per_atom_stats)
            plot_error_log_histogram(global_err_per_atom, ax, variable_label = "{} Prediction".format(variable_label), units=local_units, system_label = system_label, 
                                     exponent_range = exponent_range, color=color)    
        elif plot_type == "local_log_error_histogram":
            variable_label="Local Energy"
            if variable_label not in error_info:
                error_info[variable_label] = dict(Units=local_units, **local_err_stats)
            plot_error_log_histogram(local_err, ax, variable_label = "{} Prediction".format(variable_label), units=local_units, system_label = system_label, 
                                     exponent_range = exponent_range, color=color)
        elif plot_type == "global_error_histogram":
            variable_label="Global Energy"
            if variable_label not in error_info:
                error_info[variable_label]=dict(Units=units, **global_err_stats)
            plot_error_histogram(global_err, ax, variable_label = "{} Prediction".format(variable_label), units=units, system_label = system_label, data_range = global_err_range,
                             color=color)
        elif plot_type == "global_error_per_atom_histogram":
            variable_label="Global Energy/Atom"
            if variable_label not in error_info:
                error_info[variable_label]=dict(Units=local_units, **global_err_per_atom_stats)
            plot_error_histogram(global_err_per_atom, ax, variable_label = "{} Prediction".format(variable_label), units=local_units, system_label = system_label, 
                                     data_range = local_err_range, color=color)    
        elif plot_type == "local_error_histogram":
            variable_label="Local Energy"
            if variable_label not in error_info:
                error_info[variable_label] = dict(Units=local_units, **local_err_stats)
            plot_error_histogram(local_err, ax, variable_label = "{} Prediction".format(variable_label), units=local_units, system_label = system_label, 
                                     data_range = local_err_range, color=color)            
        else:
            print( "Do not recognize {} as a plot type. Ignoring this input. Next time please choose from: {}.".format(plot_type, available_plot_types) )
            
    error_dataframe = compile_error_dataframe(error_info)
    if in_notebook:
        display(error_dataframe)
    else:
        print(error_dataframe)
    
    if analysis_settings.make_output_files:
        energy_plots_title = "energy_prediction_plots"
        energy_errors_plot_filename = output_directory + energy_plots_title + ".png"
        # check if existing, add number to end if it is
        energy_errors_plot_filename = find_unique_filename(energy_errors_plot_filename, verbose=verbose)
        plt.savefig(energy_errors_plot_filename)
        
        energy_errors_title = "energy_errors"
        energy_errors_filename = output_directory + energy_errors_title + ".csv"
        # check if existing, add number to end if it is
        energy_errors_filename = find_unique_filename(energy_errors_filename, verbose=verbose)
        error_dataframe.to_csv(energy_errors_filename)        
        
def AnalyzeForceResults(true_forces, predicted_forces, analysis_settings,
                predicted_force_stdevs=None, in_notebook=True, output_directory="../data/", 
                ax_height = 6, ax_width=7):
    plot_types = analysis_settings.force_plots.copy()
    component_types = analysis_settings.force_plots_components.copy()
    stat_types = analysis_settings.error_stats.copy()
    units = analysis_settings.output_force_units
    verbose=analysis_settings.verbose
    color = analysis_settings.color
    system_label = analysis_settings.title + " Calculation" if analysis_settings.title else ""
    
    # This code is messy. Probably needs a class of some sort (maybe an Axis class) to clean it up
    available_plot_types = ("predicted_vs_true", "error_histogram", "log_error_histogram")
    available_component_types = ( "x", "y", "z", "magnitude", "theta", "phi" )
    index_by_component = { available_component_types[i]:i for i in range(len(available_component_types)) }
    variable_label_by_component =  { "x":"Fx", "y":"Fy", "z":"Fz", "magnitude":"|F|", "phi":"φ", "theta":"θ" }
    unit_by_component =  { "x":units, "y":units, "z":units, "magnitude":units, "phi":"°", "theta":"°" }
    
    # We have to match up the axes boundaries differently for the angles so we treat them separately
    # I treat the magnitude plot, Cartesian component plots and angle plots separately when deterimining the boundaries for the true vs predicted plots
    # I treat the force plots and angle plots separately when deterimining the boundaries for the true vs predicted plots
    # We also have to assign different units to them
    last_component_index = 2
    first_angle_index = 4
    force_indices_in_use = np.array([index_by_component[c] for c in component_types ])
    angle_indices_in_use = force_indices_in_use[np.where(force_indices_in_use >= first_angle_index)[0]]
    component_indices_in_use = force_indices_in_use[np.where(force_indices_in_use <= last_component_index)[0]]
    force_indices_in_use = force_indices_in_use[np.where(force_indices_in_use < first_angle_index)[0]]
    
    # Append columns with magnitude, theta and phi
    true_forces =      np.concatenate((true_forces,      spherical_from_cartesian(true_forces,      angle_output="deg") ), axis = 1 )
    predicted_forces = np.concatenate((predicted_forces, spherical_from_cartesian(predicted_forces, angle_output="deg") ), axis = 1 )
    err_stats, err  = compile_error_stats( predicted_forces,  true_forces, stats_to_calculate = stat_types, verbose=verbose)
    
    
    # Only determine plot ranges if they will be used
    if len(force_indices_in_use):
        # For the log hisogram error plots
        exponent_range = get_exponent_range([err[:,i] for i in force_indices_in_use], max_min=-5, min_max=-1)
        # For the hisogram error plots
        err_range = get_data_bounds([err[:,i] for i in force_indices_in_use], buffer=0.05)
    if len(angle_indices_in_use):
        # For the hisogram error plots
        angle_err_range = get_data_bounds([err[:,i] for i in angle_indices_in_use], buffer=0.05)
        # For the log hisogram error plots
        angle_exponent_range = get_exponent_range([err[:,i] for i in angle_indices_in_use], max_min=-5, min_max=-1)
        # For the true vs predicted plots
        angle_range = get_data_bounds([[0, true_forces[:,i], predicted_forces[:,i]] for i in angle_indices_in_use], buffer=0.05)
    if len(component_indices_in_use):
        # For the true vs predicted plots
        force_range = get_data_bounds([[0, true_forces[:,i], predicted_forces[:,i]] for i in component_indices_in_use], buffer=0.05)
    if "magnitude" in component_types:
        # For the true vs predicted plots
        magnitude_range = get_data_bounds([true_forces[:,index_by_component["magnitude"]], predicted_forces[:,index_by_component["magnitude"]]], buffer=0.05)
    
    # for possible cropped true vs predicted plots
    # Currently (2021/12/14) we use an adjusted IQR method to check for outliers,
    # i.e. outliers exists outside of the IQR +/- 1.5 IQR and outliers must not be too close to non-outliers
    if analysis_settings.crop_force_outliers:
        # We don't crop unless necessary so this flag is False by default
        cropping_necessary = False
        # For Fx, Fy, Fz if we plot those
        if len(component_indices_in_use):
            if verbose:
                print("Checking if outliers exists among Cartesian force components predictions.")
            cropped_force_range = suggest_outlier_cropping_for_true_vs_predicted_plot(true_forces[:, component_indices_in_use],
                                                                                      predicted_forces[:, component_indices_in_use], buffer = 0, verbose=verbose)
            # The cropping must reduce the range by at least 25% or it isn't performed
            # If the cropping is small it might just confuse people without making a plot that is any more readable than the original
            if np.ptp(cropped_force_range) < 0.75 * np.ptp(force_range):
                cropping_necessary = True
        # For F magnitudes we plot those
        if "magnitude" in component_types:
            if verbose:
                print("Checking if outliers exists among force magnitude predictions.")
            cropped_magnitude_range = suggest_outlier_cropping_for_true_vs_predicted_plot(true_forces[:, index_by_component["magnitude"]],
                                                                                      predicted_forces[:, index_by_component["magnitude"]], buffer = 0, verbose=verbose)
            # The cropping must reduce the range by at least 25% or it isn't performed
            # If the cropping is small it might just confuse people without making a plot that is any more readable than the original
            if np.ptp(cropped_magnitude_range) < 0.75 * np.ptp(magnitude_range):
                cropping_necessary = True
        # For F angles if we plot those
        if len(angle_indices_in_use):
            if verbose:
                print("Checking if outliers exists among force angle predictions.")            
            cropped_angle_range = suggest_outlier_cropping_for_true_vs_predicted_plot(true_forces[:, angle_indices_in_use],
                                                                                      predicted_forces[:, angle_indices_in_use], buffer = 0, verbose=verbose)
            # The cropping must reduce the range by at least 25% or it isn't performed
            # If the cropping is small it might just confuse people without making a plot that is any more readable than the original
            if np.ptp(cropped_angle_range) < 0.75 * np.ptp(angle_range):
                cropping_necessary = True            
        
        if cropping_necessary:
            plot_types.append("predicted_vs_true_cropped")
        else:
            print("Cropping force plots determined to be unnecessary.")
            
            
    
    fig, axs = plt.subplots(figsize = (len(component_types) * ax_width, len(plot_types)*ax_height), 
                            nrows = len(plot_types), ncols=len(component_types), constrained_layout=True )
    error_info = {}
    for i in range(len(plot_types)):
        for j in range(len(component_types)):
            plot_type = plot_types[i]
            component_label = component_types[j]
            if component_label not in available_component_types:
                print( "Do not recognize {} as a component type. Ignoring this input. Next time please choose from: {}.".format(component_label, available_component_types) )
                continue
            variable_label = variable_label_by_component[component_label]
            units_j = unit_by_component[component_label]
            component_index = index_by_component[component_label]
            true_force_component = true_forces[:, component_index]
            predicted_force_component = predicted_forces[:, component_index]
            err_j = err[:, component_index]
            r2 = err_stats["r2"][component_index] if "r2" in stat_types else None
            r2 = r2 if np.isfinite(r2) else None
            linear_fit_params = (err_stats["m"][component_index], err_stats["b"][component_index]) if "linfit" in stat_types else None
            linear_fit_params = linear_fit_params if linear_fit_params != (None, None) else None
            predicted_stdev = predicted_force_stdevs[:, component_index] if predicted_force_stdevs is not None else None
            error_info_j = {key:value[component_index] for key,value in err_stats.items()}
            exponent_range_j = exponent_range if component_index < first_angle_index else angle_exponent_range
            err_range_j = err_range if component_index < first_angle_index else angle_err_range
            if component_index <= last_component_index:
                range_j = force_range
            elif component_index >= first_angle_index:
                range_j = angle_range
            else:
                range_j = magnitude_range
            
            ax = axs[i, j]
            
            if plot_type == "predicted_vs_true":
                plot_predicted_vs_true(predicted_force_component, true_force_component, ax, variable_label = variable_label, 
                                       units=units_j, linear_fit_params = linear_fit_params, x_range=range_j, y_range=range_j, 
                                       system_label = system_label, r2=r2, stderr=predicted_stdev, color=color, ms=5)
            elif plot_type == "predicted_vs_true_cropped":
                if component_index <= last_component_index:
                    range_j = cropped_force_range
                elif component_index >= first_angle_index:
                    range_j = cropped_angle_range
                else:
                    range_j = cropped_magnitude_range
                plot_predicted_vs_true(predicted_force_component, true_force_component, ax, variable_label = variable_label, 
                                       units=units_j, linear_fit_params = linear_fit_params, x_range=range_j, y_range=range_j, 
                                       system_label = system_label + " (Cropped)", r2=r2, stderr=predicted_stdev, color=color, ms=5)
            elif plot_type == "log_error_histogram":
                if variable_label not in error_info:
                    error_info[variable_label]=dict(Units=units_j, **error_info_j)
                plot_error_log_histogram(err_j, ax, variable_label = "{} Prediction".format(variable_label), units=units_j, system_label = system_label, 
                                     exponent_range = exponent_range_j, color=color)
            elif plot_type == "error_histogram":
                if variable_label not in error_info:
                    error_info[variable_label]=dict(Units=units_j, **error_info_j)
                plot_error_histogram(err_j, ax, variable_label = "{} Prediction".format(variable_label), units=units_j, system_label = system_label, 
                                     data_range = err_range, color=color) 
            else:
                print( "Do not recognize {} as a plot type. Ignoring this input. Next time please choose from: {}.".format(plot_type, available_plot_types) )
                
    error_dataframe = compile_error_dataframe(error_info)
    if in_notebook:
        display(error_dataframe)
    else:
        print(error_dataframe)

    if analysis_settings.make_output_files:
        force_plots_title = "force_prediction_plots"
        force_errors_plot_filename = output_directory + force_plots_title + ".png"
        # check if existing, add number to end if it is
        force_errors_plot_filename = find_unique_filename(force_errors_plot_filename, verbose=verbose)
        plt.savefig(force_errors_plot_filename)
        
        force_errors_title = "force_errors"
        force_errors_filename = output_directory + force_errors_title + ".csv"
        # check if existing, add number to end if it is
        force_errors_filename = find_unique_filename(force_errors_filename, verbose=verbose)
        error_dataframe.to_csv(force_errors_filename)    
