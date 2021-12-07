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
from general_purpose_helper_functions import make_unique_directory
from ASE_helper_functions import assign_precalculated_energy
from Visualize_Structures import Structure3DAnimation
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



def import_structures(filename, n_structs, verbose=False, in_notebook=True, miniGAP_parent_directory = "../", by_indices=False, indices=[]):
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
    elif filetype in [".xyz", ".extxyz", ".extxyz.gz", ".xyz.gz"]:
        print("If you plan to use miniGAP with this dataset more than once, it is recommended that you convert this file to a .db file. \
        \nThe conversion may be slow, but this will save significant time each run of miniGAP thereafter. \
        \nTo perform this conversion, run the command 'code/convert_to_db.py {}' from the minigap parent directory".format(filename))
        if verbose:
            print("Imported the first {} structures from {} as our dataset.\
            \nIt would be better to drawn on structures evenly from throughout your full dataset instead of taking only from the beginning.\
            \nHowever, it would be prohibitively slow to import all structures from a file other than an ASE database file during every miniGAP run.\
            \nThis is partially why conversion to a .db is recommended.".format(n_structs, filename))
        StructureList = read(filename, ":{}".format(n_structs))
    elif filetype == ".db":
        # from ase.db import connect
        # db = connect(filename)
        # FullStructureList = []
        # for row in db.select():
        #     FullStructureList.append(db.get_atoms(id = row.id))
        FullStructureList = read(filename, ":")
        StructureList = FullStructureList[::len(FullStructureList)//n_structs][:n_structs]
        if verbose:
            print("Imported {} structures from {}. Structures were taken uniformly from throughout dataset which contains {} total structures.".format(len(StructureList), filename, len(FullStructureList)))
    else:
        print("Do not currently recognize filetype {}. However, it is possible ASE can read this filetype. \
        \nIf so, modify import_structures function to enable import from {}.".format(filetype, filename))
        
        if not in_notebook:
            exit()
    return StructureList


def GenerateMDTrajForMiniGAP(structure, from_diatomic, md_settings, miniGAP_parent_directory="../", traj_directory = "../data/"):
    return generate_md_traj(structure=structure, from_diatomic=from_diatomic, preoptimize=False, bond_length=md_settings.diatomic_bond_length, 
                            element=md_settings.diatomic_element, temperature=md_settings.md_temp, nsteps=md_settings.n_structs,
                            md_type = md_settings.md_algorithm, calc_type=md_settings.md_energy_calculator, md_seed= md_settings.md_seed, 
                            time_step=md_settings.md_time_step, verbose=md_settings.verbose, print_step_size=md_settings.n_structs/10, 
                            plot_energies="off", parent_directory=miniGAP_parent_directory, traj_directory = traj_directory)

def CompileStructureList(structure_settings, in_notebook, miniGAP_parent_directory, output_directory="../results/"):
    if structure_settings.structure_file in (None, "None", ""):
        if structure_settings.chemical_formula in (None, "None", ""):
            # Diatomic molecule in MD trajectory
            struct_list = GenerateMDTrajForMiniGAP(structure=None, from_diatomic=True, md_settings=structure_settings, 
                                                   miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )

            # Useful for debugging purposes
            # Diatomic molecule with evenly spaced bond lengths
            # struct_list = [make_diatomic(element = structure_settings.diatomic_element, verbose=False, bond_length=L, \
            # calc_type=structure_settings.md_energy_calculator) for L in np.linspace(.6,1.8, structure_settings.n_structs)]

        else:
            # ASE creates this g2 collection 'lazily' so we must call each member before we can get a complete list of names
            g2_list = [m for m in g2]
            del g2_list
            if structure_settings.chemical_formula in g2._names:
                struct_list = GenerateMDTrajForMiniGAP(structure=molecule(structure_settings.chemical_formula), from_diatomic=False, 
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
            starter_struct = import_structures(structure_settings.structure_file, structure_settings.n_structs, structure_settings.verbose, in_notebook, miniGAP_parent_directory, 
                                               by_indices=True, indices=[structure_settings.md_index])[0]
            struct_list = GenerateMDTrajForMiniGAP(structure=starter_struct, from_diatomic=False, md_settings=structure_settings, 
                                                   miniGAP_parent_directory=miniGAP_parent_directory, traj_directory=output_directory )
        else:
            struct_list = import_structures( structure_settings.structure_file, structure_settings.n_structs, structure_settings.verbose, in_notebook, miniGAP_parent_directory)
    return struct_list 


def PrepareDataForTraining(sp_list,
                           dsp_dx_list,
                           en_list,
                           frc_list ,
                           pos_list ,
                           split_seed,
                           prepare_forces ,
                           train_fract ,
                           scale_soaps
                          ):
    # This comment itself needs to be split up now haha
    
    # Split all data into training and test sets.
    # Test sets will not be viewed until training is done
    # Intra-structural information is not reshaped into local atomic information until after splitting
    # This means structures will be fully in the training or test sets, not split between both
    # It also means it will be possible to predict global energies in the test set
    
    train_indices, test_indices  = train_test_split(np.arange(len(sp_list)), random_state = split_seed, test_size = 1 - train_fract )
    
    train_ens, test_ens = en_list[train_indices], en_list[test_indices] #, test_ens, train_frcs, test_frcs
    train_ens, test_ens = train_ens.reshape(-1, 1), test_ens.reshape(-1, 1)
    
    # Scale energies to have zero mean and unit variance.
    # Divide forces by the same scale factor but don't subtract the mean

    ens_scaler = StandardScaler().fit(train_ens)
    train_ens, test_ens = ens_scaler.transform(train_ens), ens_scaler.transform(test_ens)
    # The following line is for the tensorflow code. If it is commented out, it is for the gpflow code
    #train_ens, test_ens = train_ens.flatten(), test_ens.flatten()
    ens_var = train_ens.var()
    
    if prepare_forces:
        train_frcs, test_frcs = frc_list[train_indices], frc_list[test_indices]
        train_frcs, test_frcs = train_frcs.reshape(-1, 3), test_frcs.reshape(-1, 3)
        train_frcs, test_frcs = train_frcs / ens_scaler.scale_, test_frcs / ens_scaler.scale_
        frcs_var = train_frcs.var()
            
    
#     split_data = train_test_split(sp_list, dsp_dx_list, en_list, frc_list, pos_list, np.arange(len(sp_list)), random_state = split_seed, test_size = 1 - train_fract )
#     print([np.array(x).shape for x in split_data])
#     split_data = [np.array]
#     train_sps_full, test_sps_full, train_dsp_dx, test_dsp_dx, train_ens, test_ens, train_frcs, test_frcs, train_pos, test_pos, train_indices, test_indices = split_data
    
    train_sps_full, test_sps_full = sp_list[train_indices], sp_list[test_indices]
    train_sps_full, test_sps_full = train_sps_full.reshape(-1, sp_list.shape[-1]), test_sps_full.reshape(-1, sp_list.shape[-1])  
    
    # Scale soaps to have zero mean and unit variance.
    # Divide derivatives by the same scale factor but don't subtract the mean
    
    soap_scaler = StandardScaler().fit(train_sps_full)
    if scale_soaps:
        train_sps_full, test_sps_full = soap_scaler.transform(train_sps_full), soap_scaler.transform(test_sps_full)
    
    if prepare_forces:
        train_dsp_dx, test_dsp_dx = dsp_dx_list[train_indices], dsp_dx_list[test_indices]
        train_dsp_dx, test_dsp_dx = train_dsp_dx.reshape(-1, 3, dsp_dx_list.shape[-1]), test_dsp_dx.reshape(-1, 3, dsp_dx_list.shape[-1])
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
        return train_sps_full, test_sps_full, train_ens, test_ens, train_indices, test_indices, soap_scaler, ens_scaler, ens_var
    else:
        return train_sps_full, test_sps_full, train_ens, test_ens, train_indices, test_indices, soap_scaler, ens_scaler, ens_var, train_dsp_dx, test_dsp_dx, train_frcs, test_frcs, frcs_var


def self_energy(element, use_librascal_values=False, dtype=np.float64):
    if use_librascal_values:
        self_contributions = {
            "H": -6.492647589968434,
            "C": -38.054950840332474,
            "O": -83.97955098636527,
        }
        return self_contributions[element]
    else:
        return np.array(0, dtype=dtype)    
    
    
def RetrieveEnergyFromASE(struct, alt_keyword=""):
    if struct.calc != None:
        return struct.get_potential_energy()
    else:
        return struct.info[alt_keyword]

    

def GatherStructureInfo(struct_list, gather_forces = True, use_self_energies=True, alt_energy_keyword="", dtype=np.float64):

    pos_list =  np.array([atoms.positions for atoms in struct_list])
    
    en_list = np.array([[RetrieveEnergyFromASE(struct, alt_keyword = alt_energy_keyword)/len(struct) \
                         - self_energy(atom.symbol, use_self_energies, dtype=dtype) for atom in struct] for struct in struct_list], dtype=dtype)
    
    n_atom_list = np.array([len(struct) for struct in struct_list])

    if gather_forces:
        frc_list = np.array([atom.get_forces() for atom in struct_list], dtype=np.float64)
    else:
        frc_list = [no_forces_string] * len(en_list)
        
    return en_list, frc_list, pos_list

def GenerateDescriptorsAndDerivatives(struct_list, nmax, lmax, rcut, smear=0.3, attach=True, is_periodic=False, return_derivatives=True, get_local_descriptors = True):
    
    relevant_species = np.unique(struct_list[0].get_chemical_symbols())
    
    dscribe_output = get_dscribe_descriptors(
                                            struct_list, 
                                            species=relevant_species, 
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
        tf.print("predict energies = ", predict_energy[:3])
        my_mse = mse(predict_energy, valid_energy)
    gradients = tape.gradient(my_mse, model.trainable_variables)
    tf.print("gradients = ", gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("TRACING train_hyperparams_without_forces")
    return my_mse

# @tf.function(autograph=False, experimental_compile=False)
def predict_energies_from_weights(c, soaps_old, soaps_new, degree):
    k = tf.math.pow( tf.tensordot(soaps_old, tf.transpose(soaps_new), axes=1), degree )
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
            if kernel_settings["degree"] != 1:
                print("Alert: Double check the training validity for degree =/= 1 when not using predict_f".format(kernel_settings["degree"]))

        return kernel
    else:
        print("Warning: Do not recognize kernel_type={}".format(kernel_type))


        
def create_miniGAP_visualization(struct_list, visualization_settings, output_directory, animation_filename="DEFAULT_FILENAME"):
    save_file = visualization_settings.save_dataset_animation & visualization_settings.make_output_files
    verbose = visualization_settings.verbose
    animation_object = Structure3DAnimation(struct_list)
    animation_html5 = animation_object.Plot()
    if save_file:
        animation_filename = output_directory + animation_filename.replace("DEFAULT_FILENAME", "structure_animation.mp4")
        animation_object.Save(animation_filename)
    return animation_html5