from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skcosmo.sample_selection import CUR as CURSample
from skcosmo.sample_selection import PCovCUR as PCovCURSample
from skcosmo.feature_selection import CUR as CURFeature
from skcosmo.feature_selection import PCovCUR as PCovCURFeature
# ----------------------------------------------------------------------------
import sys
sys.path.append('../code')
from Generate_Descriptors import get_dscribe_descriptors
# ----------------------------------------------------------------------------
no_forces_string = "Not Using Forces"
# ----------------------------------------------------------------------------



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
    
    
def RetrieveEnergyFromASE(struct, method="normal", keyword=""):
    if method == "normal":
        return struct.get_potential_energy()
    elif method == "info":
        return struct.info[keyword]
    else:
        print("Cannot interpret method {}".format(method))
    

def GatherStructureInfo(struct_list, gather_forces = True, use_self_energies=True, energy_encoding="normal", energy_keyword="", dtype=np.float64):

    pos_list =  np.array([atoms.positions for atoms in struct_list])
    
    en_list = np.array([[RetrieveEnergyFromASE(struct, method=energy_encoding, keyword = energy_keyword)/len(struct) \
                         - self_energy(atom.symbol, use_self_energies, dtype=dtype) for atom in struct] for struct in struct_list], dtype=dtype)

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