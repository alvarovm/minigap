#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../code')
from miniGAP_helper_functions import in_notebook

if not in_notebook():
    import time
    TimeBeforeStartUp = time.time()


# In[2]:


tf_cpu_debugging =False
if tf_cpu_debugging:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    #tf.debugging.set_log_device_placement(True)
    a = tf.constant(1)
    
    if tf.test.gpu_device_name():
        print("GPUs recognized by tensorflow:", tf.config.list_physical_devices('GPU'))
    else:
        print("No GPU found")

tf_gpu_debugging = False
if tf_gpu_debugging:
#     See here for possible option to clear memory github.com/tensorflow/tensorflow/issues/36465
#     import os 
#     os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    import tensorflow as tf
    tf.debugging.set_log_device_placement(True)
    a = tf.constant(1)
    if tf.test.gpu_device_name():
        print("GPUs recognized by tensorflow:", tf.config.list_physical_devices('GPU'))
    else:
        print("No GPU found")
    


# main miniGAP notebook

# In[3]:


# import functions from my files
import sys
sys.path.append('../code')
from Molecular_Dynamics import generate_md_traj, make_diatomic
from miniGAP_helper_functions import *

# import functions from modules
import os.path as path
import time
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gpflow
from itertools import islice
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
import json
from collections import namedtuple

if in_notebook():
    gpflow.config.set_default_summary_fmt("notebook")
else:
    gpflow.config.set_default_summary_fmt("grid")


# In[4]:


mse_tf = tf.function(mse, autograph=False, jit_compile=False)
mse_2factor_tf = tf.function(mse_2factor, autograph=False, jit_compile=False)
train_hyperparams_without_forces_tf = tf.function(train_hyperparams_without_forces, autograph=False, jit_compile=False)
predict_energies_from_weights_tf = tf.function(predict_energies_from_weights, autograph=False, jit_compile=False)


# In[5]:


if in_notebook():
    miniGAP_parent_directory = "../"
else:
    miniGAP_parent_directory = path.dirname(path.dirname(path.realpath(__file__))) + "/"


# In[6]:


if not in_notebook():
    import argparse
    
    # --------------------------------------------  commandline argument code   -----------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', type=bool, help="Print out details at each step") 
    parser.add_argument('-vt', '--print_timings', type=bool, help="Print out details at each step") 

    # arguments specific to forming dataset (including potentially creating md trajectory)
    parser.add_argument('-sf', '--structure_file', help="Specify a structure file to import. 'None' will be interpretted as using no structure file.")
    parser.add_argument('-cf', '--chemical_formula',  help="If no structure file is supplied, you can specify a single structure here and perform md     to generate a trajectory that you will use as your dataset. If neither this nor a structure file are supplied, we will use diatomics.")
    parser.add_argument("-md", '--molecular_dynamics', type=bool, help="Indicate if you want molecular dynamics performed. Will generate diatomic if no structure given")
#     parser.add_argument('-mdi', '--md_indices', default=[0], type=int, nargs='*', help="If performing molecular dynamics on a structure file with multiple structures, you can give indices of all structures to perform md on.")
    parser.add_argument('-mdi', '--md_index', type=int, help="If performing molecular dynamics on a structure file with multiple structures, you can give the index of the structure to perform md on.")
    parser.add_argument('-de', '--diatomic_element',  choices = ["N", "O", "H"], help="If generating diatomics, you can specify element")
    parser.add_argument('-dbl', '--diatomic_bond_length',  type=float, help="If generating diatomics, you can specify initial bond length")
    parser.add_argument('-mdt', '--md_temp',  type=float, help="If performing molecular dynamics, specify temperatutre (K) of MD")
    parser.add_argument('-mda', '--md_algorithm',  choices = ["VelocityVerlet", "Berendsen"], type=str, help="If performing molecular dynamics, specify algorithm of MD")
    parser.add_argument('-mts', '--md_time_step',  type=float, help="If performing molecular dynamics, specify time step (fs) of MD")
    parser.add_argument('-mds', '--md_seed',  type=int, help="If performing molecular dynamics, change this seed to get different trajectories")
    parser.add_argument('-mec', '--md_energy_calculator',  choices = ["EMT", "LJ", "Morse"], help = "If performing molecular dynamics, specify ASE energy/force calculator")
    parser.add_argument('-n', '--n_structs',  type=int, help="Specify # of md generated structures or # of structures to use from input file")

    # arguments specific to soap
    parser.add_argument('--rcut',  type=float, help= "Choice of SOAP cut off radius")
    parser.add_argument('--nmax',  type=int, help="Choice of SOAP n_max")
    parser.add_argument('--lmax',  type=int, help="Choice of SOAP l_max")

    # arguments specific to learning
    parser.add_argument('-ss', '--split_seed', type=int, help="Random seed for cross-validation")
    parser.add_argument('-tf', '--train_fraction', type=float, help="Specify the fraction of structures used in training")
    parser.add_argument('-ne', '--n_epochs', type=int, help="Number of epochs")

    # some housekeeping
#     parser.add_argument('remainder', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    cmdline_args = parser.parse_args()
    cmdline_args_dict = vars(cmdline_args)


# In[7]:


# Input parameter important notes:

# energy_encoding=#"info" for QM9 or "normal" for distorted propenols
# energy_keyword="U0" for QM9 or ignored for distorted propenols

# my_priority = #"efficiency" for experimenting with something new or otherwise "consistency"

# # controls initial train_test_split breaking apart training and test data
# split_seed = 2

# # controls in-training train_test_split breaking apart training_j and validation_j data
# valid_split_seed = 2

# # controls multiple tf stochastic processes including:
# # 1) Adams optimizer AND 
# # 2) batching through tf.data.Dataset.from_tensor_slices in training
# tf_seed = 2

# # controls batching through tf.data.Dataset.from_tensor_slices in training
# shuffle_seed = 2

# kernel_type = #"polynomial" for actual GAP or "exponentiated_quadratic" possibly for debugging

# prediction_calculation = #"direct" OR "predict_f" OR "cholesky" OR "alpha"

settings_json_filename = miniGAP_parent_directory + "code/miniGAP_settings.json"
with open(settings_json_filename, encoding = 'utf-8') as settings_file_object:
    default_settings_dict = json.load(settings_file_object)

settings_dict = default_settings_dict
if not in_notebook():
    for setting_name in settings_dict.keys():
        if setting_name in cmdline_args_dict.keys():
            if cmdline_args_dict[setting_name] != None:
                settings_dict[setting_name] = cmdline_args_dict[setting_name]
    
    for setting_name in cmdline_args_dict.keys():
        if setting_name not in settings_dict.keys():
            print("The commandline argument {} is currently nonfunctional because it does not exist in the JSON file.".format(setting_name))
SettingsNamespace = namedtuple("Settings", settings_dict.keys())
s = SettingsNamespace(*settings_dict.values())


# In[8]:


import resource
try:
    resource.setrlimit( resource.RLIMIT_STACK, ( resource.RLIM_INFINITY, resource.RLIM_INFINITY ) )
except:
    print("Warning: Unable to raise stack size limit. Typically miniGAP uses a lot of memory and operates better if you allow the stack size to be unlimited. You can try to do this with the command 'ulimit -s unlimited'")

if s.verbose:
    print("Stack size set to unlimited")

    print("{} GPU(s) recognized by tensorflow:".format(len(tf.config.list_physical_devices('GPU'))), tf.config.list_physical_devices('GPU'))


# 1) problem with json if run file outside of code directory  
# 2) verbose statement nto working

# In[9]:


StructureList, TimeCompileStructures = TickTock(CompileStructureList, s, in_notebook(), miniGAP_parent_directory)
if s.print_timings:
    print("Compiling structures into list took {:.2f} seconds".format(TimeCompileStructures))


ns_atoms = np.unique([len(struct) for struct in StructureList])
assert len(ns_atoms) == 1
n_atoms = ns_atoms[0]


# In[10]:


# Set convert_to_db_here_and_now to True and run this cell to convert your file to a database right now from this notebook
# Note: I do not recommend this. The database creation is very slow and will lock you out from running any cells for a long time.
#       It is better to do this from a terminal.
convert_to_db_here_and_now = False
if convert_to_db_here_and_now and in_notebook():
    get_ipython().system('python ../code/convert_to_db.py $s.structure_file ')
    # Use the following line if you need to overwrite an existing database
    # !python ../code/convert_to_db.py -efb overwrite $s.structure_file


# In[11]:


if s.print_timings and not in_notebook():
    TimeAfterStartUp = time.time()
    TimeStartUp = TimeAfterStartUp - TimeBeforeStartUp
    print("Completed various initialization tasks (including importing structures) in {:.2f} seconds".format(TimeStartUp))


# In[12]:


check_structures_visually = False
if check_structures_visually:
    from Visualize_Structures import Structure3DAnimation, Structure3DPlot
    html_animation = Structure3DAnimation(StructureList, azimuth=45, elevation=45).Plot()
    display(html_animation)


# In[13]:


[EnergyList, ForceList, PosList], TimeGather = TickTock(GatherStructureInfo, StructureList, gather_forces = s.use_forces, use_self_energies=s.use_self_energies, 
                                                     energy_encoding = s.energy_encoding,  energy_keyword = s.energy_keyword)
gather_forces_message= ", force" if s.use_forces else ""
if s.print_timings:
    print("Gathered energy{} and structure info in {:.2f} seconds".format(gather_forces_message, TimeGather))


# In[14]:


[SoapDerivativeList, SoapList], TimeSoap = TickTock(GenerateDescriptorsAndDerivatives, StructureList, s.nmax, s.lmax, s.rcut, s.smear, s.attach_SOAP_center, s.is_periodic, s.use_forces)
calculate_derivatives_message = " and derivatives" if s.use_forces else ""
if s.print_timings:
    print("Generated SOAP descriptors{} in {:.2f} seconds".format(calculate_derivatives_message, TimeSoap))
elif s.verbose:
    print("Generated SOAP descriptors{}.".format(calculate_derivatives_message))


# In[15]:


out_data, TimePrepare = TickTock(PrepareDataForTraining, 
                                sp_list=SoapList, 
                                dsp_dx_list = SoapDerivativeList, 
                                en_list = EnergyList,
                                frc_list = ForceList ,
                                pos_list = PosList, 
                                split_seed = s.split_seed, 
                                prepare_forces = s.use_forces, 
                                train_fract = s.train_fraction,
                                scale_soaps = s.scale_soaps
                                )

if s.print_timings:
    print("Reformatted data to build model in {:.2f} seconds.".format(TimePrepare))

if not s.use_forces:
    train_sps_full, test_sps_full, train_ens, test_ens, train_indices, test_indices, soap_scaler, ens_scaler, ens_var = out_data
else:
    train_sps_full, test_sps_full, train_ens, test_ens, train_indices, test_indices, soap_scaler, ens_scaler, ens_var, train_dsp_dx, test_dsp_dx, train_frcs, test_frcs, frcs_var = out_data 


# In[16]:


n_samples_full, n_features_full = train_sps_full.shape
SparsifySoapsOutput, TimeSparsify = TickTock( SparsifySoaps, train_soaps = train_sps_full, test_soaps = test_sps_full, train_energies=train_ens, sparsify_samples=s.sparse_gpflow, 
                                    n_samples=s.n_sparse, sparsify_features=s.sparse_features, n_features=s.n_sparse_features, selection_method="PCovCUR",
                                    score_tolerance=1e-5, score_threshold=1e-5, iterative_selection=False, plot_importances=False) 
train_sps, sparse_train_sps, test_sps = SparsifySoapsOutput
if s.sparse_gpflow or s.sparse_features:
    sparsity_samples_message = "" if not s.sparse_gpflow else "samples ({} --> {})".format(n_samples_full, sparse_train_sps.shape[0])
    sparsity_features_message = "" if not s.sparse_features else "features ({} --> {})".format(n_features_full, train_sps.shape[-1])
    sparsity_both_message = " and " if (s.sparse_gpflow and s.sparse_features) else ""
    if s.print_timings:
        print("Sparsified model {}{}{} in {:.2f} seconds.".format( sparsity_samples_message, sparsity_both_message, sparsity_features_message, TimeSparsify) )
    elif s.verbose:
        print("Sparsified model {}{}{}".format( sparsity_samples_message, sparsity_both_message, sparsity_features_message) )
else:
    if s.verbose:
        print("Not using any sparsity.")


# ### Future investigations for hyperparameter training
# ---
# 1. Does Adam optimizer have problems sometimes within a tf.function?
# 2. Custom loss function vs optimizer.minimize
# 3. Set certain variable untrainable

# In[17]:


# Initialize kernels and model hyperparameters
tf.random.set_seed(s.tf_seed)

TimeBeforePreEpoch = time.time()


noise_init = 1e-4 #.001# 0.0005499093576274776 #1.625e-4
obs_noise = tf.Variable(noise_init, dtype=s.dtype, name="noise")

degree=2
kernel = pick_kernel(s.kernel_type, amplitude=1, verbose=s.verbose, degree=degree)




# Split training data into training and validation sets
# Now validation set acts as temporary est set
# train_test_split and tensorflow tensors don't get along so I temporarily convert them back to numpy arrays

train_indices_j, valid_indices_j  = train_test_split(np.arange(len(train_sps)), random_state = s.valid_split_seed, test_size=(1-s.valid_fract))

train_sps_j, valid_sps_j = train_sps[train_indices_j], train_sps[valid_indices_j]
train_ens_j, valid_ens_j = train_ens[train_indices_j], train_ens[valid_indices_j]

if s.use_forces: 
    train_dsp_dx_j, valid_dsp_dx_j = train_dsp_dx[train_indices_j], train_dsp_dx[valid_indices_j]
    train_frcs_j, valid_frcs_j = train_frcs[train_indices_j], train_frcs[valid_indices_j]

# Convert to tensorflow constant tensors
# train_sps_j = tf.constant(train_sps_j, dtype=s.dtype)
# train_ens_j = tf.constant(train_ens_j, dtype=s.dtype)
# valid_sps_j = tf.constant(valid_sps_j, dtype=s.dtype)
# valid_ens_j = tf.constant(valid_ens_j, dtype=s.dtype)
if s.sparse_gpflow:
    sparse_train_sps = tf.Variable(sparse_train_sps, shape=sparse_train_sps.shape, dtype=s.dtype, trainable=False)

if s.use_forces:
    train_dsp_dx_j = tf.constant(train_dsp_dx_j, dtype=s.dtype)
    train_frcs_j = tf.constant(train_frcs_j, dtype=s.dtype)    
    valid_dsp_dx_j = tf.constant(valid_dsp_dx_j, dtype=s.dtype)
    valid_frcs_j = tf.constant(valid_frcs_j, dtype=s.dtype)        

test_sps = tf.constant(test_sps, dtype=s.dtype)


# Batch data if  training set is larger than batch_size_max
if len(train_sps_j) < s.batch_size_max:
    iterations_per_epoch = 1
    batch_size = len(train_sps_j)
    if s.verbose:
        print("Training using {} atoms without batching.".format(len(train_sps_j)))
else:
    iterations_per_epoch = int(np.ceil(len(train_sps_j)/s.batch_size_max))
    batch_size = int(np.ceil(len(train_sps_j)/iterations_per_epoch))
    if s.verbose:
        print("Training using {} atoms total using {} batches with {} atoms per batch.".format( len(train_sps_j), iterations_per_epoch, batch_size ))

# training(out_data)
        
TimeBeforeEpoch0 = time.time()


mse_history = []    
hyperparam_history = []



# import warnings
# 
# from gpflow.models import VGP, GPR, SGPR, SVGP
# from gpflow.optimizers import NaturalGradient
# from gpflow.optimizers.natgrad import XiSqrtMeanVar
# from gpflow import set_trainable


if s.my_priority == "efficiency":
    # I don't know what this does
    autotune = tf.data.experimental.AUTOTUNE
    
    
    batches = (
        tf.data.Dataset.from_tensor_slices((train_sps_j, train_ens_j))
        .prefetch(autotune) 
        .shuffle(buffer_size=len(train_sps_j), seed=s.shuffle_seed)
        .repeat(count=None)
        .batch(batch_size)
    )
    
    batch_iterator = iter(batches)

    # I also don't know why we use this
    from gpflow.ci_utils import ci_niter, ci_range
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=s.learn_rate)
    
else:
    batches = (
        tf.data.Dataset.from_tensor_slices((train_sps_j, train_ens_j)) 
        .shuffle(buffer_size=len(train_sps_j), seed=s.shuffle_seed) 
        .repeat(count=None)
        .batch(batch_size)
    )    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=s.learn_rate)
#     optimizer = tf.keras.optimizers.SGD(learning_rate=s.learn_rate)
    
train_hyperparams_without_forces_tf = tf.function(train_hyperparams_without_forces, autograph=False, jit_compile=False)

# new code to make tf.function training work
# --------------------------------------------
train_sps_j_i = tf.Variable(train_sps[:batch_size], shape=(batch_size, train_sps.shape[-1]), dtype=s.dtype, trainable=False )
train_ens_j_i = tf.Variable(train_ens[:batch_size], shape=(batch_size, 1), dtype=s.dtype, trainable=False ) 
if s.sparse_gpflow:
    if sparse_train_sps.shape[0] >= batch_size:
        print("Warning: Batch size is not greater than sparse soap size.\nThis may cause errors in the predict_f function which assumes the inducing points to be fewer than the data points.")
    if s.my_priority == "efficiency":
        gpr_model = gpflow.models.SVGP( kernel=kernel, likelihood=gpflow.likelihoods.Gaussian(),  inducing_variable=sparse_train_sps)
        gpr_model.likelihood.variance.assign(obs_noise)
        gpflow.set_trainable(gpr_model.q_mu, False)
        gpflow.set_trainable(gpr_model.q_sqrt, False)
    else:
        gpr_model = gpflow.models.SGPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel, noise_variance=obs_noise, inducing_variable=sparse_train_sps)
else:
    if s.my_priority == "efficiency":
        # it seems I cannot use  noise_variance=obs_noise for this which makes it not GAP...
        gpr_model = gpflow.models.VGP( data=(train_sps_j_i, train_ens_j_i), kernel=kernel, likelihood=gpflow.likelihoods.Gaussian())
    else:
        gpr_model = gpflow.models.GPR( data=(train_sps_j_i, train_ens_j_i), kernel=kernel, noise_variance=obs_noise)
# --------------------------------------------


print_frequency = max(s.min_print_frequency, int(s.n_epochs/10))

if s.my_priority == "efficiency":
    hyperparam_history.append([(0, np.exp(var.numpy() )) for var in gpr_model.trainable_variables])  
    gpr_objective = gpr_model.training_loss_closure(batch_iterator,  compile=True)
    for j in range(ci_niter(s.n_epochs)):
        if not j % print_frequency:
            print("Epoch {}".format(j))
        optimizer.minimize(gpr_objective, var_list=gpr_model.trainable_variables)
        mse_history.append((j+1, gpr_model.elbo(data=(train_sps, train_ens))))
        hyperparam_history.append([(j+1, np.exp(var.numpy()) ) for var in gpr_model.trainable_variables]) 
    #optimizer.minimize(gpr_model.training_loss, gpr_model.trainable_variables, options=dict(maxiter=s.n_epochs))
        
elif s.my_priority == "consistency":

    hyperparam_history.append([(0, var.numpy()) for var in gpr_model.trainable_parameters])  
    for j in range(s.n_epochs):
        if not j % print_frequency:
            print("Epoch {}".format(j))
            #print(" ".join(["{} = {:.2e} ".format(var.name, np.exp(var.numpy())) for var in trainable_variables]))

        mse_ens_j = 0
        for i, (train_sps_j_i, train_ens_j_i) in enumerate(islice(batches, iterations_per_epoch)):
            if not s.use_forces: #and not s.sparse_gpflow :
                gpr_model.data[0].assign(train_sps_j_i)
                gpr_model.data[1].assign(train_ens_j_i)        
                mse_ens_j_i = train_hyperparams_without_forces_tf(gpr_model, valid_sps_j, valid_ens_j, optimizer)
                print("valid_ens[:3] = {}".format( valid_ens_j[:3].numpy().flatten()) )
#                 print(mse_ens_j_i.numpy(), valid_ens_j[:3].numpy().flatten(), train_ens_j_i[:3].numpy().flatten()  )


            else:
                print("Using older approach (not converted to tf.function yet)")
                with tf.GradientTape() as tape:
                    with tf.GradientTape(watch_accessed_variables=False) as tape_sps:
                        tape_sps.watch(valid_sps_j)
                        if s.sparse_gpflow:
                            gpr_model = gpflow.models.SGPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel, inducing_variable=sparse_train_sps)
    #                         gpflow.set_trainable(gpr_model.inducing_variable, False)
                            if i < 3:
                                print_summary(gpr_model)            
                        else:
                            gpr_model.data[0].assign(train_sps_j_i)
                            gpr_model.data[1].assign(train_ens_j_i)
                            #gpr_model = gpflow.models.GPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel)
                        #gpr_model.likelihood.variance.assign(obs_noise)                
                        predict_ens_j_i = gpr_model.predict_f(valid_sps_j)[0]

        #                 gpr_model = gpflow.models.GPR(data=(sps_j_i, train_ens_j_i), kernel=kernel_gpf)
        #                 gpr_model.likelihood.variance.assign(obs_noise_gpf)
        #                 predict_ens_j_i_gpf = gpr_model.predict_f(valid_sps_j)

                    if s.use_forces:
                        predict_d_ens_j_i = tape_sps.gradient(predict_ens_j_i, valid_sps_j)
                        # In the following line I needed to include '* n_atoms' after breaking energies into local energies
                        # The reason is that I am effectively breaking the connection between E and F when doing that
                        # F = -dE/dx =/= -dE_local/dx where E_local = E/n_atoms - E_free
                        # When I split energies into local energies I initially calculated -dE_local/dx which is -dE/dx / n_atoms
                        # This fix is prone to breaking the code and is not robust to systems with different structure size
                        # Need to improve this with a better fix
                        predict_frcs_j_i = -1*np.einsum('ijk,ik->ij', valid_dsp_dx_j, predict_d_ens_j_i) * n_atoms
                        mse_j_i = mse_2factor_tf(predict_ens_j_i, valid_ens_j, 1/ens_var,
                                                predict_frcs_j_i, valid_frcs_j, 1/frcs_var)
                        mse_ens_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
                    else:
                        mse_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
                        mse_ens_j_i = mse_j_i


        #         grads = tape.gradient(mse_j_i, trainable_variables)
        #         optimizer.apply_gradients(zip(grads, trainable_variables))
                grads = tape.gradient(mse_j_i, gpr_model.trainable_variables)
                # print(gpr_model.trainable_variables[0])#grads[0])
                optimizer.apply_gradients(zip(grads, gpr_model.trainable_variables))
                if i < 3:
                    print_summary(gpr_model)

                if not gpr_model.data[0][0,0].numpy() == train_sps_j_i[0,0].numpy() :
                    print("ERRORERRORERRORERRORERRORERRORERROR")

            print("Adding mse_ens_j_i to mse_ens_j: {} + {} = {} ".format(mse_ens_j_i.numpy(), mse_ens_j , mse_ens_j_i.numpy() + mse_ens_j  ))
            mse_ens_j += mse_ens_j_i

        mse_ens_j /= iterations_per_epoch
        print("Epoch {},  mse = {}".format(j, mse_ens_j))
        mse_history.append((j+1, mse_ens_j))
        hyperparam_history.append([(j+1, var.numpy()) for var in gpr_model.trainable_parameters])    
else:
    print("{} is not a reconized value for my_priority.\n Training did not occur.".format(s.my_priority))



TimeBeforeWeights = time.time()


# In[18]:


# I am currently (11/30) converting the hyperparameter learning in this cell into a function
# Next will be the post-hyperparameter part of the learning (in the next cell)

# # Initialize kernels and model hyperparameters



# def train_hyperparams(train_sps, train_ens, sparse_train_sps, kernel, settings):
#     tf.random.set_seed(settings.tf_seed)


#     noise_init = 1e-4 #.001# 0.0005499093576274776 #1.625e-4
#     obs_noise = tf.Variable(noise_init, dtype=settings.dtype, name="noise")





#     # Split training data into training and validation sets
#     # Now validation set acts as temporary est set
#     # train_test_split and tensorflow tensors don't get along so I temporarily convert them back to numpy arrays

#     train_indices_j, valid_indices_j  = train_test_split(np.arange(len(train_sps)), random_state = settings.valid_split_seed, test_size=(1-settings.valid_fract))

#     train_sps_j, valid_sps_j = train_sps[train_indices_j], train_sps[valid_indices_j]
#     train_ens_j, valid_ens_j = train_ens[train_indices_j], train_ens[valid_indices_j]

#     if settings.use_forces: 
#         train_dsp_dx_j, valid_dsp_dx_j = train_dsp_dx[train_indices_j], train_dsp_dx[valid_indices_j]
#         train_frcs_j, valid_frcs_j = train_frcs[train_indices_j], train_frcs[valid_indices_j]

#     # Convert to tensorflow constant tensors
#     train_sps_j = tf.constant(train_sps_j, dtype=settings.dtype)
#     train_ens_j = tf.constant(train_ens_j, dtype=settings.dtype)
#     valid_sps_j = tf.constant(valid_sps_j, dtype=settings.dtype)
#     valid_ens_j = tf.constant(valid_ens_j, dtype=settings.dtype)
#     if settings.sparse_gpflow:
#         sparse_train_sps = tf.Variable(sparse_train_sps, shape=sparse_train_spsettings.shape, dtype=settings.dtype, trainable=False)

#     if settings.use_forces:
#         train_dsp_dx_j = tf.constant(train_dsp_dx_j, dtype=settings.dtype)
#         train_frcs_j = tf.constant(train_frcs_j, dtype=settings.dtype)    
#         valid_dsp_dx_j = tf.constant(valid_dsp_dx_j, dtype=settings.dtype)
#         valid_frcs_j = tf.constant(valid_frcs_j, dtype=settings.dtype)        

#     test_sps = tf.constant(test_sps, dtype=settings.dtype)


#     # Batch data if  training set is larger than batch_size_max
#     if len(train_sps_j) < settings.batch_size_max:
#         iterations_per_epoch = 1
#         batch_size = len(train_sps_j)
#         if s.verbose:
#             print("Training using {} atoms without batching.".format(len(train_sps_j)))
#     else:
#         iterations_per_epoch = int(np.ceil(len(train_sps_j)/settings.batch_size_max))
#         batch_size = int(np.ceil(len(train_sps_j)/iterations_per_epoch))
#         if s.verbose:
#             print("Training using {} atoms total using {} batches with {} atoms per batch.".format( len(train_sps_j), iterations_per_epoch, batch_size ))

#     # training(out_data)

#     TimeBeforeEpoch0 = time.time()


#     mse_history = []    
#     hyperparam_history = []



#     # import warnings
#     # 
#     # from gpflow.models import VGP, GPR, SGPR, SVGP
#     # from gpflow.optimizers import NaturalGradient
#     # from gpflow.optimizers.natgrad import XiSqrtMeanVar
#     # from gpflow import set_trainable


#     if settings.my_priority == "efficiency":
#         # I don't know what this does
#         autotune = tf.data.experimental.AUTOTUNE


#         batches = (
#             tf.data.Dataset.from_tensor_slices((train_sps_j, train_ens_j))
#             .prefetch(autotune) 
#             .shuffle(buffer_size=len(train_sps_j), seed=settings.shuffle_seed)
#             .repeat(count=None)
#             .batch(batch_size)
#         )

#         batch_iterator = iter(batches)

#         # I also don't know why we use this
#         from gpflow.ci_utils import ci_niter, ci_range

#         optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learn_rate)

#     else:
#         batches = (
#             tf.data.Dataset.from_tensor_slices((train_sps_j, train_ens_j)) 
#             .shuffle(buffer_size=len(train_sps_j), seed=settings.shuffle_seed) 
#             .repeat(count=None)
#             .batch(batch_size)
#         )    

#         optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learn_rate)
#         #optimizer = tf.keras.optimizers.SGD(learning_rate=settings.learn_rate)



#     # new code to make tf.function training work
#     # --------------------------------------------
#     train_sps_j_i = tf.Variable(train_sps[:batch_size], shape=(batch_size, train_sps.shape[-1]), dtype=settings.dtype, trainable=False )
#     train_ens_j_i = tf.Variable(train_ens[:batch_size], shape=(batch_size, 1), dtype=settings.dtype, trainable=False ) 
#     if settings.sparse_gpflow:
#         if sparse_train_sps.shape[0] >= batch_size:
#             print("Warning: Batch size is not greater than sparse soap size.\nThis may cause errors in the predict_f function which assumes the inducing points to be fewer than the data points.")
#         if settings.my_priority == "efficiency":
#             gpr_model = gpflow.models.SVGP( kernel=kernel, likelihood=gpflow.likelihoods.Gaussian(),  inducing_variable=sparse_train_sps)
#             gpr_model.likelihood.variance.assign(obs_noise)
#             gpflow.set_trainable(gpr_model.q_mu, False)
#             gpflow.set_trainable(gpr_model.q_sqrt, False)
#         else:
#             gpr_model = gpflow.models.SGPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel, noise_variance=obs_noise, inducing_variable=sparse_train_sps)
#     else:
#         if settings.my_priority == "efficiency":
#             # it seems I cannot use  noise_variance=obs_noise for this which makes it not GAP...
#             gpr_model = gpflow.models.VGP( data=(train_sps_j_i, train_ens_j_i), kernel=kernel, likelihood=gpflow.likelihoods.Gaussian())
#         else:
#             gpr_model = gpflow.models.GPR( data=(train_sps_j_i, train_ens_j_i), kernel=kernel, noise_variance=obs_noise)
#     # --------------------------------------------


#     print_frequency = max(settings.min_print_frequency, int(settings.n_epochs/10))

#     if settings.my_priority == "efficiency":
#         hyperparam_history.append([(0, np.exp(var.numpy() )) for var in gpr_model.trainable_variables])  
#         gpr_objective = gpr_model.training_loss_closure(batch_iterator,  compile=True)
#         for j in range(ci_niter(settings.n_epochs)):
#             if not j % print_frequency:
#                 print("Epoch {}".format(j))
#             optimizer.minimize(gpr_objective, var_list=gpr_model.trainable_variables)
#             mse_history.append((j+1, gpr_model.elbo(data=(train_sps, train_ens))))
#             hyperparam_history.append([(j+1, np.exp(var.numpy()) ) for var in gpr_model.trainable_variables]) 
#         #optimizer.minimize(gpr_model.training_loss, gpr_model.trainable_variables, options=dict(maxiter=settings.n_epochs))

#     elif settings.my_priority == "consistency":

#         hyperparam_history.append([(0, var.numpy()) for var in gpr_model.trainable_parameters])  
#         for j in range(settings.n_epochs):
#             if not j % print_frequency:
#                 print("Epoch {}".format(j))
#                 #print(" ".join(["{} = {:.2e} ".format(var.name, np.exp(var.numpy())) for var in trainable_variables]))

#             mse_ens_j = 0
#             for i, (train_sps_j_i, train_ens_j_i) in enumerate(islice(batches, iterations_per_epoch)):
#                 if not settings.use_forces: #and not settings.sparse_gpflow :
#                     gpr_model.data[0].assign(train_sps_j_i)
#                     gpr_model.data[1].assign(train_ens_j_i)        
#                     mse_ens_j_i = train_hyperparams_without_forces_tf(gpr_model, valid_sps_j, valid_ens_j)
#                     print("valid_ens[:3] = {}".format( valid_ens_j[:3].numpy().flatten()) )
#     #                 print(mse_ens_j_i.numpy(), valid_ens_j[:3].numpy().flatten(), train_ens_j_i[:3].numpy().flatten()  )


#                 else:
#                     print("Using older approach (not converted to tf.function yet)")
#                     with tf.GradientTape() as tape:
#                         with tf.GradientTape(watch_accessed_variables=False) as tape_sps:
#                             tape_sps.watch(valid_sps_j)
#                             if settings.sparse_gpflow:
#                                 gpr_model = gpflow.models.SGPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel, inducing_variable=sparse_train_sps)
#         #                         gpflow.set_trainable(gpr_model.inducing_variable, False)
#                                 if i < 3:
#                                     print_summary(gpr_model)            
#                             else:
#                                 gpr_model.data[0].assign(train_sps_j_i)
#                                 gpr_model.data[1].assign(train_ens_j_i)
#                                 #gpr_model = gpflow.models.GPR(data=(train_sps_j_i, train_ens_j_i), kernel=kernel)
#                             #gpr_model.likelihood.variance.assign(obs_noise)                
#                             predict_ens_j_i = gpr_model.predict_f(valid_sps_j)[0]

#             #                 gpr_model = gpflow.models.GPR(data=(sps_j_i, train_ens_j_i), kernel=kernel_gpf)
#             #                 gpr_model.likelihood.variance.assign(obs_noise_gpf)
#             #                 predict_ens_j_i_gpf = gpr_model.predict_f(valid_sps_j)

#                         if settings.use_forces:
#                             predict_d_ens_j_i = tape_sps.gradient(predict_ens_j_i, valid_sps_j)
#                             # In the following line I needed to include '* n_atoms' after breaking energies into local energies
#                             # The reason is that I am effectively breaking the connection between E and F when doing that
#                             # F = -dE/dx =/= -dE_local/dx where E_local = E/n_atoms - E_free
#                             # When I split energies into local energies I initially calculated -dE_local/dx which is -dE/dx / n_atoms
#                             # This fix is prone to breaking the code and is not robust to systems with different structure size
#                             # Need to improve this with a better fix
#                             predict_frcs_j_i = -1*np.einsum('ijk,ik->ij', valid_dsp_dx_j, predict_d_ens_j_i) * n_atoms
#                             mse_j_i = mse_2factor_tf(predict_ens_j_i, valid_ens_j, 1/ens_var,
#                                                     predict_frcs_j_i, valid_frcs_j, 1/frcs_var)
#                             mse_ens_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
#                         else:
#                             mse_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
#                             mse_ens_j_i = mse_j_i


#             #         grads = tape.gradient(mse_j_i, trainable_variables)
#             #         optimizer.apply_gradients(zip(grads, trainable_variables))
#                     grads = tape.gradient(mse_j_i, gpr_model.trainable_variables)
#                     # print(gpr_model.trainable_variables[0])#grads[0])
#                     optimizer.apply_gradients(zip(grads, gpr_model.trainable_variables))
#                     if i < 3:
#                         print_summary(gpr_model)

#                     if not gpr_model.data[0][0,0].numpy() == train_sps_j_i[0,0].numpy() :
#                         print("ERRORERRORERRORERRORERRORERRORERROR")

#                 print("Adding mse_ens_j_i to mse_ens_j: {} + {} = {} ".format(mse_ens_j_i.numpy(), mse_ens_j , mse_ens_j_i.numpy() + mse_ens_j  ))
#                 mse_ens_j += mse_ens_j_i

#             mse_ens_j /= iterations_per_epoch
#             print("Epoch {},  mse = {}".format(j, mse_ens_j))
#             mse_history.append((j+1, mse_ens_j))
#             hyperparam_history.append([(j+1, var.numpy()) for var in gpr_model.trainable_parameters])    
#     else:
#         print("{} is not a reconized value for my_priority.\n Training did not occur.".format(settings.my_priority))
    
#     return gpr_model



# TimeBeforeWeights = time.time()
# train_hyperparams(train_sps, train_ens, sparse_train_sps, kernel=kernel, settings=s)


# In[19]:



TimeBeforeWeights = time.time()
print("Calculating weights")

if s.my_priority == "efficiency" and s.sparse_gpflow == True:
    gpr_model =gpr_model
elif s.my_priority == "consistency":
    if s.sparse_gpflow:
        gpr_model = gpflow.models.SGPR(data=(train_sps, train_ens), kernel=kernel, noise_variance = gpr_model.likelihood.variance, inducing_variable  = sparse_train_sps)
    else:
        gpr_model = gpflow.models.GPR( data=(train_sps, train_ens), kernel=kernel, noise_variance = gpr_model.likelihood.variance)      

print_summary(gpr_model)

if s.sparse_gpflow:
    if s.prediction_calculation in ("direct", "cholesky"):
        print("Alert: {} prediction approach not implemented for sparse model. Using alpha approach instead.".format(s.prediction_calculation))
        trained_weights = gpr_model.posterior().alpha
    elif s.prediction_calculation == "alpha":
        print("Attempting to calculate trained weights using alpha method for sparse gpr model.")
        trained_weights = gpr_model.posterior().alpha
        print("Successfully calculated trained weights using alpha method for sparse gpr model.")
else:
    if s.prediction_calculation in ("direct", "cholesky"):
        KNN = gpr_model.kernel(train_sps)
        KNN_diag = tf.linalg.diag_part(KNN)
        variance_diag = tf.fill(tf.shape(KNN_diag), gpr_model.likelihood.variance)
        KNN_plus_variance = tf.linalg.set_diag(KNN, KNN_diag + variance_diag)
        if s.prediction_calculation == "direct":
            KNN_inv =  tf.linalg.inv(KNN_plus_variance)
            trained_weights = tf.matmul(KNN_inv, train_ens)
        else:
            LNN = tf.linalg.cholesky(KNN_plus_variance)
            LNN_inv = tf.linalg.inv(LNN)
            KNN_inv_from_L = tf.matmul(LNN_inv, LNN_inv,transpose_a=True)
            trained_weights = tf.matmul(KNN_inv_from_L, train_ens)
    elif s.prediction_calculation == "alpha":
        print("ERROR: alpha not implemented for gpflow GPR. Skipping prediction")
    
TimeAfterTraining = time.time()


with tf.GradientTape(watch_accessed_variables=False) as tape_sps:
    tape_sps.watch(test_sps)  
    print("Predicting final energies")
    if s.prediction_calculation == "predict_f":
        predict_ens, predict_ens_var = gpr_model.predict_f(test_sps)
    else:
        if s.sparse_gpflow:
            predict_ens = tf.reshape( predict_energies_from_weights_tf(trained_weights, sparse_train_sps, test_sps, degree), [-1,1])
        else:
            predict_ens = tf.reshape( predict_energies_from_weights_tf(trained_weights,        train_sps, test_sps, degree), [-1,1])

test_ens_rescaled = ens_scaler.inverse_transform(test_ens)
predict_ens_rescaled = ens_scaler.inverse_transform(predict_ens)
if s.prediction_calculation == "predict_f":
    predict_ens_var_rescaled =  np.array(predict_ens_var * ens_scaler.scale_ **2)
    
if s.use_forces:
    print("Predicting final forces")    
    predict_d_ens = tape_sps.gradient(predict_ens, test_sps)
    predict_frcs = -1*np.einsum('ijk,ik->ij', test_dsp_dx, predict_d_ens) * n_atoms

    test_frcs_rescaled = test_frcs * ens_scaler.scale_
    predict_frcs_rescaled = predict_frcs * ens_scaler.scale_

TimeAfterPrediction = time.time()


# In[20]:


TrainingCellNonEpochsTraining = TimeBeforeEpoch0 - TimeBeforePreEpoch + TimeAfterTraining - TimeBeforeWeights 
if s.n_epochs:
    TimePerEpoch = (TimeBeforeWeights - TimeBeforeEpoch0)/s.n_epochs
else:
    TimePerEpoch = "N/A"
PredictionTime = TimeAfterPrediction - TimeAfterTraining

if s.print_timings:
    print("{:50s}: {:.3f}".format("Training time outside of epochs in training cell", TrainingCellNonEpochsTraining))
    if s.n_epochs:
        print("{:50s}: {:.3f}".format( "Training time per epoch", TimePerEpoch))
    print("{:50s}: {:.3f}".format("Prediction time", PredictionTime) )


# In[21]:


if in_notebook():
    if 'mse_history_by_n' not in locals():
        mse_history_by_n = {}
    if 'hyperparam_history_by_n' not in locals():
        hyperparam_history_by_n = {}

    hyperparam_history_by_n[s.n_structs] = hyperparam_history
    mse_history_by_n[s.n_structs] = mse_history

    print("Stored the hyperparameters and mse values for plotting under n={}".format(s.n_structs) )


# In[22]:


plot_hyperparam_training = in_notebook()

if plot_hyperparam_training:

    palette = plt.get_cmap('gist_ncar')#'nipy_spectral')#"viridis")
    palette_size = palette.N#len(palette.colors)
    palette_itr = 0

    fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(nrows=2, ncols = 2, figsize=(16,12))

    hyperparam_names = ["kernel offset", "kernel amplitude", "observation noise variance"]

    for n in np.sort(list(mse_history_by_n.keys())):
        mse_history = mse_history_by_n[n]
        hyperparam_history = hyperparam_history_by_n[n]
        if not len(hyperparam_history):
            continue
        color = palette(palette_itr)
        palette_itr = (palette_itr + 30) % palette_size



        # hyperparameters on axes 00, 01, 10
        hyperparams = np.swapaxes(hyperparam_history, 0, 1)

        ax00.plot(*zip(*hyperparams[0]), color=color, label="{}".format(n), lw=3)
        ax00.plot(*hyperparams[0][-1], "o", color=color)
        label00 = hyperparam_names[0]
        ax00.set_ylabel("{}".format(label00))
    #     #ax00.set_yscale('log')
    #     annotation00 = ax00.annotate('{:.1f}'.format(amplitudes[-1][1]) , xy=amplitudes[-1], xycoords='data', xytext=(-30,100),
    #                                  textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
        ax00.legend()
        ax00.ticklabel_format(useOffset=False)

        ax01.plot(*zip(*hyperparams[1]), color=color, label="{}".format(n), lw=3)
        ax01.plot(*hyperparams[1][-1], "o", color=color)
        label01 = hyperparam_names[1]
        ax01.set_ylabel("{}".format(label01))
    #     #ax01.set_yscale('log')
    #     annotation01 = ax01.annotate('{:.1f}'.format(lengths[-1][1]) , xy=lengths[-1], xycoords='data', xytext=(100,-30), 
    #                                  textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
        ax01.legend()
        ax01.ticklabel_format(useOffset=False)

        ax10.plot(*zip(*hyperparams[-1]), color=color, label="{}".format(n), lw=3)
        ax10.plot(*hyperparams[-1][-1], "o", color=color)
        label10 = hyperparam_names[-1]
        ax10.set_ylabel("{}".format(label10))
        #ax10.set_yscale('log')
    #     annotation01 = ax10.annotate('{:.1e}'.format(noises[-1][1]) , xy=noises[-1], xycoords='data', xytext=(-30,100),
    #                                  textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
    #     ax10.legend()
        ax10.ticklabel_format(useOffset=False)

        if not len(mse_history):
            continue

        #loss on axis 11
        ax11.plot(*zip(*mse_history), color = color, label="{}".format(n), lw=3)
        ax11.plot(*mse_history[-1], "o", color=color)
        #ax11.set_yscale('log')
        bottom, top = ax11.get_ylim()
        bottom2, top2 = ax11.get_ylim()
        ax11.set_ylabel("mse")
        ax11.legend()
        ax11.ticklabel_format(useOffset=False)

        #fig.suptitle("{}".format(s.n_structs))


# In[23]:


# make a regroup function
test_ens_regrouped = test_ens_rescaled.reshape(-1, len(StructureList[0]))
predict_ens_regrouped = predict_ens_rescaled.reshape(-1, len(StructureList[0]))
self_energies_regrouped = [[self_energy(atom.symbol, use_librascal_values=True) for atom in StructureList[i]] for i in test_indices]
test_global_ens = np.sum(test_ens_regrouped + self_energies_regrouped, axis=1)
predict_global_ens = np.sum(predict_ens_regrouped + self_energies_regrouped, axis=1)

if s.prediction_calculation == "predict_f":
    predict_ens_var_regrouped = predict_ens_var_rescaled.reshape(-1, len(StructureList[0]))
    predict_global_ens_var = np.sum(predict_ens_var_regrouped, axis=1)
    predict_global_ens_std = predict_global_ens_var ** 0.5 
    input_std = (gpr_model.likelihood.variance.numpy() * ens_scaler.scale_[0] **2) ** 0.5
    print("Our observation noise variance implies our reference error is +/- {:.3} /atom".format( input_std) )
else:
    predict_global_ens_std = None
plot_errors(model_description = "gpflow model",
            use_local=True,
            global_ens=test_global_ens,   predicted_global_ens= predict_global_ens,
            local_ens= test_ens_rescaled, predicted_local_ens = predict_ens_rescaled,
            color="mediumseagreen", predicted_stdev = None, n_atoms=n_atoms )

settings_string = ""
important_settings = ["nmax", "lmax", "rcut", "n_structs", "n_sparse", "n_epochs"]
for key, value in settings_dict.items():
    if key in important_settings:
        settings_string += "_" +str(key) + "_" + str(value)
import datetime as dt 
today = dt.datetime.today()
date_string = "_{:02d}_{:02d}_{:d}".format(today.month, today.day, today.year)
energy_results_title = "energy_results" + date_string + settings_string
plt.savefig(miniGAP_parent_directory + "results/" + energy_results_title)


# In[30]:


predict_global_ens.shape


# In[31]:


test_global_ens.shape


# In[32]:


np.corrcoef(test_ens_rescaled.flatten(), predict_ens_rescaled.flatten())[0,1]


# In[24]:


if s.use_forces:
    predict_frcs_rescaled

    fig, axs = plt.subplots(ncols=3, figsize=(20,5))
    components = ["x", "y", "z"]
    force_max = max(np.max(test_frcs_rescaled), np.max(predict_frcs_rescaled)) + np.std(test_frcs_rescaled)/2
    force_min = min(np.min(test_frcs_rescaled), np.min(predict_frcs_rescaled)) - np.std(test_frcs_rescaled)/2

    for i in range(3):

        axs[i].plot([force_min, force_max], [force_min, force_max], "-", c="k")
        axs[i].plot(test_frcs_rescaled[:,i], test_frcs_rescaled[:,i], "o", c="k", ms=4)
        axs[i].plot(test_frcs_rescaled[:,i], predict_frcs_rescaled[:,i], "o", label="custom", c="mediumseagreen", ms=5, alpha=.5)
        #axs[i].legend()
        axs[i].set_xlim(force_min, force_max); axs[i].set_ylim(force_min, force_max)

        try:
            m, b = np.polyfit(test_frcs_rescaled[:,i], predict_frcs_rescaled[:,i], 1)
            r2 = np.corrcoef(test_frcs_rescaled[:,i], predict_frcs_rescaled[:,i])[0,1]
            print("Least-squares regresion for F{}({}) produces the line line m {}+b with m = {:.5f} and b = {:.5f} which has r2 = {:.5f} ".format(components[i],components[i],components[i],m,b, r2))
        except:
            pass    

    #plt.savefig("../media/librascal_database_local_energy_force_learning")


# In[ ]:





