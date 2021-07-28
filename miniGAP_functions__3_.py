#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import matplotlib.pyplot as plt
import numpy.random as rand
import time
is_notebook = False

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential

# from ase.optimize import BFGS
from ase.optimize import MDMin

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import read

from dscribe.descriptors import SOAP

#from sklearn.metrics.pairwise import rbf_kernel

from numpy import polyfit
from numpy import poly1d

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
from tqdm.notebook import tqdm #smart iteration progress printer
from itertools import islice

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
#import tensorflow_probability as tfp
from  tensorflow_probability import distributions as tfd 
tfk = tfp.math.psd_kernels

from sklearn.metrics import mean_squared_error




def PrintNoScientificNotation(x):
    np.set_printoptions(suppress=True) # Do not print in scientific notation
    print(x)
    np.set_printoptions(suppress=False)


def TickTock(func, *args, **kwargs):
    tick = time.time()
    func_output = func(*args, **kwargs)
    tock = time.time()
    return func_output, tock - tick

def TimedExit(start_time):
    total_elapsed_time = time.time() - start_time
    print("\nThis script ran for a total of {:.1f} seconds.\n".format(total_elapsed_time))
    #exit()

    
def Distance(x1, x2):
    x1 = np.array(x1); x2 = np.array(x2)
    return np.sqrt(((x1 - x2) ** 2).sum())

def get_bond_lengths(atoms_list):
    return [Distance(*atoms.get_positions()) for atoms in atoms_list]



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
    molecule = Atoms('2'+ element, [(0., 0., 0.), (0., 0., bond_length)])
    
    if calc_type == "EMT":
        atom.calc = EMT()
        molecule.calc = EMT()
    elif calc_type == "LJ":
        atom.calc = LennardJones()
        molecule.calc = LennardJones()
    elif calc_type == "Morse":
        atom.calc = MorsePotential()
        molecule.calc = MorsePotential()
    else:
        print("This function does not recognize '{}' as a currently supported Atoms calculator type".format(calc_type))
        return
    
    #not sure if I need to add a cell
    #molecule.set_cell(np.array([[10,0,0],[0,10,0],[0,0,10]]))


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




def make_optimized_diatomic(element = "N", optimizer_type="MDMin", fmax=0.0001, verbose=False, 
                            bond_length=1.1, optimize_step=.02, calc_type="EMT", return_history = False):
    molecule = make_diatomic(element=element, bond_length=bond_length, verbose=verbose, calc_type=calc_type)
    
    traj_filename = element + "_" + optimizer_type + ".traj"
    
    if optimizer_type == "MDMin":
        optimizer = MDMin(molecule, trajectory=traj_filename, logfile=None, dt= optimize_step)
    elif optimizer_type == "BFGS":
        optimizer = BFGS(molecule, trajectory=traj_filename, logfile=None)
    else:
        print("This function does not currently support the optimizer type '{}'".format(optimizer_type))
        return
    
    optimizer.run(fmax=fmax)
    
    if return_history:
        return read(traj_filename,index=':')
    else:
        return molecule





def print_md_progress(molecule, i):
    epot = molecule.get_potential_energy() / len(molecule)
    ekin = molecule.get_kinetic_energy() / len(molecule)
    print('Step %2.0f: Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK) Etot = %.3feV' 
          % (i, epot, ekin, ekin / (2.5 * units.kB), epot + ekin))
    
def generate_md_traj(structure=None, from_diatomic=False, element = "N", nsteps=10, md_type="VelocityVerlet", time_step=1, bond_length=1.1,
                           temperature=300, verbose = False, print_step_size = 10, calc_type="EMT"):
    
    if structure is None and from_diatomic == False:
        print("Must provide a structure from which to start a trajectory or specify that you want to generate a trajectory of diatomic molecules.")
        return
    elif from_diatomic == True:
        molecule = make_optimized_diatomic(element=element, verbose=verbose, bond_length=bond_length, calc_type=calc_type)
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

    
    traj_filename = chemical_formula + "_" + md_type +".traj"
    
    MaxwellBoltzmannDistribution(molecule, temperature_K=temperature)# * (2.5 * units.kB))
    
    if md_type == "VelocityVerlet":
        md = VelocityVerlet(molecule, time_step * units.fs,
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
        
    atoms_traj_list = read(traj_filename,index=':')
    
    if verbose:
        plt.plot([atoms.get_kinetic_energy()/len(atoms)/(2.5*units.kB) for atoms in atoms_traj_list])
    
    return atoms_traj_list


# Get SOAP descriptor for each structure

def get_dscribe_descriptors(atoms_list, species = ["N"], rcut = 3.2, nmax = 5, lmax = 5, is_global=True, return_derivatives = False):
    atoms_list = ([atoms_list] if isinstance(atoms_list, Atoms) else atoms_list)
    averaging_keyword = "outer" if is_global else "off"
    positions = [atoms.get_positions() for atoms in atoms_list]
    soap = SOAP(average=averaging_keyword,  species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax)
    if return_derivatives:
        return soap.derivatives(atoms_list, positions = positions, method="auto")
    else:
        return soap.create(atoms_list)
# In[ ]:


# Explore gamma-space to see what gamma values produce nonzero off-diagonal elements
# gamma = 1/(2 * length_scale^2)
# length_scale = 1/sqrt(2 * gamma)




def RegularizeData(data):
    data = np.array(data)
    if len(data.shape) == 1:
        data_min = np.min(data)
        data_max = np.max(data)
    elif len(data.shape) == 2:
        data_min = np.min(data, axis=1).reshape(len(data), 1)
        data_max = np.max(data, axis=1).reshape(len(data), 1)
    else:
        print("This functionis not able to regularize data of the shape {}".format(data.shape))
    small_number = 10 ** -6 # 0 causes problems
    return (data - data_min)/(data_max - data_min) + small_number

def UnRegularizeData(data, old_data):
    data = np.array(data); old_ta = np.array(old_data)
    if len(data.shape) == 1 and len(old_data.shape) == 1:
        old_data_min = np.min(old_data)
        old_data_max = np.max(old_data)
    else:
        print("This functionis not able to regularize data of the shape {}".format(data.shape))
    small_number = 10 ** -6 # 0 causes problems
    return (data - small_number) * (old_data_max - old_data_min) + old_data_min





class PolynomialRegressionModel:
    def __init__(self, order):
        self.order = order       
        
    def fit(self, xs, ys):
        xs = np.array(xs)
        if len(xs.shape) == 2:
            self.xs = np.mean(xs, axis=1)
        elif len(xs.shape) == 1:
            self.xs = xs
        else:
            return
        self.regularized_xs = RegularizeData(self.xs)
        self.ys = ys
        self.polynomial_coefficients = polyfit(self.regularized_xs, self.ys, self.order)
        self.regression_model = poly1d(self.polynomial_coefficients)
        return self#.regression_model
    
    def predict(self, test_x):
        test_x = np.array(test_x)
        if len(test_x.shape) == 2:
            self.test_x = np.mean(test_x, axis=1)
        elif len(test_x.shape) == 1:
            self.test_x = test_x
        else:
            return
        self.test_x_regularized = RegularizeData(self.test_x)
        return self.regression_model(self.test_x_regularized)






class GaussianProcessRegressionModel_saveable(tfd.GaussianProcessRegressionModel):
    def __init__(self,  kernel,index_points, observation_index_points, observations, observation_noise_variance, jitter):#  test_xs, xs, ys,observation_noise_variance):
        super(GaussianProcessRegressionModel_saveable, self).__init__(kernel, index_points, observation_index_points,
                                                                      observations, observation_noise_variance, jitter=jitter)# kernel, test_xs, xs, ys,observation_noise_variance)
    
    @tf.function()
    def mean_compiled(self):
        return self.mean()
        
        
class TensorflowGaussianProcessRegressionModel():
    def __init__(self,  sps, dsps_dr, ens, frcs, test_sps, test_dsps_dr, test_ens, test_frcs, 
                 amplitude_init = .1, length_scale_init = .1, noise_init = 1e-5, jitter = 1e-6, verbose=False):
        self.verbose=verbose
        
        # Rescale sps
        print(sps.shape)
        self.sps_scaler = StandardScaler().fit(sps)
        self.sps, self.test_sps = self.sps_scaler.transform(sps), self.sps_scaler.transform(test_sps)
        self.dsps_dr = dsps_dr / self.sps_scaler.scale_[None, None, None, :]
        self.test_dsps_dr = test_dsps_dr / self.sps_scaler.scale_[None, None, None, :]

        # Also rescale energies and forces
        self.ens_scaler = StandardScaler().fit(ens[:,None])
        self.ens, self.test_ens = self.ens_scaler.transform(ens[:,None])[:,0], self.ens_scaler.transform(test_ens[:,None])[:,0]
        self.frcs = frcs / self.ens_scaler.scale_
        self.test_frcs = test_frcs / self.ens_scaler.scale_

        # Calculate variance for weighting mse_2factor
        self.ens_var = self.ens.var()
        self.frcs_var = self.frcs.var()
        
        self.sps_tf = tf.constant(self.sps, dtype=np.float64)
        self.dsps_dr_tf = tf.constant(self.dsps_dr, dtype=np.float64)
        self.ens_tf = tf.constant(self.ens, dtype=np.float64)
        self.frcs_tf = tf.constant(self.frcs, dtype=np.float64)        
        self.test_sps_tf = tf.constant(self.test_sps, dtype=np.float64)
        self.test_dsps_dr_tf = tf.constant(self.test_dsps_dr, dtype=np.float64)
        self.test_ens_tf = tf.constant(self.test_ens, dtype=np.float64)
        self.test_frcs_tf = tf.constant(self.test_frcs, dtype=np.float64)      
        
        self.constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
        # kernel hyperparameters
        self.observation_noise_variance = tfp.util.TransformedVariable(
            initial_value=noise_init, bijector=self.constrain_positive, dtype=np.float64, 
            name='observational_noise_variance')        
        self.amplitude = tfp.util.TransformedVariable(
            initial_value=amplitude_init, bijector=self.constrain_positive, dtype=np.float64,
            name='amplitude')
        self.length_scale = tfp.util.TransformedVariable(
            initial_value=length_scale_init, bijector=self.constrain_positive, dtype=np.float64,
            name='length_scale')
        self.kernel = tfk.ExponentiatedQuadratic( amplitude=self.amplitude, length_scale=self.length_scale, name = "kernel")

        
        self.trainable_variables = [var.variables[0] for var in [self.amplitude,
                                                                 self.length_scale, 
                                                                 self.observation_noise_variance]]
        
        self.jitter = tf.constant(jitter, dtype=np.float64)
        
        self.batch_error_history = []
        #self.full_loss_history = []
        self.hyperparam_history = []
        
    def fit(self, batch_size_max=30, batch_size_nll=3, 
            n_epochs=10, n_epochs_nll = 3, 
            learn_rate=0.05, shuffle_seed=1, valid_seed = 1, valid_fract=0.75):
        self.n_epochs = n_epochs
        
        if self.n_epochs:
            #self.optimizer = gpopts.Scipy()
            #self.optimizer = gpopts.NaturalGradient()
            #self.optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
            
            # Train with negative log likelihood for a few epochs
            # This reduces the chance of getting stuck in a bad local minimum of 
            # hsperparameters if hyperparameters have bad initial values
            # Only use energies here, no forces

            batches_nll = (tf.data.Dataset.from_tensor_slices((self.sps_tf, self.ens_tf))
                                    .shuffle(buffer_size=len(self.sps), seed=shuffle_seed)
                                    .repeat(count=None)
                                    .batch(batch_size_nll))
            iterations_per_epoch = int(len(self.ens)/ batch_size_nll)
            for j in range(n_epochs_nll):
                for i, (sps_i, ens_i) in enumerate(islice(batches_nll, iterations_per_epoch)):
                    with tf.GradientTape() as tape:
                        loss_i = self.gp_method_nll(sps_i, ens_i)
                        loss_i = tf.constant(loss_i, dtype=np.float64)
                    grads = tape.gradient(loss_i, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            
            # Train on mse of forces and energies
            # We predict forces with f = -dE/dsoap * dsoap/dr
            for j in range(self.n_epochs):
                split_data_j = train_test_split(self.sps, self.ens, self.dsps_dr, self.frcs, random_state =valid_seed +j, test_size=(1-valid_fract))
                sps_j, valid_sps_j, ens_j, valid_ens_j, dsps_dr_j, valid_dsps_dr_j, frcs_j, valid_frcs_j = split_data_j
                
                sps_j_tf = tf.constant(sps_j, dtype= np.float64)
                valid_sps_j_tf = tf.constant(valid_sps_j, dtype=np.float64)
                ens_j_tf = tf.constant(ens_j, dtype=np.float64)
                
                
                if len(sps_j) < batch_size_max:
                    iterations_per_epoch = 1
                    batch_size = len(sps_j)
                else:
                    iterations_per_epoch = int(np.ceil(len(sps_j)/batch_size_max))
                    batch_size = int(np.ceil(len(sps_j)/iterations_per_epoch))

                batches_j = (
                    tf.data.Dataset.from_tensor_slices((sps_j_tf, ens_j_tf, dsps_dr_j, frcs_j)) 
                    .shuffle(buffer_size=len(sps_j), seed=shuffle_seed) # Should I modify this to be seed = shuffle_seed + j or does it not matter?
                    .repeat(count=None)
                    .batch(batch_size)
                )
                
                # frcs_j_i might be extraneous unless we learn forces directly
                for i, (sps_j_i, ens_j_i, dsps_j_i, frcs_j_i) in enumerate(islice(batches_j, iterations_per_epoch)):
                    with tf.GradientTape() as tape:
                        with tf.GradientTape(watch_accessed_variables=False) as tape_dens_dsps:
                            tape_dens_dsps.watch(valid_sps_j_tf)    
                            gprm_j_i = tfd.GaussianProcessRegressionModel(
                                kernel = self.kernel,
                                index_points = valid_sps_j_tf,
                                observation_index_points = sps_j_i,
                                observations = ens_j_i,
                                observation_noise_variance = self.observation_noise_variance)
                            predict_ens_j_i = gprm_j_i.mean()
                        predict_dens_dsps_j_i = tape_dens_dsps.gradient(predict_ens_j_i, valid_sps_j_tf)
                        predict_frcs_j_i = -1*np.einsum('imjkl,il->ijk', valid_dsps_dr_j, predict_dens_dsps_j_i)#[:,1] # only get force of one atom while we are doing global soap
                        error_j_i = self.mse_2factor(predict_ens_j_i, valid_ens_j, 1/self.ens_var, predict_frcs_j_i, valid_frcs_j, 1/self.frcs_var)
                    grads = tape.gradient(error_j_i, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    self.batch_error_history.append((j*iterations_per_epoch + i, error_j_i))
                self.hyperparam_history.append([(j, self.constrain_positive(var).numpy()) for var in self.trainable_variables])
        else:
            # I could allow for collection of full mse error here (and also in the j for loop)
            # but this is low priority
            pass
        
        return self
    
    # Use tf.function for more efficient function evaluation
    # I do not know why we turn autograph off or what experimental_compile does
    @tf.function(autograph=False, experimental_compile=False) 
    def gp_method_nll(self, xs, ys):
        """Gaussian process negative-log-likelihood loss function."""
        if self.verbose:
            print("Now tracing TensorflowGaussianProcessRegressionModel.gp_method_nll()")
        gp = tfd.GaussianProcess(
            kernel=self.kernel,
            index_points=xs,
            observation_noise_variance=self.observation_noise_variance, 
            jitter=self.jitter
        )
        return -gp.log_prob(ys)
    
    @tf.function(autograph=False, experimental_compile=False)
    def mse(self, y_predict, y_true):
        return tf.math.reduce_mean(tf.math.squared_difference(y_predict, y_true))

    #@tf.function(autograph=False, experimental_compile=False)
    def mse_2factor(self, y1_predict, y1_true, weight1, y2_predict, y2_true, weight2):
        mse1 = self.mse(y1_predict, y1_true)
        mse2 = self.mse(y2_predict, y2_true)
        return mse1 * weight1 + mse2 * weight2
    
    
    def plot_learning(self, color="blue", fontsize=14, title="Learning"):
        matplotlib.rc('font', size=fontsize)
        fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(nrows=2, ncols = 2, figsize=(16,12))
        
        # hyperparameters on axes 00, 01, 10
        amplitudes, lengths, noises = np.swapaxes(self.hyperparam_history, 0, 1)
                                               
        ax00.plot(*zip(*amplitudes), color=color)
        ax00.plot(*amplitudes[-1], "o", color=color)
        ax00.set_ylabel("kernel ampliutude")
        #ax00.set_yscale('log')
        annotation00 = ax00.annotate('{:.1f}'.format(amplitudes[-1][1]) , xy=amplitudes[-1], xycoords='data', xytext=(-30,100),
                                     textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
                                               
        ax01.plot(*zip(*lengths), color=color)
        ax01.plot(*lengths[-1], "o", color=color)
        ax01.set_ylabel("kernel length scale")
        #ax01.set_yscale('log')
        annotation01 = ax01.annotate('{:.1f}'.format(lengths[-1][1]) , xy=lengths[-1], xycoords='data', xytext=(100,-30), 
                                     textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
                                               
        ax10.plot(*zip(*noises), color=color)
        ax10.plot(*noises[-1], "o", color=color)
        ax10.set_ylabel("kernel observational noise variance")
        ax10.set_yscale('log')
        annotation01 = ax10.annotate('{:.1e}'.format(noises[-1][1]) , xy=noises[-1], xycoords='data', xytext=(-30,100),
                                     textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'}, zorder=2)
                                               
        # loss on axis 11
        ax11.plot(*zip(*self.batch_loss_history), color=color)#, label="batch")
        ax11.plot(*self.batch_loss_history[-1], "o", color=color)
        ax11_twin = ax11.twinx()
        ax11_twin.plot(*zip(*self.full_loss_history), "--", color = "k", label="full")
        ax11_twin.plot(*self.full_loss_history[-1], "o", color="k")
        bottom, top = ax11.get_ylim()
        bottom2, top2 = ax11_twin.get_ylim()
        x, y = self.batch_loss_history[-1]
        y = bottom2 + (y- bottom)/(top - bottom)*(top2 - bottom2)
        annotation = ax11_twin.annotate('{:.0f}'.format(self.batch_loss_history[-1][1]) , xy=(x, y), xycoords='data', xytext=(100,30),
                                        textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'})
        annotation2 = ax11_twin.annotate('{:.0f}'.format(self.full_loss_history[-1][1]) , xy=self.full_loss_history[-1], xycoords='data', xytext=(30,-100),
                                         textcoords='offset points', bbox={'fc':"1"}, arrowprops={'fc':'k'})
        ax11.set_ylabel("batch negative log likelihood")
        ax11_twin.set_ylabel("full negative log likelihood", rotation=-90, labelpad=15)
        #ax11.legend()
        ax11_twin.legend()

    def predict(self, redundant_test_sps):        
        self.regression_model = tfd.GaussianProcessRegressionModel(
                                        kernel=self.kernel,
                                        index_points=self.test_sps_tf,
                                        observation_index_points=self.sps_tf,
                                        observations=self.ens_tf,
                                        observation_noise_variance=self.observation_noise_variance)
        self.predict_ens = self.regression_model.mean()
        return self.predict_ens


    def predict_ens_and_frcs(self, redundant_test_sps):        
        with tf.GradientTape(watch_accessed_variables=False) as tape_dens_dsps:
            tape_dens_dsps.watch(self.test_sps_tf)    

            self.regression_model = tfd.GaussianProcessRegressionModel(
                                        kernel=self.kernel,
                                        index_points=self.test_sps_tf,
                                        observation_index_points=self.sps_tf,
                                        observations=self.ens_tf,
                                        observation_noise_variance=self.observation_noise_variance)

            self.predict_ens = self.regression_model.mean()

        self.predict_dens_dsps = tape_dens_dsps.gradient(self.predict_ens, self.test_sps_tf)
        self.predict_frcs = -1*np.einsum('imjkl,il->ijk', self.test_dsps_dr_tf, self.predict_dens_dsps)#[:,1] # only get force of one atom while we are doing global soap

        return (self.predict_ens, self.predict_frcs)



# The following function, LearnEnergyFromSoap:
# 
#     - Takes soap descriptors and energies as input.
# 
#     - Splits this data into test and training sets
#     
#     - Trains a regression model on the training set
#          
#          -Can be sklearn gaussian processors or linear fit
#      
#      - Returns the model and test data
# 






def LearnEnergyFromSoap(soap_list, energy_list, training_fraction=0.7, verbose=False, 
                        kernel_type="RBF", model_type="GP_sklearn", split_seed = 1, gamma = 1, order = 0):
#     # For consistency we start with the same random seed everytime (used for partitioning step)
#     rand.seed(1)    
#     # Partition the soap and energy data into training and test groups
#     train_N = int(training_fraction * len(energy_list))
#     train_indices = rand.choice(range(len(energy_list)), size=train_N, replace=False)  
#     train_sps = [soap_list[i] for i in train_indices]
#     train_ens = [energy_list[i] for i in train_indices]
#     test_indices = [i for i in range(len(energy_list)) if i not in train_indices]
#     test_sps = [soap_list[i] for i in test_indices]
#     test_ens = [energy_list[i] for i in test_indices]

    energy_list = RegularizeData(energy_list)
    
    train_sps, test_sps, train_ens, test_ens = train_test_split(soap_list, energy_list, random_state=split_seed,
                                                                test_size=(1-training_fraction))

    if verbose:
        print("Initiating model training")
        
    if model_type == "GP_sklearn":
        if kernel_type == "RBF":
            rbf_kernel = RBF(length_scale=1/(2 * gamma )** 0.5, length_scale_bounds=(1e-4, 1e6))
            if verbose:
                median_element = np.median(rbf_kernel(train_sps).flatten())
                if median_element == 0 or median_element == 1:
                    print("Warning: This kernel may not be a good fit for this data")
                    print("Problem kernel[:4]:")
                    print(rbf_kernel(train_sps[:4]))
        else:
            print("This function does not currently support the kernel type '{}'".format(kernel_type))
            return
        regression_model = GaussianProcessRegressor(kernel=rbf_kernel).fit(train_sps, train_ens)
    elif model_type == "Polynomial":
        regression_model = PolynomialRegressionModel(order=order).fit(train_sps, train_ens)
    elif model_type == "GP_Tensorflow":
        regression_model = TensorflowGaussianProcessRegressionModel(test_xs=test_sps).fit(train_sps, train_ens)
    else:
        print("This function does not currently support the model type '{}'".format(model_type))
        return

    return regression_model, test_sps, test_ens, train_sps, train_ens




def LearnFromSoap(soap_list, dsoap_dr_list, energy_list, force_list,
                  training_fraction=0.7, valid_fraction = 0.7, verbose=False, model_type="GP_sklearn",
                        split_seed = 1, learn_seed = 1, valid_seed = 1, is_global = False, learn_forces = True,
                        batch_size_max=30, n_epochs=5, learn_rate=0.05,
                        noise_init = 1e-10, amplitude_init = 1, length_scale_init = 1, jitter=1e-6):
    if is_global:
        print("This program cannot compute forces with global soaps")
        return
    
    #energy_list = np.repeat(np.array(energy_list)/soap_list.shape[1], soap_list.shape[1])#, dtype=np.float64) 
    #print(soap_list.shape)
    #soap_list = np.reshape(soap_list, (soap_list.shape[0] * soap_list.shape[1], soap_list.shape[2]))
    #print(soap_list.shape)
    #soap_list = soap_list.reshape(len(soap_list),-1)
    soap_list = soap_list[:,0,:]


    #energy_list = RegularizeData(energy_list)
    
    print([x.shape for x in [soap_list, dsoap_dr_list, energy_list, force_list]])

    split_data = train_test_split(soap_list, dsoap_dr_list, energy_list, force_list, random_state=split_seed,
                                                                test_size=(1-training_fraction))
    train_sps, test_sps, train_dsps_dr, test_dsps_dr, train_ens, test_ens, train_frcs, test_frcs = split_data
    
    if verbose:
        print("Initiating model training")
    
    if model_type == "GP_Tensorflow":
        regression_model = TensorflowGaussianProcessRegressionModel(train_sps, train_dsps_dr, train_ens, train_frcs, test_sps,  test_dsps_dr, test_ens, test_frcs, 
                                                                    amplitude_init, length_scale_init, noise_init, jitter, verbose)\
                           .fit(batch_size_max=batch_size_max, n_epochs=n_epochs, learn_rate=learn_rate, shuffle_seed=learn_seed, valid_seed = valid_seed, valid_fract=valid_fraction)
    else:
        print("This function does not currently support the model type '{}'".format(model_type))
        return

    return regression_model, test_sps, test_dsps_dr, test_ens, test_frcs, train_sps, train_dsps_dr, train_ens, train_frcs





# This function:
#     - Plots predicted energies vs actual energies

def PlotPredictedEnergies(actual_energies, predicted_energies, all_energies = [], title="", show=True):
    if show:
        plt.cla()
    all_energies = [ *actual_energies, *predicted_energies, *all_energies]
    #min_energy = min(all_energies); max_energy = max(all_energies)
    #plt.plot([min_energy, max_energy], [min_energy, max_energy], "-k")
    plt.plot(actual_energies, predicted_energies, "o")
    plt.xlabel("Actual Test Energies")
    plt.ylabel("Energies Predicted by Regression Model")
    plt.title(title)
    if show:
        plt.show()
    



def GetErrorFromModel(model, test_xs, test_ys, error_types = "absolute"):
    predicted_ys = model.predict(test_xs)
    
    if type(error_types) == str:
        error_types = [error_types]
    
    errors = []
    for error_type in error_types:
        if error_type == "absolute":
            error = np.mean(np.absolute(test_ys - predicted_ys))
        elif error_type == "rms":
            error = mean_squared_error(test_ys, predicted_ys)#, squared=False)
        elif error_type == "r2":
            error = np.corrcoef(test_ys, predicted_ys)[0,1] **2
        else:
            print("This function does not currently support the error type '{}'".format(error_type))
            return
        errors.append(error)
        
    return errors
