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

def get_dscribe_descriptors(atoms_list, species = ["N"], rcut = 3.2, nmax = 5, lmax = 5, is_global=True):
    atoms_list = ([atoms_list] if isinstance(atoms_list, Atoms) else atoms_list)
    if is_global:
        AveragedSoap   = SOAP(average='outer',  species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax)
        return [  AveragedSoap.create(atom) for atom in atoms_list]
    else:
        UnAveragedSoap   = SOAP(average='off',  species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax)
        return [  UnAveragedSoap.create(atom) for atom in atoms_list]

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
    def __init__(self,  kernel,index_points, observation_index_points, observations, observation_noise_variance):#  test_xs, xs, ys,observation_noise_variance):
        super(GaussianProcessRegressionModel_saveable, self).__init__(kernel, index_points, observation_index_points, observations, observation_noise_variance)# kernel, test_xs, xs, ys,observation_noise_variance)
    
    @tf.function()
    def mean_compiled(self):
        return self.mean()
        
        
class TensorflowGaussianProcessRegressionModel():
    def __init__(self, test_xs):
        self.test_xs = tf.constant(test_xs, dtype=np.float64)
        
    def fit(self, xs, ys):
        #self.xs =xs
        #self.ys = ys
        self.xs= tf.constant(xs, dtype=np.float64)
        self.ys = tf.constant(RegularizeData(ys), dtype=np.float64)
        #index_points = np.mean(test_sps, axis=1,dtype=np.float64)[..., np.newaxis]

        self.observation_noise_variance = 1e-10

        self.constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
        # Smooth kernel hyperparameters
        self.smooth_amplitude = tfp.util.TransformedVariable(
            initial_value=.1, bijector=self.constrain_positive, dtype=np.float64,
            name='smooth_amplitude')
        self.smooth_length_scale = tfp.util.TransformedVariable(
            initial_value=.1, bijector=self.constrain_positive, dtype=np.float64,
            name='smooth_length_scale')
        # Smooth kernel
        self.smooth_kernel = tfk.ExponentiatedQuadratic(
            amplitude=self.smooth_amplitude, 
            length_scale=self.smooth_length_scale, name = "MySmoothKernel")

        
        # Recently updated code for training hyperparameters
        # ---------------------------------------------
        self.trainable_variables = [var.variables[0] for var in [self.smooth_amplitude, self.smooth_length_scale]]

        # Define mini-batch data iterator
        self.batch_size = 30

        self.batched_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (self.xs, self.ys))
            .shuffle(buffer_size=len(self.xs))
            .repeat(count=None)
            .batch(self.batch_size)
        )
        

        # Use tf.function for more efficient function evaluation
        @tf.function(autograph=False, experimental_compile=False)
        def gp_loss_fn(index_points, observations):
            """Gaussian process negative-log-likelihood loss function."""
            gp = tfd.GaussianProcess(
                kernel=self.smooth_kernel,
                index_points=index_points,
                observation_noise_variance=self.observation_noise_variance
            )

            negative_log_likelihood = -gp.log_prob(observations)
            return negative_log_likelihood


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        self.batch_nlls = []  # Batch NLL for plotting
        self.full_ll = []  # Full data NLL for plotting
        self.nb_iterations = 100
        for i, (index_points_batch, observations_batch) in enumerate(islice(self.batched_dataset, self.nb_iterations)):
            # Run optimization for single batch
            with tf.GradientTape() as tape:
                loss = gp_loss_fn(index_points_batch, observations_batch)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.batch_nlls.append((i, loss.numpy()))
            # Evaluate on all observations
            if i % 100 == 0:
                # Evaluate on all observed data
                ll = gp_loss_fn(
                    index_points=self.xs,
                    observations=self.ys)
                self.full_ll.append((i, ll.numpy()))

        # ---------------------------------------------

        
        self.regression_model = GaussianProcessRegressionModel_saveable(kernel=self.smooth_kernel,
            index_points=self.test_xs,
            observation_index_points=self.xs,
            observations=self.ys,
            observation_noise_variance=self.observation_noise_variance)

        return self
    
    def predict(self, redundant_test_xs):
        self.redundant_test_xs = tf.constant(redundant_test_xs, dtype=np.float64)
        return self.regression_model.mean()
    



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
