import gpflow
import numpy as np

from gpflow.ci_utils import ci_niter, ci_range

from gpflow.utilities import print_summary

import tensorflow as tf

from itertools import islice

from .data import convert_energy, convert_force
#from ..logging import logger

class sgpr(object):
    
    def __init__(self, kernel, mgtrain, **kwargs ):
        """ GPR constructor. MODE: consitency, no sparse, predict_f
        Args:
        """
        self.train = mgtrain
        self.settings = mgtrain.settings
       

        
        self.mse_history = []    
        self.hyperparam_history = []
        
        self.kernel = kernel
        
        self.init()
        return


    def init(self):
        """ Initialize kernels and model hyperparameters
        Args:
        """
        #print(self.train.sps_full)
        #print(self.settings)
        tf.random.set_seed(self.settings.tf_seed)
        
        noise_init = 1e-4 #.001# 0.0005499093576274776 #1.625e-4
        self.obs_noise = tf.Variable(noise_init, dtype=self.settings.dtype, name="noise")

        
        # Batch data if  training set is larger than batch_size_max
        if len(self.train.sps_full) < self.settings.batch_size_max:
            self.iterations_per_epoch = 1
            self.batch_size = len(self.train.sps_full)
            if self.settings.verbose:
                print("Training using {} atoms without batching.".format(len(self.train.sps_full)))
        else:
            self.iterations_per_epoch = int(np.ceil(len(self.train.sps_full)/self.settings.batch_size_max))
            self.batch_size = int(np.ceil(len(self.train.sps_full)/self.iterations_per_epoch))
            if self.settings.verbose:
                print("Training using {} atoms total using {} batches with {} atoms per batch.".format( len(self.train.sps_full), self.iterations_per_epoch, self.batch_size ))

        
        if self.settings.use_forces:
            train_dsp_dx_j = tf.constant(self.train.dsp_dx, dtype=self.settings.dtype)
            train_frcs_j = tf.constant(self.train.frcs, dtype=self.settings.dtype)    
            
        # new code to make tf.function training work
        # --------------------------------------------
        self.train_sps_j_i = tf.Variable(self.train.sps_full[:self.batch_size], shape=(self.batch_size, self.train.sps_full.shape[-1]), dtype=self.settings.dtype, trainable=False )
        self.train_ens_j_i = tf.Variable(self.train.ens[:self.batch_size], shape=(self.batch_size, 1), dtype=self.settings.dtype, trainable=False ) 
        
        self.batches = (
            tf.data.Dataset.from_tensor_slices((self.train.sps_full, self.train.ens)) 
            .shuffle(buffer_size=len(self.train.sps_full), seed=self.settings.shuffle_seed) 
            .repeat(count=None)
            .batch(self.batch_size)
            )
        
        if self.train.sps_sparse.shape[0] >= self.batch_size:
            print("Warning: Batch size is not greater than sparse soap size.\nThis may cause errors in the predict_f function which assumes the inducing points to be fewer than the data points.")
        
        self.sparse_train_sps = tf.Variable(self.train.sps_sparse, shape=self.train.sps_sparse.shape, dtype=self.settings.dtype, trainable=False)
        
        return
                
    def model(self):
        self.gpr_model = gpflow.models.SGPR(data=(self.train_sps_j_i, self.train_ens_j_i), 
                                            kernel=self.kernel, 
                                            noise_variance=self.obs_noise, 
                                            inducing_variable=self.train.sps_sparse)
        
        return
        
    def optimize(self, mgvalid=None):
                #self.valid = None
        if not mgvalid is None:
            print('here0')
            self.valid = mgvalid
        
# #if s.my_priority == "efficiency": NO SPARSE ONLY EFFICIENCY
#         hyperparam_history = []
    # I don't know what this does
#         prefetch_size = tf.data.experimental.AUTOTUNE

#         batches = (
#             tf.data.Dataset.from_tensor_slices((self.train.sps_full, self.train.ens))
#             .prefetch(prefetch_size) 
#             .shuffle(buffer_size=len(self.train.sps_full), seed=self.settings.shuffle_seed)
#             .repeat(count=None)
#             .batch(self.batch_size)
#         )

#         batch_iterator = iter(batches)

#         # I also don't know why we use this

#         optimizer = tf.keras.optimizers.Adam(learning_rate=self.settings.learn_rate)
        hyperparam_history = []        


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.settings.learn_rate)    

        ###
        print ('eerror check again')
        self.gpr_model = gpflow.models.GPR( data=(self.train_sps_j_i, self.train_ens_j_i), kernel=self.kernel, noise_variance=self.obs_noise)
        
######################
#
        print('here',self.valid,self.settings.n_epochs  )
        
        if self.settings.n_epochs == 0 or self.valid is None:
            return
        
        print_frequency = max(self.settings.min_print_frequency, int(self.settings.n_epochs/10))
        hyperparam_history.append([(0, var.numpy()) for var in self.gpr_model.trainable_parameters])
        
        # FOR CONSISTENCY
        def mse(y_predict, y_true):
            return tf.math.reduce_mean(tf.math.squared_difference(y_predict, y_true))
        
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
        def mse_2factor(y1_predict, y1_true, weight1, y2_predict, y2_true, weight2):
            mse1 = mse(y1_predict, y1_true)
            mse2 = mse(y2_predict, y2_true)*3

            return mse1 * weight1 + mse2 * weight2
        
        mse_2factor_tf = tf.function(mse_2factor, autograph=False, jit_compile=False)
        mse_tf = tf.function(mse, autograph=False, jit_compile=False)


        train_hyperparams_without_forces_tf = tf.function(train_hyperparams_without_forces, autograph=False, jit_compile=False)

#         train_hyperparams_without_forces_tf = tf.function(train_hyperparams_without_forces, autograph=False, jit_compile=False)

        
        valid_sps_j = tf.constant(self.valid.sps_full, dtype=self.settings.dtype)
        valid_ens_j = tf.constant(self.valid.ens, dtype=self.settings.dtype)
        
        train_nats_j = self.train.nats
        valid_nats_j = self.valid.nats
        
        if self.settings.use_forces:
            valid_dsp_dx_j = tf.constant(self.valid.dsp_dx, dtype=self.settings.dtype)
            valid_frcs_j   = tf.constant(self.valid.frcs,   dtype=self.settings.dtype)
        
        for j in range(self.settings.n_epochs):
            #print("Epoch {}".format(j))
            if not j % print_frequency:
                print("Epoch {}".format(j))
            mse_ens_j = 0
            
            for i, (self.train_sps_j_i, self.train_ens_j_i) in enumerate(islice(self.batches, self.iterations_per_epoch)):
                
                if not self.settings.use_forces: #and not s.sparse_gpflow :
                    print("Epoch {}".format(j))
                    self.gpr_model.data[0].assign(self.train_sps_j_i)
                    self.gpr_model.data[1].assign(self.train_ens_j_i)        
                    mse_ens_j_i = train_hyperparams_without_forces_tf(self.gpr_model, 
                                                                      valid_sps_j, valid_ens_j, self.optimizer)
#                     print("valid_ens[:3] = {}".format( valid_ens_j[:3].flatten()) )
                    print("valid_ens[:3] = {}".format( valid_ens_j[:3] ))
                else:
                    print("Using older approach (not converted to tf.function yet)")
                    
                    with tf.GradientTape() as tape:
                        with tf.GradientTape(watch_accessed_variables=False) as tape_sps:
                            tape_sps.watch(valid_sps_j)
            
                            self.gpr_model = gpflow.models.SGPR(data=(self.train_sps_j_i, self.train_ens_j_i), 
                                            kernel=self.kernel,  
                                            inducing_variable=self.train.sps_sparse)
    #                         gpflow.set_trainable(gpr_model.inducing_variable, False)
                            if i < 1:
                                print_summary(gpr_model)  
                
              
                            predict_ens_j_i = self.gpr_model.predict_f(valid_sps_j)[0]

    
        
                        if self.settings.use_forces:
                            
                            
                            #THIS SHOULD BE within TAPE
                            #THIS IS ALWAYS TRUE use_forces TRUE
                            #
                            predict_d_ens_j_i = tape_sps.gradient(predict_ens_j_i, valid_sps_j)
                        # In the following line I needed to include '* n_atoms' after breaking energies into local energies
                        # The reason is that I am effectively breaking the connection between E and F when breaking energies into local energies
                        # F = -dE/dx =/= -dE_local/dx where E_local = E/n_atoms - E_free
                        # When I split energies into local energies I initially calculated -dE_local/dx which is -dE/dx / n_atoms
                        # We can't just rescale the validation_frcs beforehand, by dividing them by n_atoms, because this 
                        # rescales their weight in the mse by n_atoms which
                        # would lead to forces from smaller structures being overweighted in importances to mse
                       
                            #print(valid_dsp_dx_j.shape, predict_d_ens_j_i.shape, valid_nats_j.shape)
                            #print(valid_nats_j.shape)
                            
                            #predict_frcs_j_i = -1*np.einsum('ijk,ik->ij', valid_dsp_dx_j, predict_d_ens_j_i)  * valid_nats_j
                            predict_frcs_j_i = -1*np.einsum('ijk,ik->ij', valid_dsp_dx_j, predict_d_ens_j_i)  * valid_nats_j
                            mse_j_i = mse_2factor_tf(predict_ens_j_i, valid_ens_j, 1/self.valid.ens_var,
                                                predict_frcs_j_i, valid_frcs_j, 1/self.valid.frcs_var)
                            mse_ens_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
                        else:
                            #THIS IS ALWAYS TRUE use_forces TRUE WHAT IS the point of THIS???
                        
                            mse_j_i = mse_tf(predict_ens_j_i, valid_ens_j)
                            mse_ens_j_i = mse_j_i
                        grads = tape.gradient(mse_j_i, self.gpr_model.trainable_variables)
                    
                    self.optimizer.apply_gradients(zip(grads, self.gpr_model.trainable_variables))
                    if i < 1:
                        print_summary(self.gpr_model)

                    if not self.gpr_model.data[0][0,0].numpy() == self.train_sps_j_i[0,0].numpy() :
                        print("ERRORERRORERRORERRORERRORERRORERROR")
                print("Adding mse_ens_j_i to mse_ens_j: {} + {} = {} ".format(mse_ens_j_i.numpy(), 
                                                                              mse_ens_j , 
                                                                              mse_ens_j_i.numpy() + mse_ens_j  ))
                mse_ens_j += mse_ens_j_i
            mse_ens_j /= self.iterations_per_epoch
            print("Epoch {},  mse = {}".format(j, mse_ens_j))
            self.mse_history.append((j+1, mse_ens_j))
            hyperparam_history.append([(j+1, var.numpy()) for var in self.gpr_model.trainable_parameters])
        
        return
    
    def weight(self):
        """Calculating weights
        
        """

        self.gpr_model = gpflow.models.SGPR(data=(self.train_sps_j_i, self.train_ens_j_i), 
                                            kernel=self.kernel, 
                                            noise_variance=self.gpr_model.likelihood.variance, 
                                            inducing_variable=self.train.sps_sparse)
#         print_summary(self.gpr_model)
        
    
    
    
        if self.settings.prediction_calculation in ("direct", "cholesky"):
            print("Alert: {} prediction approach not implemented for sparse model. Using alpha approach instead.".format(s.prediction_calculation))
            self.trained_weights = self.gpr_model.posterior().alpha
        elif self.settings.prediction_calculation == "alpha":
            print("Attempting to calculate trained weights using alpha method for sparse gpr model.")
            self.trained_weights = self.gpr_model.posterior().alpha
            print("Successfully calculated trained weights using alpha method for sparse gpr model.")
        
        return

    def predict(self, testdata, rescale = True):
        predict_ens= []
        predict_ens_var = []
        predict_frcs = []
        
        
        
        if not testdata is None:
            test_sps = tf.constant(testdata.sps_full, dtype=self.settings.dtype)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape_sps:
                tape_sps.watch(test_sps)  
                print("Predicting final energies")
                if self.settings.prediction_calculation == "predict_f":
                    predict_ens, predict_ens_var = self.gpr_model.predict_f(test_sps)
                else:
                    
                    def predict_energies_from_weights(c, soaps_old, soaps_new, degree, amplitude):
                        k = amplitude * tf.math.pow( tf.tensordot(soaps_old, tf.transpose(soaps_new), axes=1), degree=2 )
                        return tf.linalg.matmul(c, k, transpose_a=True)
                    
                    predict_energies_from_weights_tf = tf.function(predict_energies_from_weights, 
                                                                       autograph=False, jit_compile=False)
                    
                    predict_ens = tf.reshape( predict_energies_from_weights_tf(self.trained_weights,
                                                                               train.sps_sparse, test_sps, degree=2), [-1,1])
                    
                if self.settings.use_forces and not testdata.dsp_dx is None:
                    print("Predicting final forces")    
                    predict_d_ens = tape_sps.gradient(predict_ens, test_sps)
                    predict_frcs = -1*np.einsum('ijk,ik->ij', testdata.dsp_dx, predict_d_ens) * testdata.nats
        else:
            print('Nothing to predict')
            return [], [] ,[]
        if rescale:
            ens, ens_var, frcs = self.rescale(predict_ens, predict_ens_var, predict_frcs)
        else:
            ens, ens_var, frcs = predict_ens, predict_ens_var, predict_frcs
        #predict_ens, predict_ens_var, predict_frcs = ens, ens_var, frcs 
        return ens, ens_var, frcs 


            
    def rescale(self, pred_ens, pred_ens_var = None, pred_frcs = None ):
        print("rescale")
        pred_ens_rescaled = self.train.ens_scaler.inverse_transform(pred_ens).flatten()
        pred_ens_rescaled = convert_energy(pred_ens_rescaled, "eV", self.settings.output_energy_units)
        
        pred_ens_var_rescaled = []
        if self.settings.prediction_calculation == "predict_f" and not pred_ens_var is None:
            # rescale
            pred_ens_var_rescaled =  np.array(pred_ens_var * self.train.ens_scaler.scale_ **2).flatten()
            # convert to final units if necessary
            # Apply the conversion twice to account because the energy variance has units of energy squared
            pred_ens_var_rescaled = convert_energy(pred_ens_var_rescaled, "eV", self.settings.output_energy_units)
            pred_ens_var_rescaled = convert_energy(pred_ens_var_rescaled, "eV", self.settings.output_energy_units)

        pred_frcs_rescaled = []
        if  self.settings.use_forces and not pred_frcs is None:
            # rescale
            pred_frcs_rescaled = pred_frcs * self.train.ens_scaler.scale_
            # convert to final units if necessary
            pred_frcs_rescaled = convert_force(pred_frcs_rescaled, "eV/ang", self.settings.output_force_units)


        return pred_ens_rescaled, pred_ens_var_rescaled, pred_frcs_rescaled