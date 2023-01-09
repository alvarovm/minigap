# Compiles some functions as TensorFlow tf functions not all of which are currently used
# Compiled tf functions are several times faster than normal functions
mse_tf = tf.function(mse, autograph=False, jit_compile=False)
mse_2factor_tf = tf.function(mse_2factor, autograph=False, jit_compile=False)
train_hyperparams_without_forces_tf = tf.function(train_hyperparams_without_forces, autograph=False, jit_compile=False)
predict_energies_from_weights_tf = tf.function(predict_energies_from_weights, autograph=False, jit_compile=False)