import numpy as np
import tensorflow as tf
import gpflow
from gpflow.kernels import Polynomial
import matplotlib.pyplot as plt

gpflow.config.set_default_float(np.float32)
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

print("{} GPU(s) recognized by tensorflow:".format(len(tf.config.list_physical_devices('GPU'))), tf.config.list_physical_devices('GPU'))

n=20000        
xs = np.random.rand(n).astype(np.float32)
def f(x):
    y =x ** 2 + np.random.rand(1)/100
    return y
ys = f(xs).astype(np.float32)

train_fract = .9
n_train = int(len(xs)*train_fract)
xs_train = tf.constant(xs[:n_train].reshape(-1,1), dtype=np.float32)
ys_train = tf.constant(ys[:n_train].reshape(-1,1), dtype=np.float32)
xs_test = tf.constant(xs[n_train:].reshape(-1,1), dtype=np.float32)


noise_init = 1e-4 
obs_noise = tf.Variable(noise_init, dtype=np.float32, name="noise")

slope_init = 3.0 
slope = tf.constant(slope_init, dtype=np.float32, name="kernel_amplitude")
offset_init = .01
offset = tf.constant(offset_init, dtype=np.float32, name="offset")
degree_init = 2.0
degree = tf.constant(degree_init, dtype=np.float32, name="degree")
kernel = Polynomial(variance=slope, offset=offset, degree=degree) 

gpr_model = gpflow.models.GPR( data=(xs_train, ys_train), kernel=kernel, noise_variance = obs_noise)
print("Begin predictions now.")
ys_predict = gpr_model.predict_f(xs_test)[0]

print("Predictions completed.")

ys_test = tf.constant(ys[n_train:], dtype=np.float32)
plt.plot(ys_test, ys_predict.numpy().flatten())