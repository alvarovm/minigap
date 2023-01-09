import gpflow
import numpy as np
from gpflow.kernels import SquaredExponential, Polynomial
#from ..logging import logger



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
        "verbose":True,
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
        return 
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
        return None
