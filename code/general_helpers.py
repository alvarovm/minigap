import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import collections
import base64
import pandas as pd #I use this for printing a table. Place in try except eventually


# Adapted from stackoverflow.com/a/41477104

class Logger(object):
    def __init__(self, filename):
        # This makes it so the Logger.terminal attribute accesses the sys.stdout object that 
        #     already existed when this Logger was initiated
        self.terminal = sys.stdout
        # This saves the existant sys.stderr so we can reset it after finished using this Logger
        self.error = sys.stderr
        # This opens a file for logging
        self.logfile = open(filename, "a")
        # 'sys.stdout = self' gives sys.stdout the Logger methods
        # Therefore whenever I print anything, the script accesses the Logger.write() method
        sys.stdout = self
        # 'sys.stderr = self' gives sys.stderr the Logger methods
        sys.stderr = self

    def write(self, message, logfile_only=False):
        if not logfile_only:
            # This writes to the sys.stdout object that existed before this Logger was initiated.
            # That prints output on the terminal.
            # Unfortunately that means both stderr and stdout are compressed to a new stdout.
            # However, this seems less important than the ability to save both stdout and stderr to a log file.
            self.terminal.write(message)
        # This writes to the log file
        self.logfile.write(message)

    def flush(self):
        # Apparently this needs to exist. Does nothing.
        pass
    
    def stop(self):
        # This closes the log file
        sys.stdout.logfile.close()
        # This resets sys.stdout as its initial value. Importantly, that means its write() method
        #     returns to its default state. If you don't do this and you print after you run this 
        #     Logger.stop() method, you will call the Logger.write() method instead of the default
        #     write() method. This will try to write to the log file which is already closed and 
        #     therefore will cause an error
        sys.stdout = sys.stdout.terminal
        # Same as above, but for stderr
        sys.stderr = sys.stderr.error

def check_if_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
    
def PrintNoScientificNotation(*x):
    np.set_printoptions(suppress=True) # Do not print in scientific notation
    print(*x)
    np.set_printoptions(suppress=False)
    
       
def TickTock(func, *args, **kwargs):
    tick = time.time()
    func_output = func(*args, **kwargs)
    tock = time.time()
    return func_output, tock - tick

# Inspired by stackoverflow.com/a/6027615/
def flatten_dict(d_in, d_out='',  key_min_parent_level=0, key_max_parents=0, key_path = []):
    # I use the following two lines instead of 'd_out={}' in the function's parameter list because
    # you must use an immutable variable as a function default value. If the default value is mutable,
    # then when the function is run twice, the variable will start off with the second run with
    # the value it finished with on the first run.
    if d_out == '':
        d_out = {}
    for k, v in d_in.items():
        
        key_path_temp = key_path + [k]
        if isinstance(v, collections.MutableMapping):
            d_out = flatten_dict(v, d_out, key_min_parent_level, key_max_parents, key_path_temp)
        else:
            hierarchical_key_child = key_path_temp.pop(-1)
            parent_level = min(len(key_path_temp), max(key_min_parent_level, len(key_path_temp) - key_max_parents) )
            hierarchical_key_i = "_".join(key_path_temp[parent_level:] + [hierarchical_key_child] ) 
            if hierarchical_key_i in d_out.keys():
                duplicate_key_error_message = "Error while trying to flatten dictionary. Attempted to create more than one entry with key '{}'.".format(hierarchical_key_i)
                raise KeyError(duplicate_key_error_message)
            else:
                d_out[hierarchical_key_i] = v
    return d_out

def generate_unique_id():
    return base64.b64encode(os.urandom(64)).decode().replace("/", "").replace("+", "")[:5]

def make_unique_directory(ideal_directory_name, identifier_type="counter", verbose=False):
    # Remove possible trailing "/" because it would interfere with this code
    ideal_directory_name = ideal_directory_name if ideal_directory_name[-1] != "/" else ideal_directory_name[:-1]
    unique_directory_name, extension = os.path.splitext(ideal_directory_name)
    if len(extension):
        print("Error in requested directory name, '{}'. Extenstion '{}' detected. Try again without extension.".format(unique_directory_name, extension) )
        return 1
    try:
        os.mkdir(unique_directory_name)
        if verbose:
            print("Successfully created '{}' directory.".format(ideal_directory_name) )
    except FileExistsError:
        if identifier_type=="counter":
            counter = 2
            while os.path.isdir(unique_directory_name):
                unique_directory_name = ideal_directory_name + "_" +  str(counter)
                counter += 1
        elif identifier_type == "random_string":
            while os.path.isdir(unique_directory_name):
                unique_directory_name = ideal_directory_name + "_" +  generate_unique_id()
        else:
            print("Do not recognize identifier_type '{}'. \
            \nUse 'counter' for an integer index suffix or use 'random_string' for a random 5 character string.".format(identifier_type))
            return 2
        
        os.mkdir(unique_directory_name)
        if verbose:
            print("Could not create '{}', because it already existed. Created '{}' instead.".format(ideal_directory_name, unique_directory_name) )
    # For consistency, it is best to return a path ending with a "/"
    unique_directory_name = unique_directory_name if unique_directory_name[-1] == "/" else unique_directory_name + "/"
    return unique_directory_name 


def find_unique_filename(ideal_filename, identifier_type="counter", verbose=False):
    ideal_filename_base, extension = os.path.splitext(ideal_filename)
    if not len(extension) > 1:
        print("Error in requested filename, '{}'. No extenstion detected. Try again with extension.".format(ideal_filename) )
        return 1
    if not os.path.isfile(ideal_filename):
        if verbose:
            print("Filename '{}' selected.".format( ideal_filename ) )
        return ideal_filename
    else:
        unique_filename = ideal_filename
        if identifier_type=="counter":
            counter = 2
            while os.path.isfile(unique_filename):
                unique_filename = ideal_filename_base + "_" +  str(counter) + extension
                counter += 1
        elif identifier_type == "random_string":
            while os.path.isfile(unique_filename):
                unique_filename = ideal_filename_base + "_" + generate_unique_id() + extension
        else:
            print("Do not recognize identifier_type '{}'. \
            \nUse 'counter' for an integer index suffix or use 'random_string' for a random 5 character string suffix.".format(identifier_type))
            return 2

        if verbose:
            print("Filename '{}' selected because requested filename '{}' is already assigned".format(unique_filename, ideal_filename) )
        return unique_filename

# These plotting settings make graphs easier to read
# This is a very clunky way to do this and I want to do it more elegantly (suggestions welcome)

SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIG_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title


def plot_energy_errors(global_ens, predicted_global_ens, model_description = "model", 
                use_local=False, local_ens=[], predicted_local_ens=[],
                color="mediumseagreen", predicted_stdev=None, n_atoms=10, in_notebook=True):
   
    global_ens, predicted_global_ens = np.array(global_ens), np.array(predicted_global_ens)
    local_ens, predicted_local_ens = np.array(local_ens).flatten(), np.array(predicted_local_ens).flatten()
    global_r2 = np.corrcoef(global_ens, predicted_global_ens)[0,1]
    local_r2 = np.corrcoef(local_ens, predicted_local_ens)[0,1]
    
    if use_local:
        fig, axs = plt.subplots(figsize=(20,4.5), ncols=3)
    else:
        fig, axs = plt.subplots(figsize=(12, 5), ncols=2)
    
    
    axs[0].set_title("Predicted Global Energy vs True Global Energy\nfor {}".format(model_description))
    axs[0].set_xlabel("True Global Energy ")
    axs[0].set_ylabel("Predicted Global Energy ")
    axs[0].plot(global_ens, global_ens, "-", c="k", lw=0.5)
    if type(predicted_stdev) != type(None):
        axs[0].errorbar(global_ens, predicted_global_ens, yerr=predicted_stdev, fmt="o", c=color, ms=5, label= "mean -/+ std")
        axs[0].legend()
    else:
        axs[0].plot(global_ens, predicted_global_ens ,"o", c=color, ms=5)
    axs[0].text(0.25, 0.75, "r2 = {:.3f}".format(global_r2), horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    

    
    global_errors = abs(global_ens-predicted_global_ens)/n_atoms
    errors = abs(predicted_local_ens-local_ens) if use_local else global_errors
    
    # For generating tickmarks on axes
    max_err_exp = max(-1, int(np.ceil(np.log10(max(global_errors)))), int(np.ceil(np.log10(max(errors)))) )
    min_err_exp = min(-5, int(np.ceil(np.log10(min(global_errors)))), int(np.ceil(np.log10(min(errors)))) )

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(errors)
    max_abs_error = np.max(errors)
    error_dataframe = pd.DataFrame(data={"Local Energy":[rmse, mae, max_abs_error, local_r2]}, index = ["Root Mean Squared Error", "Mean Absolute Error", "Max Absolute Error", "r²"])
    #print("For LOCAL energies the rms error = {:.3e}, the mean absolute error = {:.3e} and the max absolute error = {:.3e}".format(rmse, mae, max_abs_error))

    global_rmse = np.sqrt(np.mean(global_errors ** 2))
    global_mae = np.mean(global_errors)
    global_max_abs_error = np.max(global_errors)
    error_dataframe["Global Energy"] = [global_rmse, global_mae, global_max_abs_error, global_r2]
    #print("For GLOBAL energies the rms error = {:.3e}, the mean absolute error = {:.3e} and the max absolute error = {:.3e}".format(global_rmse, global_mae, global_max_abs_error))
    if in_notebook:
        display(error_dataframe)
    else:
        print(error_dataframe)
    
    logbins = np.logspace(min_err_exp, max_err_exp, 4*(max_err_exp - min_err_exp))
    logticklabels = np.logspace(min_err_exp, max_err_exp, max_err_exp - min_err_exp + 1)
    axs[1].hist(global_errors, bins=logbins, color=color)
    axs[1].set_xscale('log')
    axs[1].set_xticks(logticklabels)
    axs[1].set_xticklabels(logticklabels)
    axs[1].set_xlabel("Error in Predicted Global Energy/Atom")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Error Histogram of Global Energy Predictions\nfor {}".format(model_description))
    
    if use_local:
        logbins = np.logspace(min_err_exp, max_err_exp, 4*(max_err_exp - min_err_exp))
        logticklabels = np.logspace(min_err_exp, max_err_exp, max_err_exp - min_err_exp + 1)
        axs[2].hist(errors, bins=logbins, color=color)
        axs[2].set_xscale('log')
        axs[2].set_xticks(logticklabels)
        axs[2].set_xticklabels(logticklabels)
        axs[2].set_xlabel("Error in Predicted Local Energy/Atom")
        axs[2].set_ylabel("Frequency")
        axs[2].set_title("Error Histogram of Local Energy Predictions\nfor {}".format(model_description))