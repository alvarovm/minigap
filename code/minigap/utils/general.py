import timeit
import numpy as np
import sys
import os
import collections
import base64

from os.path import join
#from ..logging import logger



# Fromstackoverflow.com/a/16671271
# Number to ordinal string
def ord(n):
    return str(n)+("th" if 4<=n%100<=20 else {1:"st",2:"nd",3:"rd"}.get(n%10, "th"))

def generate_unique_id():
    return base64.b64encode(os.urandom(64)).decode().replace("/", "").replace("+", "")[:5]

# def check_if_in_notebook():
#     try:
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter
    
def cast(in_variable, out_type=""):
    if out_type == "":
        out_type = type(in_variable)
    return out_type(in_variable)
    
def PrintNoScientificNotation(*x):
    np.set_printoptions(suppress=True) # Do not print in scientific notation
    print(*x)
    np.set_printoptions(suppress=False)
    
       
def TickTock(func, *args, **kwargs):
    """
    Wrapper Function Timer, TickTock
    Args:
        func :    function to time
        *args:    function arguments
        **kwargs: dicts
    """
    
    start = timeit.default_timer()
    
    func_output = func(*args, **kwargs)
    
    ticktock_time = timeit.default_timer() - start
    

    
    return func_output, ticktock_time

# From stackoverflow.com/a/2158532
# Useful if you have iterables of different shapes and types (otherwise you can use np.concatenate)
def flatten_to_generator( nested_list ):
    for element in nested_list:
        if isinstance(element, collections.abc.Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten_to_generator(element)
        else:
            yield element
    return

# Useful if you have iterables of different shapes and types (otherwise you can use np.concatenate)
def flatten_to_list( *nested_list ):
    return (list(flatten_to_generator(nested_list)))

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


def make_unique_directory(ideal_dir_name, identifier_type="counter", verbose=False):
    # Remove possible trailing "/" because it would interfere with this code
    ideal_dir_name = ideal_dir_name if ideal_dir_name[-1] != "/" else ideal_dir_name[:-1]
    unique_directory_name, extension = os.path.splitext(ideal_dir_name)
    if len(extension):
        print("Error in requested directory name, '{}'. Extenstion '{}' detected. Try again without extension.".format(unique_directory_name, extension) )
        return 1
    try:
        os.mkdir(unique_directory_name)
        if verbose:
            print("Successfully created '{}' directory.".format(ideal_dir_name) )
    except FileExistsError: end=""
    
    if identifier_type=="counter":
        counter = 2
        while os.path.isdir(unique_directory_name):
            unique_directory_name = ideal_dir_name + "_" +  str(counter)
            counter += 1
    elif identifier_type == "random_string":
        while os.path.isdir(unique_directory_name):
            unique_directory_name = ideal_dir_name + "_" +  generate_unique_id()
    else:
        print("Do not recognize identifier_type '{}'. \
        \nUse 'counter' for an integer index suffix or use 'random_string' for a random 5 character string.".format(identifier_type))
        return 2

    os.mkdir(unique_directory_name)
    
    if verbose:
        print("Could not create '{}', because it already existed. Created '{}' instead.".format(ideal_dir_name, unique_directory_name) )
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


    
def make_results_subdirectory(settings, parent_results_directory="../results"):
    title = settings.title
    subdirectory_name = title if title != None else "results"
    if settings.append_date_to_title:
        subdirectory_name += settings.date
    ideal_dir_name = join(parent_results_directory , subdirectory_name)
    unique_dir_name = make_unique_directory(ideal_dir_name, identifier_type='counter', verbose=settings.verbose)
    return unique_dir_name



