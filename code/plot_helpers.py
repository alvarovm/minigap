import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator
import numpy as np

# --------
import sys
sys.path.append('../code')
from general_helpers import flatten_to_list
from analysis_helpers import get_data_bounds, separate_outliers, get_exponent_range
# --------

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

def suggest_outlier_cropping_for_true_vs_predicted_plot(true, predicted, buffer = 0, center_mode="percentile", range_mode="percentile", 
                                               neighbor_factor=0.2, check_boundary_neighbors=True, range_factor=1.5, percentile_range=50, verbose=False):
    # Ignore large errors when picking our range
    # First determine which errors are outliers
    errors = (predicted - true).flatten()
    # I use np.unique so that a large multiplicity doesn't dominate when, for instance, 
    # the problem is not fully 3D so some errors are identically 0
    # I had to remove this because it caused an indices mismatch. Maybe I can use np.unique within separate_outliers?
    # errors = np.unique(errors)
    
    # Modified IQR method by default
    _, outliers, nonoutlier_indices, outlier_indices = separate_outliers(errors, center_mode=center_mode, range_mode=range_mode, 
                                                    neighbor_factor=neighbor_factor, check_boundary_neighbors=check_boundary_neighbors,
                                                    range_factor=range_factor, percentile_range=percentile_range, verbose=verbose, data_label="error" )
    nonoutlier_predictions = predicted.flatten()[nonoutlier_indices]

    # Buffer = 0 by default to make cropping more effective
    cropped_range = get_data_bounds(nonoutlier_predictions, buffer=0)
    # I don't think we ever want to crop out any of the true value range
    cropped_range = ( np.min( flatten_to_list( ( cropped_range, true ) ) ), np.max(flatten_to_list( ( cropped_range, true ) ) ) ) 
    return cropped_range

def plot_predicted_vs_true(predicted, true, ax, variable_label = "", units = "", system_label = "", title = "", r2=None, linear_fit_params=None, stderr=None,
                           x_range = None, y_range = None, color="mediumseagreen", ms=5):
    if not title:
        system_label = "\nfor " + system_label if system_label else ""
        title= "Predicted {} vs True {}{}".format(variable_label, variable_label, system_label)
    if units:
        units = " ({})".format(units)
    ax.set_xlabel("True {}{}".format(variable_label, units) )
    ax.set_ylabel("Predicted {}".format(variable_label, units) )
    ax.plot(sorted(true), sorted(true), "-", c="k", lw=1)
    ax.grid()
    ax.set_axisbelow(True)
    if stderr is not None:
        ax.errorbar(true, predicted, yerr=stderr, fmt="o", c=color, ms=ms, mec='k', label= "mean -/+ std")
        ax.legend()
    else:
        ax.plot(true, predicted, "o", c=color, ms=ms, mec='k')
    if x_range is None:
        x_range = get_data_bounds(true, buffer=0.05)
    if y_range is None:
        y_range = get_data_bounds([true, predicted], buffer=0.05)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.plot(x_range, x_range, ls=(0, (10, 20)), c="gray", lw=0.5, zorder=0)
    on_plot_text = []
    if linear_fit_params is not None:
        on_plot_text.append( "Linear Regression:\ny = {:.2f} x + {:.1f}".format(linear_fit_params[0], linear_fit_params[1]))
    if r2 is not None:
        on_plot_text.append( "r² = {:.3f}".format(r2) )
    ax.text(0.25, 0.75, "\n".join(on_plot_text), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(title)        

def plot_error_log_histogram(errors, ax, variable_label = "", units="", system_label = "", title = "", exponent_range = None,
                             color="mediumseagreen"):
    if not title:
        system_label = "\nfor " + system_label if system_label else ""
        title = "Error Histogram of {}{}".format( variable_label, system_label )
    if units:
        units = " ({})".format(units)        
    if exponent_range is not None:
        min_exponent, max_exponent = exponent_range
    else:
        min_exponent, max_exponent = get_exponent_range(errors, max_min=-5, min_max=-1)
    logbins = np.logspace(min_exponent, max_exponent, 4*(max_exponent - min_exponent))
    logticklabels = np.logspace(min_exponent, max_exponent, max_exponent - min_exponent + 1)
    ax.hist(errors, bins=logbins, color=color)
    ax.set_xscale('log')
    ax.set_xticks(logticklabels)
    ax.set_xticklabels(logticklabels)
    minor_tick_locator = LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator( minor_tick_locator )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("Error in {}{}".format( variable_label, units ) )
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    
def plot_error_histogram(errors, ax, variable_label = "", units="", system_label = "", title = "", data_range = None,
                             color="mediumseagreen"):
    if not title:
        system_label = "\nfor " + system_label if system_label else ""
        title = "Error Histogram of {}{}".format( variable_label, system_label )
    if units:
        units = " ({})".format(units)        
    if data_range is not None:
        bin_min, bin_max = data_range
    else:
        bin_min, bin_max = get_data_bounds(errors, buffer=0.05)
    nbins = 20
    bins = np.linspace(bin_min, bin_max, nbins)
    ax.hist(errors, bins=bins, color=color)
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("Error in {}{}".format( variable_label, units ) )
    ax.set_ylabel("Frequency")
    ax.set_title(title)