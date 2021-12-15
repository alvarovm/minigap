import numpy as np
import warnings

# --------
import sys
sys.path.append('../code')
from general_helpers import *
# --------


def calculate_r2(x,y):
    return np.corrcoef(x, y)[0,1] ** 2

def least_squares_fit(x, y, verbose):
    warnings.filterwarnings('error')
    try:
        m, b = np.polyfit(x, y, 1)
    except Exception as fit_error:
        if verbose:
            print("Received the error '{}' when trying to produce a least squares fit.".format(fit_error))
        m, b = None, None
    warnings.filterwarnings('default')
    return m, b

def get_data_bounds(*data, buffer=0):
    data = flatten_to_list(data)
    return minus_plus(np.array([min(data), max(data)]), np.ptp(data)* buffer)

def minus_plus(a, b):
    if not isinstance(a, collections.abc.Iterable):
        return a-b, a+b
    else:
        return np.array(a-b)[0], np.array(a+b)[1]

def compile_error_stats(predicted, true, stats_to_calculate=(), verbose=False):
    errors = predicted - true
    abs_errors = np.abs(errors)
    dim = len(errors.shape)
    n_measurements = errors.shape[-1]
    
    all_stat_options = ("mean", "median", "var", "mae", "mxae", "rmse", "r2", "vae", "logvae", "linfit")
    unrecognized_stats = []
    for stat in stats_to_calculate:
        if stat not in all_stat_options:
            unrecognized_stats.append(stat)
    if len(unrecognized_stats):
        print( "Do not recognize statistic(s): {}. Ignoring this input. Next time please choose from {}.".format(unrecognized_stats, all_stat_options) )
    if stats_to_calculate == ():
        stats_to_calculate = all_stat_options
    
    error_statistics = {}
    for stat in stats_to_calculate:
        if stat == "mean":
            # mean
            error_statistics[stat] = np.mean(errors, axis=0)
        elif stat == "median":
            # median
            error_statistics[stat] = np.median(errors, axis=0)
        elif stat == "var":
            # variance
            error_statistics[stat] = np.var(errors, axis=0)
        elif stat == "mae":
            # mean absolute error
            error_statistics[stat] = np.mean(abs_errors, axis=0)
        elif stat == "mxae":
            # max absolute error
            error_statistics[stat] = np.max(abs_errors, axis=0)
        elif stat == "rmse":
            # root mean squared error
            error_statistics[stat] = np.sqrt(np.mean(abs_errors ** 2, axis=0))
        elif stat == "r2":
            # Coefficient of determination 
            if dim == 1:
                error_statistics[stat] = calculate_r2(predicted, true)
            elif dim == 2:
                r2s = []
                for i in range(n_measurements):
                    r2s.append( calculate_r2(predicted[:,i], true[:,i]) )
                error_statistics[stat] = r2s
            else:
                raise NotImplementedError("Coefficient of determination calculation for arrays of dimension 3+ is not implemented.")
        elif stat == "vae":
            # Variance of absolute errors
            error_statistics[stat] = np.var(abs_errors, axis=0)
        elif stat == "logvae":
            # Variance of logarithms of absolute errors
            error_statistics[stat] = np.var(np.log(abs_errors), axis=0)
        elif stat == "linfit" in stats_to_calculate:
            # y-intercept and slope of least squares regression line
            if dim == 1:
                m, b = least_squares_fit(true, predicted, verbose)
            elif dim == 2:
                m, b = [], []
                for i in range(n_measurements):
                    m_i, b_i = least_squares_fit(true[:,i], predicted[:,i], verbose)
                    m.append(m_i); b.append(b_i)
            else:
                raise NotImplementedError("Linear fit for arrays of dimension 3+ is not implemented.")
            error_statistics["m"] = m
            error_statistics["b"] = b

    if verbose:
        print("Calculated the following error statistics: {}".format(tuple(error_statistics.keys()) ) )
    return error_statistics, abs_errors


def get_exponent_range(data, max_min=-2, min_max=2):
    data = flatten_to_list(data)
    # We have to get rid of nonfinite numbers and 0's because they will cause a problem
    # when we try to convert the logs to ints
    # Numpy can manage most (all?) operations on non-finite numbers, but only finite
    # numbers can be cast to int
    data = np.array(data)[np.isfinite(data)]
    data = data[ np.nonzero(data)[0]]
    
    if len(data):
        min_exponent = min(max_min, int( np.ceil(np.log10(min(data)))) )
        max_exponent = max(min_max, int( np.ceil(np.log10(max(data)))) )   
        return (min_exponent, max_exponent)
    else:
        # If there is no data (possibly because it was filtered out as not finite )
        # then return default values
        return (max_min, min_max)

def compile_error_dataframe(error_info):
    import pandas as pd #I use this for printing a table. Place in try except eventually
    # error_info has the format {"Column Heading":{"stat keyword":stat value, ... }, ... }
    
    statistic_labels = {"mean":"Mean Error", "median":"Median Error", "var": "Variance of Error", "mae": "Mean Absolute Error", "mxae":"Max Absolute Error", 
                       "rmse":"Root Mean Squared Error", "r2": "r²", "vae": "Variance of Absolute Error", "logvae" : "Variance of Log Absolute Error",
                       "m": "Linear Fit y-Intercept", "b": "Linear Fit Slope"}
    row_labels = dict(Units="Units", **statistic_labels)
    
    first_column = True
    for column_heading, info_dict in error_info.items():
        if first_column:
            error_dataframe_rows = [row_labels[info] for info in info_dict.keys()]
            error_dataframe = pd.DataFrame(data={column_heading:info_dict.values()}, index = error_dataframe_rows)
            first_column = False
        else:
            error_dataframe[column_heading] = info_dict.values()
    return error_dataframe

def spherical_from_cartesian(v, angle_output="rad"):
    v=np.array(v)
    x, y, z = v.T
    r = np.linalg.norm(v, axis=len(v.shape)-1)
    theta = np.arccos(z/r)
    # This give nan when theta = 0 or pi. So far I think that is okay (2021/12/14)
    phi = np.arccos(x  / np.sqrt(x **2 + y **2) ) * np.sign(y)
    if angle_output=="rad":
        pass
    elif angle_output=="deg":
        theta *= 180/np.pi
        phi *= 180/np.pi
    else:
        raise TypeError("Do not recognize angle_output '{}'. Use 'deg' or 'rad'".format(angle_output))
    return np.array([r, theta, phi]).T


def separate_outliers(dataset, center_mode = "mean", range_mode="std_dev", percentile_range=50, range_factor=3, 
                      check_boundary_neighbors=True, neighbor_factor = 0.1, verbose = False, predefined_mode=None, data_label="data"):
    # You can think of this function as defining non-outlier "normal" data and what doesn't fit is an outlier
    # We define a range around the center of the data to be "normal"
    # In general this range has the form [A_L - f * B, A_U + f * B ]
    # We can choose A_L and A_U to be the values of lower and upper percentiles, such as 25% and 75% or we can set them both to the mean
    # B can be either a percentile range or the standard deviation
    # f is a factor you can choose to scale B by
    # 
    # This range can split apart data points that are very close in value if the range boundaries is between them
    # This is often undesirable so we can prevent this by iteratively including data that is close enough to the extrma of the "normal" data
    # Close enough is defined as c * B where c is chosen
    #
    # I think this currently only works on a 1D array, but it could be tweaked slightly to word for higher D
    
    # I have included the 2 most common definitions as predefined modes. Just supply the keyword
    if predefined_mode == "3sigma":
        center_mode = "mean"
        range_mode="std_dev"
        range_factor = 3
        check_boundary_neighbors = False
    elif predefined_mode == "IQR":
        center_mode = "percentile"
        range_mode="percentile"
        percentile_range=50
        range_factor = 1.5
        check_boundary_neighbors = False
    elif predefined_mode != None:
        error_message = "Do not recognize predefine_mode '{}'.".format(predefined_mode)
        error_message += "Choose either '3sigma' or 'IQR' or specify your own settings."
        raise ValueError(error_message)
        
    percentile_upper = 50 + percentile_range/2
    percentile_lower = 50 - percentile_range/2
    
    dataset = np.array(dataset)
    if len(dataset.shape) not in (1,2):
        if verbose:
            print( "separate_outliers does not currently support a {}-dimensional dataset".format(len(dataset.shape)) )
        dataset_invalid = True
    elif len(dataset) <= 2:
        if verbose:
            print( "separate_outliers cannot operate on a dataset of size {}".format(len(dataset) ) )
        dataset_invalid = True
    else:
        dataset_invalid = False
    if dataset_invalid:
        return dataset, [], range(len(dataset)), []

    if center_mode == "mean":
        center_upper = np.mean(dataset, axis = 0)
        center_lower = center_upper
    elif center_mode == "percentile":
        center_upper = np.percentile(dataset, percentile_upper, axis=0)
        center_lower = np.percentile(dataset, percentile_lower, axis=0)
    else:
        error_message = "Do not recognize center_mode = '{}'.".format(center_mode)
        error_message += " Choose either 'mean' or 'percentile'."
        raise ValueError(error_message)
        
    if range_mode == "std_dev":
        range_constant = np.std(dataset, axis=0)
    elif range_mode == "percentile":
        range_constant = np.percentile(dataset, percentile_upper, axis=0) - np.percentile(dataset, percentile_lower, axis=0)
    else:
        error_message = "Do not recognize range_mode = '{}'.".format(range_mode)
        error_message += " Please choose either 'std_dev' or 'percentile'"
        raise ValueError(error_message)
        
    # The following line allows this function to include the center of the data distribution as "normal data"
    # even when there is a very large multiplicity which causes the range_factor to be 0
    # e.g. when using the IQR method, range_factor would otherwise be 0 for data for which the middle half is all the same value
    range_constant = np.max( ( range_constant, np.finfo("float32").tiny) )
    
    # This finds the normal range if you don't consider the neighborhood
    # You have to use the else statement below if you do consider the neighborhood
    normal_range = {"lower":center_lower - range_factor * range_constant, "upper": center_upper + range_factor * range_constant}
    lower_outlier_indices = np.where( dataset < normal_range["lower"] )[0]
    upper_outlier_indices = np.where( dataset > normal_range["upper"] )[0]
    normal_indices   = np.where( (dataset > normal_range["lower"]) & (dataset < normal_range["upper"]) )[0]
    normal_data  = dataset[normal_indices]
    if not len(normal_data):
        print("No data is not an outlier.")
    
    if not check_boundary_neighbors:
        outlier_indices = np.sort( np.concatenate((lower_outlier_indices, upper_outlier_indices)), axis=0 )
        outlier_data = dataset[outlier_indices]
    else:
        neighborhood = range_constant * neighbor_factor
        lower_outliers = dataset[lower_outlier_indices]
        lower_outliers_sort_order = np.argsort(lower_outliers, axis=0)[::-1]
        lower_outliers_plus = np.concatenate((np.min(normal_data, axis = 0).reshape(1, *normal_data.shape[1:]), lower_outliers))
        lower_outliers_sorted_diffs = np.diff(np.sort(lower_outliers_plus, axis=0)[::-1], axis=0)
        lower_close_neighbors_indices = lower_outliers_sort_order[np.cumprod(-lower_outliers_sorted_diffs < neighborhood) == 1]
        formerly_lower_outliers_indices = lower_outlier_indices[lower_close_neighbors_indices]
        
        upper_outliers = dataset[upper_outlier_indices]
        upper_outliers_sort_order = np.argsort(upper_outliers, axis=0)
        upper_outliers_plus = np.concatenate((np.max(normal_data, axis = 0).reshape(1, *normal_data.shape[1:]), upper_outliers))
        upper_outliers_sorted_diffs = np.diff(np.sort(upper_outliers_plus, axis=0), axis=0)
        upper_close_neighbors_indices = upper_outliers_sort_order[np.cumprod(upper_outliers_sorted_diffs < neighborhood) == 1]
        formerly_upper_outliers_indices = upper_outlier_indices[upper_close_neighbors_indices]

        normal_indices = np.sort(np.concatenate((normal_indices, formerly_upper_outliers_indices, formerly_lower_outliers_indices) ), axis=0)
        all_indices = np.arange(len(dataset))
        outlier_indices = all_indices[~np.isin(all_indices, normal_indices)]
    
        outlier_data = dataset[outlier_indices]
        normal_data  = dataset[normal_indices]

    if verbose:
        if center_mode == "mean":
            center_upper_string = "mean"
            center_lower_string = "mean"
        else:
            center_upper_string = "{} percentile".format(ord(int(percentile_upper) ) )
            center_lower_string = "{} percentile".format(ord(int(percentile_lower) ) )
        
        if range_mode == "std_dev":
            range_constant_string = "std_dev"
        else:
            range_constant_string = "{}-{} percentile range".format(ord(int(percentile_lower) ), ord(int(percentile_upper) ) )
        outlier_settings_message = "Outliers of {} defined to exist outside the range ({} - {}*{}, {} + {}*{}), i.e. "\
                                    .format(data_label, center_lower_string, range_factor, range_constant_string, center_upper_string, \
                                    range_factor, range_constant_string)
        outlier_settings_message += "({:.2f}, {:.2f}).".format(normal_range["lower"], normal_range["upper"])
        if check_boundary_neighbors:
            outlier_settings_message += "\nAdditionally, {} was not considered an outlier if it fell outside that range, but was within".format(data_label)
            outlier_settings_message += " {} * {} ( = {:.2f}) of non-outlier {}".format(neighbor_factor, range_constant_string, neighborhood, data_label)
        if len(dataset.shape) == 1:
            outlier_settings_message += "\n{}/{} {} were found to be outliers.".format(len(outlier_data), len(dataset), data_label)
        print(outlier_settings_message)
    
    return normal_data, outlier_data, normal_indices, outlier_indices
