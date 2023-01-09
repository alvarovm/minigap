import inspect
import logging
import os
import sys
import warnings

# import tensorflow as tf
import tensorflow as tf

from tqdm import tqdm



from ._settings import mgset as SETTINGS

# # disable tensorflow messages
# tf.logging.set_verbosity(tf.logging.WARN)
# # make a logger

# Configuring logger:
LOG_FORMAT = "%(message)s "
logging.basicConfig(stream=sys.stdout,
                     level=logging.DEBUG,
                     format=LOG_FORMAT)

#logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#logging.warning('This will get logged to a file')

def set_level(astr):
    """ Sets the level of what logging info is displayed when utilizing GPMol.

    Args:
        astr (string): Keyword (debug, info, warning, error, critical, notset) that determines what info is logged.
    """
    astr = astr.lower()
    str2level = {'debug': logging.DEBUG, 'info': logging.INFO,
                 'warning': logging.WARN, 'error': logging.ERROR,
                 'critical': logging.CRITICAL, 'notset': logging.NOTSET}

    logger.setLevel(str2level[astr])
    return


def set_logfile(filename):
    """ Sets the level of what logging info is displayed when utilizing GPMol.

    Args:
        astr (string): Keyword (debug, info, warning, error, critical, notset) that determines what info is logged.
    """
    f_handler = logging.FileHandler(filename,mode='w')
    f_format = logging.Formatter(LOG_FORMAT)
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    
    logger.debug('This will get logged to a file')

    return


# Create logger:


logger = logging.getLogger('minigap')
if 'log_level' in SETTINGS:
    set_level(SETTINGS['log_level'])

logger.debug('This will get logged ')
logging.getLogger('matplotlib.font_manager').disabled = True
#logger('matplotlib.font_manager').disabled = True

def make_banner():
    from ._version import __version__ as version
    import datetime as dt 
    from pathlib import Path
    from os.path import join
    banner = ""
    
    parent_directory = Path(__file__).parents[1]
    

    # Save date for use in naming output files
    today = dt.datetime.today()
    
    today_string = "_{:d}_{:02d}_{:02d}".format(today.year, today.month, today.day)
    today_string_alt = "{:d}/{:02d}/{:02d}".format(today.year, today.month, today.day)

    # Save date for use in naming output files
    version_placeholder = "_VERSION_PLACEHOLDER_"
    date_placeholder = "_DATE_PLACEHOLDER_"
    version_formatted = "{:{str_len}s}".format(version, str_len=len(version_placeholder) )
    date_formatted = "{:{str_len}s}".format(today_string_alt, str_len=len(date_placeholder) )
    
    banner_filename = join(parent_directory,'miniGAP_banner.txt')
    with open(banner_filename, "r") as f:
        banner = f.read()
    banner = banner.replace(version_placeholder, version_formatted).replace(date_placeholder, date_formatted)
    
    return banner

# def log_fncall(obj): return "{} {} called".format(
#     obj.__class__.__name__, inspect.stack()[1][3])

# def warn(astr):
#     """ Logs and raises a warning.

#     Args:
#         astr (string): Message that is logged and raised.
#     """
#     warnings.warn(astr)
#     logger.warn('Warning: {}'.format(astr))
#     return

#     # hide prints class

#     def __init__(self, stdout=None, stderr=None):
#         self.devnull = open(os.devnull, 'w')
#         self._stdout = stdout or self.devnull or sys.stdout
#         self._stderr = stderr or self.devnull or sys.stderr

#     def __enter__(self):
#         self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
#         self.old_stdout.flush()
#         self.old_stderr.flush()
#         sys.stdout, sys.stderr = self._stdout, self._stderr

#     def __exit__(self, exc_type, exc_value, traceback):
#         self._stdout.flush()
#         self._stderr.flush()
#         sys.stdout = self.old_stdout
#         sys.stderr = self.old_stderr
#         self.devnull.close()


# class HidePrints(object):
#     """A context manager for doing a "deep suppression" of stdout and stderr in
#     Python, i.e. will suppress all print, even if the print originates in a
#     compiled C/Fortran sub-function.

#         This will not suppress raised exceptions, since exceptions are printed

#     to stderr just before a script exits, and after the context manager has
#     exited (at least, I think that is why it lets exceptions through).
#     """

#     def __init__(self):
#         """ Open a pair of null files and save the actual stdout (1) and stderr (2) file descriptors.
#         """
#         self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
#         self.save_fds = [os.dup(1), os.dup(2)]

#     def __enter__(self):
#         """Assign the null pointers to stdout and stderr.
#         """
#         os.dup2(self.null_fds[0], 1)
#         os.dup2(self.null_fds[1], 2)

#     def __exit__(self, *_):
#         """Re-assign the real stdout/stderr back to (1) and (2)
#         """
#         os.dup2(self.save_fds[0], 1)
#         os.dup2(self.save_fds[1], 2)
#         # Close all file descriptors
#         for fd in self.null_fds + self.save_fds:
#             os.close(fd)


# if SETTINGS['in_jupyter']:
#     logger.debug('Running in Jupyter notebook')
# else:
#     logger.debug('Running in terminal')
