
mgset = {}

def injupyter_test():
    """ Test that determines whether in jupyter environment.

    Returns:
        bool

    """
    try:
        ipy_str = type(get_ipython()).__module__
        if ipy_str.startswith('ipykernel.'):
            return True
        else:
            return False
    except NameError:
        return False
        pass

def default_settings():
    import json
    from os.path import join, isfile
    from pathlib import Path
    from .utils.general import flatten_dict
    
    import datetime as dt 
    
    default_settings={}

    # Save date for use in naming output files
    today = dt.datetime.today()
    date = "_{:d}_{:02d}_{:02d}".format(today.year, today.month, today.day)
    
    default_settings['date'] = date
    
    miniGAP_parent_directory = Path(__file__).parents[0]

    
    

    filename = 'default_settings.json'
    filename = join(miniGAP_parent_directory,'./default_settings.json')
    
    if isfile(filename):
        with open(filename, encoding = 'utf-8') as f:
            default_settings.update(json.load(f))
    else:
        print('WARN: no default_settings found {}'.format(filename))
        
    default_settings['in_notebook'] = injupyter_test()
           
    return flatten_dict(default_settings)

def update_settings(settings, newdict):
    from .utils.general import flatten_dict
    
    settings.update(flatten_dict(newdict))
    #mgset = mgset | flatten_dict(set_dict)
    
    return settings

def settings_tuple(settings):
    from collections import namedtuple
    SettingsNamespace = namedtuple("Settings", mgset.keys())
    return SettingsNamespace(*settings.values())



mgset = default_settings()