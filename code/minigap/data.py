import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ase.units import Rydberg, kJ, Hartree, mol, kcal

from .descriptors import get_dscribe_descriptors


train_ids=[]
test_ids=[]

# -------- WHAT
no_forces_string = "Not Using Forces"

        
def get_ase_energy(struct, alt_keyword=""):
    if struct.calc != None:
        return struct.get_potential_energy()
    else:
        return struct.info[alt_keyword]

def convert_energy(energy, units_in, units_out):
    # relative to 1 eV
    # using colby.edu/chemistry/PChem/Hartree.html where ASE didn't have it
    conversion_factors = {
        "eV": 1.0,
        "Ha": 1 / Hartree,
        "J": 1000/kJ,
        "kJ/mol": mol/kJ,
        "kcal/mol": mol/kcal,
        "Ry": 1/Rydberg,
        "K": 11604.588577015096
    }
    #energy = np.array(energy)
    unrecognized_units = []
    if units_in != units_out:
        if units_in not in conversion_factors:
            unrecognized_units.append(units_in)
        if units_out not in conversion_factors:
            unrecognized_units.append(units_out)
        if len(unrecognized_units):
            units_error = "Do not recognize energy units: {}. Recognized energy units are: {}".format(unrecognized_units, list(conversion_factors.keys()))
            raise KeyError(units_error)
    
        energy = energy / conversion_factors[units_in] * conversion_factors[units_out]
    return energy

def convert_force(force, units_in, units_out):
    # relative to 1 eV/Å
    # using greif.geo.berkeley.edu/~driver/conversions.html as reference
    conversion_factors = {
        "eV/ang": 1.0,
        "Ry/Bohr": 1 / 25.71104309541616,
        "Ha/Bohr": 1 / 51.42208619083232,
        "N": 1e13/kJ
    }
    force = np.array(force)
    unrecognized_units = []
    if units_in != units_out:
        if units_in not in conversion_factors:
            unrecognized_units.append(units_in)
        if units_out not in conversion_factors:
            unrecognized_units.append(units_out)
        if len(unrecognized_units):
            units_error = "Do not recognize force units: {}. Recognized force units are: {}".format(unrecognized_units, list(conversion_factors.keys()))
            raise KeyError(units_error)
    
        force = force / conversion_factors[units_in] * conversion_factors[units_out]
    return force


def isolated_energy_parser(user_input):
    # returns dictionary from input format
    # e.g. "H:-0.4::Li:-7.5" --> {"H":-0.4, "Li":-7.5}
    return {atom_energy.split(":")[0]:float(atom_energy.split(":")[1]) for atom_energy in user_input.split("::")}


# returns isolated energies in eV
def get_isolated_energy(element, user_provided_isolated_energy_string, user_unit):
        # Calculated with B3LYP+D3 on CRYSTAL17
        default_isolated_energies = {
            "H":  -0.43796573720365,
            "Li": -7.4538249414114,
            "Be": -14.644518869964,
            "B": -24.512608074927,
            "C": -37.768216780161,
            "N": -54.572189501408,
            "O": -75.003393876879,
            "F": -99.672000678934,
            "Na": -162.20736395632,
            "Mg": -199.4784909209,
            "Al": -241.9640517253,
            "Si": -289.26042225551,
            "P": -341.20691893397,
            "S": -398.04415675952,
            "Cl": -460.04770109924,
            "K": -599.77830261799,
            "Ca": -677.27165785863,
            "Sc": -760.34151748052,
            "Ti": -848.8244251979,
            "V": -943.67762714996,
            "Cr": -1044.1147378511,
            "Mn": -1150.8928446101,
            "Co": -1382.3371803164,
            "Ni": -1508.0647901075,
            "Cu": -1640.1713647787,
            "Zn": -1779.2571044599,
            "Ga": -1924.5685189651,
            "Ge": -2076.7379664988,
            "As": -2235.7034565607,
            "Tc": -4205.5805934383
        }
        
        if user_provided_isolated_energy_string is not None:
            user_provided_isolated_energies = isolated_energy_parser(user_provided_isolated_energy_string)
            if element in user_provided_isolated_energies:
                isolated_energy = convert_energy(user_provided_isolated_energies[element], user_unit, "eV")
            elif element in default_isolated_energies:
                isolated_energy = default_isolated_energies[element]
            else:
                isolated_energy = 0
        else:
            if element in default_isolated_energies:
                isolated_energy = default_isolated_energies[element]
            else:
                isolated_energy = 0  
        return isolated_energy



    
class dataset(object):
    
    def __init__(self, StructureList, settings ):
        self.train_ids=[]
        self.test_ids=[]
        self.StructureList = StructureList
        self.settings = settings
        self.get_ids()
        return
    
    def get_ids(self):
        ids = [e for e in range(len(self.StructureList))]
        seed = 42
        train_fraction = 0.25
        if self.settings.split_seed != None:
            seed = self.settings.split_seed
        if self.settings.train_fraction != None:
            train_fraction = self.settings.train_fraction

        self.train_ids, self.test_ids = train_test_split(ids, train_size=train_fraction, 
                                                         random_state=seed)
        return
    
    def get_testid(self):
        return self.test_ids
    
    def get_trainid(self):
        return self.train_ids
    
    def getstruc_train(self):
        self.structrain = [self.StructureList[e] for e in self.get_trainid()]
        return self.structrain
    
    def getstruc_test(self):
        self.structest = [self.StructureList[e] for e in self.get_testid()]
        return self.structest
    
class mgtrain(object):
    
    def __init__(self, dataset, mode):
        self.ens = np.array([])
        self.n_atom_list = np.array([])
        self.en_shift_list = np.array([])
        self.frcs = np.array([])
        self.dsp_dx = np.array([])
        self.sps_full = np.array([])
        self.soap_scaler = 0
        self.ens_scaler = 0
        
        
 #       super().__init__(settings)
        
        self.settings = dataset.settings
        if mode == 'train':
            self.ids = dataset.get_trainid()
            self.StructureList = dataset.getstruc_train()

                
        elif mode == 'test':
            self.ids = dataset.get_testid()
            
            self.StructureList = dataset.getstruc_test()
            
            print(len(self.StructureList))
        else:
            print('error only train/mode')
            return
        print(len(self.StructureList))
        
        self.prepare_training()

#         if self.settings.sparse_gpflow:
#             self.SparsifySoaps()
        return
    
    def get_pos(self):
        """ Get atoms positions
        Args:
        """
        
        self.pos_list =  [list(atoms.positions) for atoms in self.StructureList]
        return 
    
    def get_atom_bool_list(self):
        """ Get atom and boolean lists
        Args:
        """
        self.n_atom_list = np.array([len(atoms) for atoms in self.StructureList])
        nats = self.n_atom_list
        n = len(self.ids)
        self.struct_bools = np.repeat(np.eye(n, dtype=np.float64), nats, axis=1)
        self.nats = np.repeat(nats, nats).reshape((-1, 1))

        return 
    
    def get_energies(self):
        """ Get global energies
        Args:
        """
        glob_energies = []
        alt_en_kw = self.settings.alt_energy_keyword
        for struct in self.StructureList:
            etotal = get_ase_energy(struct, alt_keyword = alt_en_kw)
            glob_energies.append(etotal)
        return np.array(glob_energies)
            
    def get_shift_list(self):
        self.en_shift_list = []
        for struct in self.StructureList:
            isolated_energies = np.array([ get_isolated_energy(atom.symbol, self.settings.isolated_energies, self.settings.isolated_energy_units) for atom in struct ])
            self.en_shift_list.append(sum(isolated_energies))
        self.en_shift_list = np.array(self.en_shift_list)
        return
        
    def get_local_energies(self):        
        """ Get local energies
        Args:
        """
        self.ens = []
        #self.en_shift_list = []
        alt_en_kw = self.settings.alt_energy_keyword
        for struct in self.StructureList:
            avetotal = get_ase_energy(struct, alt_keyword = alt_en_kw)/len(struct)
            en_list_i = np.array([avetotal for atom in struct])
            # Convert units if necessary
            # Convert to eV and eV/Å during calculation
            # Because F = -dE/dx, we must use the same x units for the known F values and the predicted F values
            # In particular, I choose eV and eV/Å because ASE positions are always assumed to be in angstroms
            # If you are using an input file which has structural information not in angstroms, forces learned by miniGAP will not be accurate

            en_list_i = convert_energy(en_list_i, self.settings.input_energy_units, "eV") 
            
            isolated_energies = np.array([ get_isolated_energy(atom.symbol, self.settings.isolated_energies, self.settings.isolated_energy_units) for atom in struct ])
            
            en_list_i -= isolated_energies
         
            
            self.ens.append(en_list_i)
 #           self.en_shift_list.append(sum(isolated_energies))

        
        self.ens = np.concatenate(self.ens).reshape(-1, 1)
 
        self.ens_scaler = StandardScaler().fit(self.ens)
        
        self.ens = self.ens_scaler.transform(self.ens)
        
        self.ens_var = self.ens.var()
            
        return 
    
##         self.en_shift_list = np.array(self.en_shift_list)
            
    def get_forces(self):        
        """ Get forces
        Args:
        """

        self.frcs = []
        for atoms in self.StructureList:
            struct = atoms
            frc_list_i = struct.get_forces()
                # See note in energy section about unit conversion
            frc_list_i = convert_force(frc_list_i, self.settings.input_force_units, "eV/ang")
            self.frcs.append(frc_list_i)

        self.frcs_var = []
        
        if self.settings.use_forces:
            self.frcs = np.concatenate(self.frcs, axis=0)
            self.frcs = self.frcs / self.ens_scaler.scale_                                                   
            self.frcs_var = self.frcs.var()

        return
    
    def get_soap_sp(self):
        """ Compute SOAP descriptors
        Args:
        """
        attach = self.settings.attach_SOAP_center
        is_periodic=self.settings.is_periodic
        return_derivatives= self.settings.use_forces
        get_local_descriptors = True
        
        
#         struct_list = []
#         for atoms in self.StructureList:
#             struct_list.append(atoms)
        
        self.dsp_dx=[]
        self.sps_full = []
        dscribe_output = get_dscribe_descriptors(
                                            self.StructureList, 
                                            attach=attach, 
                                            is_periodic = is_periodic,
                                            is_global= not get_local_descriptors,
                                            return_derivatives= return_derivatives, 
                                            nmax=self.settings.nmax, 
                                            lmax=self.settings.lmax,
                                            rcut=self.settings.rcut,
                                            smear=self.settings.smear)
        
        if return_derivatives:
            self.dsp_dx, self.sps_full = dscribe_output
            self.dsp_dx = np.moveaxis(self.dsp_dx.diagonal(axis1=1, axis2=2), -1, 1)
        else:
            self.sps_full = dscribe_output
            self.dsp_dx = [no_forces_string] * len(self.sps_full)
            
        self.sps_full = np.concatenate(self.sps_full)

        self.soap_scaler = StandardScaler().fit(self.sps_full)
        
    
        if self.settings.scale_soaps:
            self.sps_full =  self.soap_scaler.transform(self.sps_full)
            
        if self.settings.use_forces:
            self.dsp_dx = np.concatenate(self.dsp_dx).reshape(-1, 3, self.dsp_dx.shape[-1])
            
            if self.settings.scale_soaps:
                self.dsp_dx = self.dsp_dx/self.soap_scaler.scale_        
    
        return 
        
            
    def prepare_training(self):
        """ Prepare for training atoms and booleans lists, local energies , forces, and SOAP
        Args:
        """
        
#         #ens = []
#         self.ens, self.frcs, self.n_atom_list, self.en_shift_list = self.get_e_f_lists()

        self.get_atom_bool_list()
        self.get_shift_list()
        self.get_local_energies()
        
        if self.settings.use_forces:
            self.get_forces()
                
        self.get_soap_sp()
        
        print("shift list=",self.en_shift_list.shape)
        return  
    
    def prepare_testing(self):
        """ Prepare atoms and booleans lists, and SOAP
        Args:
        """
        self.get_atom_bool_list()
        self.get_shift_list()
        self.get_soap_sp()
        return
        

        
def sparsify(train, test=None, sparsify_samples=False, n_samples=0, sparsify_features=False, n_features=0, selection_method="CUR", **kwargs):
    
    
        
#         def SparsifySoaps(train_soaps, train_energies= [], test_soaps = [], sparsify_samples=False, n_samples=0, sparsify_features=False, n_features=0, selection_method="CUR", **kwargs):
    from skcosmo.sample_selection import CUR as CURSample
    from skcosmo.sample_selection import PCovCUR as PCovCURSample
    from skcosmo.feature_selection import CUR as CURFeature
    from skcosmo.feature_selection import PCovCUR as PCovCURFeature


    implemented_selection_methods = ["CUR", "PCovCUR"]


    if selection_method not in implemented_selection_methods:
        print("Do not recognize selecton method(s): {}".format(method))
        return

    if "PCovCUR" == selection_method and len(train.ens) == 0:
        print("You must provide training energies to use PCovCUR. Terminating")
        return

    if sparsify_features == True and len(test.sps_full) == 0 and n_features == 0 or test is None :
        print("You must provide test soaps to sparsify features. Terminating")
        return        

    default_kwargs = {
        "progress_bar":False,
        "score_threshold":1e-12,
        "full":False,
        "score_tolerance":1e-12,
        "iterative_selection":True,
        "mixing":0.5,
        "n_eigenvectors":1,
        "plot_importances":False
    }
    selection_settings = default_kwargs
    for kw, arg in kwargs.items():
        if kw in default_kwargs.keys():
            selection_settings[kw] = arg
        else:
            print("Do not recognize kwarg '{}'. Valid kwargs include {}.".format(kw, default_kwargs.keys()))

            selection_settings.update(kwargs)

    if sparsify_samples:
        if selection_method == "PCovCUR":
            sample_selector = PCovCURSample(n_to_select=n_samples, progress_bar=selection_settings["progress_bar"],
                                            score_threshold=selection_settings["score_threshold"],
                                            full = selection_settings["full"], k = selection_settings["n_eigenvectors"], 
                                            mixing = selection_settings["mixing"],
                                            iterative = selection_settings["iterative_selection"], 
                                            tolerance=selection_settings["score_tolerance"]
                                           )
            sample_selector.fit(train.sps_full, y=train.ens)
        elif selection_method == "CUR":
            sample_selector = CURSample(n_to_select=n_samples, progress_bar=selection_settings["progress_bar"],
                                        score_threshold=selection_settings["score_threshold"],
                                        full = selection_settings["full"], 
                                        k = selection_settings["n_eigenvectors"], 
                                        iterative = selection_settings["iterative_selection"], 
                                        tolerance=selection_settings["score_tolerance"]
                                        )
            sample_selector.fit(train.sps_full)
            print("CUR sample selection")

    else:
        sample_selector=lambda x: x
        sample_selector.transform = sample_selector.__call__ 

    ##        
    ## sparsify_features & sparsify_samples could have different methods?
    ##
    if sparsify_features:
        if selection_method == "PCovCUR":
            feature_selector = PCovCURFeature(n_to_select=n_features, 
                                              progress_bar=selection_settings["progress_bar"], 
                                              score_threshold=selection_settings["score_threshold"],
                                              full = selection_settings["full"], 
                                              k = selection_settings["n_eigenvectors"], 
                                              mixing = selection_settings["mixing"],
                                              iterative = selection_settings["iterative_selection"], 
                                              tolerance=selection_settings["score_tolerance"]
                                              )
            feature_selector.fit(train.sps_full, y=train.ens)
        elif selection_method == "CUR":
            feature_selector = CURFeature(n_to_select=n_features, 
                                          progress_bar=selection_settings["progress_bar"], score_threshold=selection_settings["score_threshold"],
                                          full = selection_settings["full"], k = selection_settings["n_eigenvectors"], 
                                          iterative = selection_settings["iterative_selection"], tolerance=selection_settings["score_tolerance"]
                                          )
            feature_selector.fit(train.sps_full)
            print("CUR feature selection")
    else:

        feature_selector=lambda x: x
        feature_selector.transform = feature_selector.__call__

    train.sps_sparse = sample_selector.transform(train.sps_full)

    if sparsify_features:
        train.sps_full = feature_selector.transform(train.sps_full)    
        train.sps_sparse = feature_selector.transform(train.sps_sparse)

    ## WHY?   test.sps_full = feature_selector.transform(test.sps_full)


#         ## NEED TO DEAL WITH THIS LATER

#     if selection_settings["plot_importances"]: # and (sparsify_samples or sparsify_features):
#     #     feature_scores = feature_selector._compute_pi(train_sps_full, train_ens)
#         if sparsify_features:
#             feature_scores = feature_selector.score(train_soaps)
#     #     sample_scores = sample_selector._compute_pi(train_sps_full, train_ens)
#         if sparsify_samples:
#             sample_scores = sample_selector.score(train_soaps)

#         fig, axs = plt.subplots(ncols = 2, figsize=(12, 5))
#         if sparsify_features:
#             axs[0].plot(np.sort(feature_scores)[::-1])
#             axs[0].set_ylabel("Feature Scores")
#         if sparsify_samples:
#             axs[1].plot(np.sort(sample_scores)[::-1] )
#             axs[1].set_ylabel("Sample Scores")
#         for ax in axs:
#             ax.set_ylim(bottom=0)

    return