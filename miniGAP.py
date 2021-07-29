#!/usr/bin/env python
# coding: utf-8
import time
initial_time = time.time()
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from ase.io import read
import numpy as np
import cProfile #added on 7/14


if True:
    # --------------------------------------------  commandline argument code   -----------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true', help="Print out details at each step") 

    parser.add_argument('-v', '--version', action='store_true', help="Print out current version of miniGAP and most recent update") 

    #parser.add_argument('-t', '--time', action='store_true', help="Option to time tasks")

    parser.add_argument('-sf', '--structure_file', default=None, help="Specify a structure file to import. If none given and -md used, will generate diatomics.")

    parser.add_argument("-md", '--molecular_dynamics', action='store_true', help="Indicate if you want molecular dynamics performed. Will generate diatomic if no structure given")
    parser.add_argument('-mdi', '--md_indices', default=[0], type=int, nargs='*', help="If performing molecular dynamics on a structure file with multiple structures, you can give indices of all structures to perform md on.")

    #structure_source = parser.add_mutually_exclusive_group(required=True)
    #structure_source.add_argument('-gd', '--generate_diatomics', action='store_true', help="Option to generate a MD trajectory of diatomic molecules")
    #structure_source.add_argument('-is', '--import_structure', help="Specify a structure file to import")

    parser.add_argument('-n', '--n_structs', default=0, type=int, help="Specify # of diatomic molecules or # of structures to use from input file")

    # arguments specific to generating diatomics
    parser.add_argument('-el', '--element', default="N", choices = ["N", "O", "H"], help="If generating diatomics, specify element")
    parser.add_argument('--temp', default=300, type=float, help="If generating diatomics, specify temperatutre (K) of MD")
    parser.add_argument('-mds', '--md_seed', default=1, type=int, help="If generating diatomics, change this seed to get different trajectories")
    parser.add_argument('-ec', '--energy_calculator', default="EMT", choices = ["EMT", "LJ", "Morse"], help = "If generating diatomics, specify ASE energy/force calculator")

    # arguments specific to soap
    parser.add_argument('-si', '--soap_implementation', default = 'dscribe', choices = ["dscribe", "None"], help = "Option to calculate soap descriptors")
    parser.add_argument('--rcut', default=3.2, type=float, help= "Choice of SOAP cut off radius")
    parser.add_argument('--nmax', default =5, type=int, help="Choice of SOAP n_max")
    parser.add_argument('--lmax', default =5, type=int, help="Choice of SOAP l_max")
    parser.add_argument('-ls', '--local_soap', action='store_true', help="Option to generate a SOAP descriptor for each atom, rather than each structure")

    # arguments specific to learning
    parser.add_argument('-rm', '--regression_models', nargs='*', default=["GP_Tensorflow"], choices= ["GP_sklearn", "GP_Tensorflow", "Polynomial"], help = "Choice of regression models. No regression will be performed if you use -rm with no value")
    parser.add_argument('-ss', '--split_seed', default=1, type=int, help="Random seed for cross-validation")
    parser.add_argument('-tf', '--train_fraction', default=0.8, type=float, help="If performing regression, specify the fraction of structures used in training")
    parser.add_argument('-po', '--polynomial_order', nargs='*', type=int, default=[2], help = "If using polynomial regression model, choice polynomial order(s)")

    # some housekeeping
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ---------------------------------------------------------------------------------------------------------------------------------------------------- 





    # --------------------------------------------------------     descriptive code    -------------------------------------------------------------------

    # print out version and exit if requested
    version = "0.3"
    date = "June 15th, 2021"
    if args.version:
        from miniGAP_functions import TimedExit
        print("Current version of miniGAP is {} as of {}".format(version, date))
        TimedExit(initial_time)


    # for initial version of the script, this will be helpful. Will remove later
    print("\n##########################   Printing out args for debugging   ##############################")
    print(args)
    if len(args.remainder):
        print("Ignoring these arguments, because they are not recognized: {}".format(args.remainder))
    print("#############################################################################################\n\n")


    # Print out an overview of what the script is doing  
    action1 = "generate diatomic molecules" if (args.structure_file is None) else str("import structures from {}".format(args.structure_file) + str(" and use them to generate new structures with molecular dynamics" if args.molecular_dynamics else ""))
    action2 = "\n                   (2) create dscribe soap descriptors from them" if args.soap_implementation=="dscribe" else ""
    action3 = "\n                   (3) fit a regression model to learn their energies" if len(args.regression_models) else ""
    script_overview_beginning = "This script will:\n                   (1) "
    script_overview = script_overview_beginning + "{}{}{} and then terminate.".format(action1, action2, action3)
    print(script_overview)
    print("".join("_" for i in range(max(len(action1 + script_overview_beginning), len(action2), len(action3)) + 10))+"\n")
    # ----------------------------------------------------------------------------------------------------------------------------------------------------




    # -------------------------------------------------     importing       -----------------------------------------------------------------------------
    # Import some functions I wrote that will soon be used. I didn't import them at the beginning to minimize the time until the first printed message
    from miniGAP_functions import TimedExit, TickTock
    # ----------------------------------------------------------------------------------------------------------------------------------------------------


    # ---------------------------     code for getting structures -> ASE Atoms lists  --------------------------------------------------------------------
    print("(1)")
    if args.structure_file is None:
        if True:#args.molecular_dynamics:
            from miniGAP_functions import generate_md_traj, assign_calc
            import numpy.random as rand
            rand.seed(args.md_seed)

            print("Generating a MD trajectory of {} {}₂ molecules at temperature {:.0f} K (md_seed={}).".format(args.n_structs, args.element, args.temp, args.md_seed))
            AtomsList, TrajTime = TickTock(generate_md_traj,from_diatomic=True, element=args.element, verbose=args.verbose, nsteps = args.n_structs - 1, temperature=args.temp, 
                                           print_step_size=100, calc_type=args.energy_calculator)
            print("Successfully generated and stored MD trajectory. This took {:.3f} seconds.\n".format(TrajTime))
        else:
            print("No structure file provided and -md flag not used to generate an molecular dynamics trajectory of diatomic molecules. No action performed. Script terminating.")    
            TimedExit(initial_time)
    else:
        print("Importing structures from {}".format(args.structure_file))
        TempAtomsList, ImportTime = TickTock(read, args.structure_file, index=':')
        if not args.molecular_dynamics:
            all_struct_n = len(TempAtomsList)
            if args.n_structs:
                AtomsList = TempAtomsList[:min(all_struct_n, args.n_structs)]
            else:
                AtomsList = TempAtomsList
            if len(AtomsList) < all_struct_n:
                struct_n_str = "the first {} out of {}".format(len(AtomsList), all_struct_n)
            elif len(AtomsList) == all_struct_n:
                struct_n_str = "all {}".format(all_struct_n)
            print("Imported {} structures from {}. Import took {:.3f} seconds.\n".format(struct_n_str, args.structure_file, ImportTime))
        else:
            from miniGAP_functions import generate_md_traj, assign_calc
            import numpy.random as rand
            rand.seed(args.md_seed)

            print("Import took {:.3f} seconds. Using imported structures to generate one or more MD trajectories at temperature {:.0f} K (md_seed={}).".format(ImportTime, args.temp, args.md_seed))
            AtomsList = []
            TrajTime = 0
            for i in args.md_indices:
                AtomsList_i, TrajTime_i = TickTock(generate_md_traj, structure = TempAtomsList[i], from_diatomic=False,  verbose=args.verbose, nsteps = args.n_structs - 1, temperature=args.temp, 
                                                   print_step_size=100, calc_type=args.energy_calculator)
                AtomsList += AtomsList_i
                TrajTime += TrajTime_i
            if len(args.md_indices) == 1:
                print("Successfully generated a {}-structure MD trajectory from structure #{} in {}. This took {:.3f} seconds.\n".format(args.n_structs, args.md_indices[0], args.structure_file, TrajTime))
            elif len(args.md_indices) > 1:
                index_list_string = ", ".join("#"+str(index) for index in args.md_indices[:-1]) + " and #" + str(args.md_indices[-1])
                print("Successfully generated a {}-structure MD trajectory for each of imported structures {} from {} for a total of {} generated structures. This took a total of {:.3f}s.\n".format(args.n_structs, index_list_string, args.structure_file, len(AtomsList), TrajTime))

    n_structs = len(AtomsList)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------


    
    
    
    # ---------------------------------------------------       code for getting soaps    --------------------------------------------------------------------

    if args.soap_implementation == "None":
        print("No further action requested. Script terminating.\n")
        TimedExit(initial_time)
    elif args.soap_implementation == "dscribe":
        from miniGAP_functions import get_dscribe_descriptors
        elements = AtomsList[0].get_chemical_symbols()
        soap_scope = "local" if args.local_soap else "global"
        print("(2)\nGenerating {} soap descriptors for {} structures using dscribe with nmax = {}, lmax = {}, and rcut = {:.2f}".format(soap_scope, n_structs, args.nmax, args.lmax, args.rcut))
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            SoapList, SoapTime = TickTock(get_dscribe_descriptors, AtomsList, species=elements, rcut = args.rcut, nmax = args.nmax, lmax = args.lmax, is_global = not args.local_soap)
        except ValueError:
            elements = np.unique(np.array([atoms.get_chemical_symbols() for atoms in AtomsList]).flatten())
            
            SoapList, SoapTime = TickTock(get_dscribe_descriptors, AtomsList, species=elements, rcut = args.rcut, nmax = args.nmax, lmax = args.lmax, is_global = not args.local_soap)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()


        print("Successfully generated soap descriptors in {:.3f} seconds.\n".format(SoapTime))
    # ----------------------------------------------------------------------------------------------------------------------------------------------------



    # -------------------------------------------------------- code for performing regression -----------------------------------------------------------------

    if len(args.regression_models):# and "None" not in args.regression_models:
        print("(3)")
        if args.local_soap:
            print("This script is not yet set up to learn from local soap descriptors. Terminating.")
            exit()
        # Make a list of dictionaries which contain all details about models
        model_param_dicts = []
        for model in args.regression_models:
            if model == "Polynomial":
                for order in args.polynomial_order:
                    model_param_dicts.append({"model_type":model, "order":order, "model_name":"{} Order {}".format(model, order)})
            else:
                model_param_dicts.append({"model_type":model, "order":None, "model_name":model})

        EnergyList = np.array([atom.get_total_energy() for atom in AtomsList], dtype=np.float32)
        from miniGAP_functions import LearnEnergyFromSoap, GetErrorFromModel
        for params in model_param_dicts:
            print("Producing a {} model which learns energies from soap descriptors.".format(params["model_name"]))
            RegressionOutputs, RegressionTime = TickTock(LearnEnergyFromSoap, SoapList, EnergyList, training_fraction=args.train_fraction, verbose=args.verbose, model_type=params["model_type"],
                                                         split_seed = args.split_seed, gamma = 300, order=params["order"])
            RegressionModel, TestSoapList, TestEnergyList, TrainSoapList, TrainEnergyList = RegressionOutputs

            [AbsoluteError, RMSError, R2], ErrorTime = TickTock(GetErrorFromModel,RegressionModel, TestSoapList, TestEnergyList, error_types = ["absolute", "rms", "r2"])

            print("For this {}-structure dataset, the {} regression model with seed {} and training fraction {} returns absolute error of {:.5f}, rms error of {:.5f}, and" \
                  " r^2 of {:.5f}.\nThe training took {:.3f}s and the prediction/error calculation took {:.3f}s.\n".format(n_structs, params["model_name"], args.split_seed, args.train_fraction, AbsoluteError, RMSError, R2, RegressionTime, ErrorTime))
    #  ------------------------------------------------------------------------------------------------------------------------------------------------------------



    TimedExit(initial_time)
    
# if __name__ == '__main__':
#     import cProfile
#     cProfile.run('main()')
    
# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()
#     profiler.dump_stats("miniGAP_profiler")
