

import random
import datetime
import time

import multiprocessing
import numpy as np
import math

import argparse

import copy
import pandas as pd
from pathlib import Path

# import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination

# Algorithms
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.algorithms.moo.nsga3 import NSGA3
# from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
# from pymoo.util.ref_dirs import get_reference_directions # This is required for MOEAD
# the algorithm is based on Reference Directions 

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

# flatten(get_params(moead))

# You only can minimize
from pymoo.optimize import minimize

# Crossover
from pymoo.operators.crossover.pntx import TwoPointCrossover

# # Binary Sampling
# from sampling import BinaryRandomSamplingCustom
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling


# Parallel Slave-Master
from pymoo.core.problem import StarmapParallelization

# Biased initialization
# from pymoo.core.evaluator import Evaluator
# from pymoo.core.population import Population

# Mutation types
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation

# CUSTOMS

from output import FileOutput
from problem import *
from crossover import *
from mutation import *
from sampling import *



## NUMBER OF OBJECTIVES
NUMBER_OBJECTIVES = 1  

# # I/O Configuration
import sys
IN_COLAB = 'google.colab' in sys.modules

base_path = ''
# if IN_COLAB:
#     from google.colab import drive
#     drive.mount('/content/gdrive')
#     base_path = '/content/gdrive/Shareddrives/happy_mob'
#     get_ipython().system('ls /content/gdrive/Shareddrives/happy_mob/')
# else:



if __name__ == "__main__":
           
    """"
        TO HAVE ARGUMENTS FROM COMMAND LINE
    """
    
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-s","-seed", type=int, help='Seed for random', required=True)
    parser.add_argument("-pm", type=float, default=0.1, help='Mutation probability. Probability to mutate. 1 Mutate always. 0 Never change the indvs', required=False)
    parser.add_argument("-pc", type=float, default=0.9, help='Crossover probability.Probability to crossover. 1 Exchange always. 0 Never change the indvs', required=False)
   
    parser.add_argument("-POB","-MU", "-poblacion",  "-pob", type=int, help='Number of individual for population', required=False, default=4)
    parser.add_argument("-GEN","-generacion", "-NGEN","-gen", type=int, help='Number of generations', required=False, default=5)
    parser.add_argument("-CPUS","-cpus","-hilos", type=int, help='Number of cpus', required=False, default=4)
    
    
    ### MonaLisa
    parser.add_argument("-npoly","-number_polygons", type=int, help='Number of polygons', required=False, default=50)
    parser.add_argument("-maxvp", "-max_vertices_polygon", type=int, help='Max vertixes per polygons', required=False, default=3)
    
    
    parser.add_argument("-img", "-image_path", type=str, required=False, default="./data/", help="Where is the real image of the photo")

    parser.add_argument("-swapMUT_PB", "-prob_swap_layers", type=float, help='Probability of mutate an individual by swaping two polygons', required=False, default=0)
    parser.add_argument("-colorMUT_PB", "-prob_mut_color", type=float, help='Probability of mutate an individual by perturbate a color', required=False, default=0)
    parser.add_argument("-alphaMUT_PB", "-alph_mut", type=float, help='Probability of mutate an individual by perturbate an alpha', required=False, default=0)
    parser.add_argument("-shapeMUT_PB","-shape_perturbation_probability", type=float, help='Probability of mutate an individual by changing the whole shape (all the xy that represent the shape)', required=False, default=0)
    parser.add_argument("-coordMUT_PB","-coordinate_perturbation_probability", type=float, help='Probability of mutate an individual by changing only one coordinate selected at random (only one x or y that represent a vertex)', required=False, default=0)
    parser.add_argument("-vertexMUT_PB","-vertex_perturbation_probability", type=float, help='Probability of mutate an individual by changing only one vertex selected at random (only one xy that represent a vertex)', required=False, default=0)
    
    # TODO next_version include fitness to the scale (std_desviation)
    # parser.add_argument("-type_scale","-vertex_perturbation_probability", type=float, help='Probability of mutate an individual by changing only one vertex selected at random (only one xy that represent a vertex)', required=False, default=0.5)
    parser.add_argument("-color", "-c",
                            type=str,
                            choices=['bw', 'BW', 'rgb', 'RGB'],
                            default='BW', required=False,
                            help='Color Representation.  B/W (1) + Alpha or R,G,B (3)+1 ')

    #### MOAED ####    
    parser.add_argument("-tr", "-type_reference","-reference_dir", required=False, help="Reference type for partions", 
                        default="uniform", choices=["uniform","das-dennis","energy","multi-layer","layer-energy","reduction"])
    
    parser.add_argument("-np","-n_partitions", "-n_points", type=int, required=False, help="Number of partitions or the points", 
                        default=7)

    parser.add_argument("-n_neighbors", "-ns", type=int, required=False, help="Size of Neighborhood", 
                        default=3)
    parser.add_argument("-pnm","-prob_neighbor_mating", type=float, required=False, help="Probability Neigbor mating", 
                        default=0.7)


    parser.add_argument("-algorithm", "-algo", "-a",
                        type=str,
                        choices=['NSGA2', 'nsga2', 'GA', 'ga'],
                        default='GA', required=False,
                        help='Choose algorithms between options')
    
    

    parser.add_argument("-mut", "-mutation",
                        type=str,
                        choices=["multiflip","MULTIFLIP"],
                        default='multiflip', required=False,
                        help='Choose algorithms between options')



    ## EVALUATION


    start_time_stamp = time.time() 
    format_start_time = time.strftime("%H:%M:%S %d-%m-%Y",  time.gmtime(start_time_stamp))
    print(f'Started at {time.strftime("%H:%M:%S %d-%m-%Y",  time.gmtime(start_time_stamp))}')
    args = parser.parse_args()
    arguments = vars(args)

    # arguments = {'s': 64, 'pm': 0.01, 'pc': 0.9, 'POB': 8, 'GEN': 1,
    #                 'CPUS': 2, 'v': False, 'show_ind': False,
    #                 'f': None, 'algorithm': 'NSGA2'}
    # print(arguments, flush=True)

    # show_pareto = arguments["v"] # Boolean, print pareto_front individual solution
    # show_ind = arguments["show_ind"] # Boolean, print each individual solution for generation

    




    SEED = arguments["s"] # SEED
      
    random.seed(SEED)  # RANDOM MODULE
    np.random.seed(seed=SEED) # NP RANDOM MODULE

    # ARGUMENTS PARSING

    NGEN = arguments["GEN"] # Number of Generation
    MU = arguments["POB"]  # Number of individual in population if not MOEAD
    cpus = arguments["CPUS"]
    CXPB = arguments["pc"] #Crossover probability
    MUFLIP = arguments["pm"] # Mutation probability of changing each bit

    #### MOEAD options #####
    type_reference = arguments["tr"] #type of reference
    n_p = arguments["np"] #n_partitions or n_points
    n_neighbors = arguments["n_neighbors"] #size of subproblems
    prob_neighbor_mating = arguments["pnm"] # probability of mating neighbor
    #### MOEAD options #####
    
    # MTPB = 0.9 #Mutation probability
    MAX_VERTICES_POLYGON = arguments["maxvp"] # Max vertices per polygon
    NUMBER_POLYGONS = arguments["npoly"]  # Number of polygons

    COLOR_TYPE = (arguments["color"]) #  Color Representation.  B/W (1) + Alpha or R,G,B (3)+Alpha')
    if(COLOR_TYPE.upper() == "BW"):
        number_color_representation = 1 + 1
    else:
        number_color_representation = 3 + 1
    
    arguments["number_color_representation"] = number_color_representation
    
    
    # MUTATION PROBABILITIES
    prob_swap_layers = arguments["swapMUT_PB"]
    prob_color_mut = arguments["colorMUT_PB"]
    prob_alpha_mut = arguments["alphaMUT_PB"]
    
    prob_shape_mut =  arguments["shapeMUT_PB"]
    prob_coord_mut = arguments["coordMUT_PB"]
    prob_vertex_mut = arguments["vertexMUT_PB"]
    


    # NDIM = number_polygons*block_numbers  #Number of dimension of the individual (=number of gene) == 50*(6 for triangle + 2 for color and alpha)
    # print("Number of variables of the individual(==number of gene)",NDIM,flush=True)
    
    algorithm_choice=arguments["algorithm"].upper()

    type_mutation = arguments["mut"].upper()

    real_image_path = arguments["img"]

    
    

    scenario = f'{MAX_VERTICES_POLYGON}-{NUMBER_POLYGONS}-{number_color_representation}-{SEED}'
    scenario += f'-{NGEN}-{MU}-{CXPB:02.3f}-{MUFLIP:02.3f}-{cpus}-{algorithm_choice}'
    scenario += f'-{prob_swap_layers:02.3f}-{prob_color_mut:02.3f}-{prob_alpha_mut:02.3f}'
    scenario += f'{prob_shape_mut:02.3f}-{prob_coord_mut:02.3f}-{prob_vertex_mut:02.3f}'
    scenario += '-{}'.format(time.strftime("%Y-%m-%d_%H-%M-%S",  time.gmtime(start_time_stamp)))


    arguments['base_path'] = "./"
    
    base_path = '.'

    data_path = base_path + '/data/'


    results_path = base_path + "/results/" + scenario + "/"
    images_path = results_path + "images/"

    arguments['results_path'] = results_path
    arguments['images_path'] = images_path

    Path(results_path).mkdir(parents=True, exist_ok=True)
    Path(images_path).mkdir(parents=True, exist_ok=True)

    Path(data_path).mkdir(parents=True, exist_ok=True)


    print(arguments, flush=True)

    # initialize the multiprocessing pool and create the runner
    with multiprocessing.Pool(cpus) as pool:
        # n_proccess =4
        # pool = multiprocessing.Pool(n_proccess)
        runner = StarmapParallelization(pool.starmap)
        # define the problem by passing the starmap interface of the multiprocessing pool
        # print("BEFORE", arguments, flush=True)
        problem = MonaLisaProblem(elementwise_runner=runner,hp=arguments)
        # print("\nAFTER", arguments, flush=True)
        # sys.exit(-1)
        
        time_stamp = time.time() 

        output2 = FileOutput(filepath=results_path + "fitness_values.csv", show_ind=True)

        # Type of mutations
        print("Init Mutation First", flush=True)
        if(type_mutation == "MULTIFLIP"):
            mutation = MonaLisaMutation(prob=MUFLIP,hp=arguments)
                                        
            
        else: # (type_mutation == "ANOTHER"): 
            raise("Not Implemented")
        
        print("Post Mutation First", flush=True)
        
        print("Pre Crossover", flush=True)
        crossover = OnePointCrossoverMonaLisa(prob=CXPB, hp=arguments)
        print("Post Crossover", flush=True)
        
        print("Sampling", flush=True)
        sampling = MonaLisaSampling(hp=arguments)
        # TODO INIT the individuals with only one figure


        print("Post Sampling", flush=True)





        print("Algorithm Choice",flush=True)
        if(algorithm_choice == "NSGA2"):
            algorithm = NSGA2(
                pop_size=MU,
                termination=get_termination("n_gen", NGEN),
                sampling = sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True,
                output=output2
                )
        elif(algorithm_choice == "GA"):
            algorithm = GA(
                pop_size=MU,
                termination=get_termination("n_gen", NGEN),
                sampling = sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True,
                output=output2
                )
        print("Post Algorithm Choice",flush=True)
        

    
        end_time_stamp = time.time() 
    
        print("Pre-Minimize",flush=True)
        results = minimize(problem, algorithm, verbose=True, termination=("n_gen", NGEN), seed=SEED )
        print("Post-Minimize",flush=True)
        # print(results)
        # print('Threads:', results.exec_time)
        # print("Best solution: \nX = %s\nF = %s" % (results.X.astype(int), results.F))
        

        print("fin de Main\n",flush=True)
        end_time_stamp = time.time() 
        print('\n\nProcessed population {} in NGEN {} in {} seconds. CXPB {} MUFLIP {} CPUS {} algorithm {}\n'.format(MU,NGEN,float(end_time_stamp-start_time_stamp),CXPB, MUFLIP,cpus,algorithm_choice))
    
    print("\n-----END OF THE JOB-----------\n",flush=True)        
    print("\n-----END OF THE JOB-----------\n",flush=True,file=sys.stderr)    


