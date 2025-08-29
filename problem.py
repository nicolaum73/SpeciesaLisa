
from pymoo.core.problem import ElementwiseProblem
import numpy as np


from evaluation import *



#################################################
##  
##  PROBLEM CREATION PYMOO STYLE
##
#################################################

class MonaLisaProblem(ElementwiseProblem):

    # basic setups
    # # n_var -> number of variables
    # # n_obj -> number of objectives
    # # xl -> lower boundaries of variables
    # # xu -> upper boundaries of variables
    def __init__(self, hp:dict, **kwargs):
        """
            kwargs it used to the Hyperparameters
            image_real_path : where the real image
            generation_folder_path = path of the folder where to put the generate images

            max_vertices_polygon: Max number of vertices. For example, if only are allowed triangles, then is 3. If are allowed triangles and 
            number_polygons: Represent how many polygons we will set as a maximum to be used.
            
        """

        max_vertices_polygon = hp["maxvp"]
        number_polygons = hp["npoly"]
       
        
        number_floats_required = hp["number_color_representation"]
        size_block = (max_vertices_polygon*2 + number_floats_required )
        indvidual_size = size_block* number_polygons
        

        super().__init__(n_var=indvidual_size, n_obj=1, n_ieq_constr=0)
        
        # Adding BLOCK SIZE
        hp["size_block"] = size_block
        hp["individual_size"] = indvidual_size

        # self.image_real_path = image_real_path
        # self.generation_folder_path = generation_folder_path

        # self.NUMBER_POLYGONS = number_polygons
        # self.MAX_NUMBER_VERTICES = max_vertices_polygon
        # self.SIZE_BLOCK = (self.MAX_NUMBER_VERTICES*2 + (c_a*2+1) )
        # self.C_A = (c_a*2+1)
        # self.SIZE_VECTOR = self.SIZE_BLOCK * self.NUMBER_POLYGONS
        
        
        # print("Number of variables of the individual(==number of gene)",self.SIZE_VECTOR ,flush=True)
        # print("BLOCK SIZE",self.SIZE_BLOCK ,flush=True)
        
        


    #  x, out are needed, it's basic
    #  x -> solutions, for this kind of Problem, the x is a (population, n_var) matrix. 
    #  out -> the corresponding results, including function values, constrain values and so on. 
    #        we only fill what we need. 
    def _evaluate(self, x, out, *args, **kwargs):
        # HOW to evaluate 
        # Coordinates  width -1, height -1 
        # Alpha = 0 - 255
        # Color = 0 - 255
        out["F"] = (np.random.random()) 
        # # out["F"] -> Fitness
        # # out["G"] -> Constrains values
        # out["F"] = (np.random.random(),np.random.random()) 
        # evaluation(self.NUMBER_POLYGONS,
        #             self.MAX_NUMBER_VERTICES,
        #             sequence_number) transformed
        # print("\tI am doing an evaluation\n",flush=True)
        
        # objective1, objective2, objective3 = evaluation(x.astype(int), list_pair_od=self.list_pair_od, extremes_min_max=self.extremes_min_max)
        # out["F"] = tuple((objective1, objective2, objective3))