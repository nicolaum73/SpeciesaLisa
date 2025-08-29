
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
    def __init__(self, image_real_path:str, generation_folder_path:str, number_polygons:int, max_vertices_polygon:int, c_a:bool, **kwargs):
        """
            image_real_path : where the real image
            generation_folder_path = path of the folder where to put the generate images

            max_vertices_polygon: Max number of vertices. For example, if only are allowed triangles, then is 3. If are allowed triangles and 
            number_polygons: Represent how many polygons we will set as a maximum to be used.
            c_a: If it is True, then it should have 3 floats for color. Otherwise, b/n only need 1
        """

        
        super().__init__(n_var=(max_vertices_polygon*2 + (c_a*2+1) ) * number_polygons, n_obj=1, n_ieq_constr=0)
        self.image_real_path = image_real_path
        self.generation_folder_path = generation_folder_path

        self.NUMBER_POLYGONS = number_polygons
        self.MAX_NUMBER_VERTICES = max_vertices_polygon
        self.SIZE_BLOCK = (self.MAX_NUMBER_VERTICES*2 + (c_a*2+1) )
        self.SIZE_VECTOR = self.SIZE_BLOCK * self.NUMBER_POLYGONS
        
        print("Number of variables of the individual(==number of gene)",self.SIZE_VECTOR ,flush=True)
        print("BLOCK SIZE",self.SIZE_BLOCK ,flush=True)
        # super().__init__(n_var=size_vector,
        #         n_obj=1,
        #         xl=0,
        #         xu=1, vtype=float, **kwargs)

    #  x, out are needed, it's basic
    #  x -> solutions, for this kind of Problem, the x is a (population, n_var) matrix. 
    #  out -> the corresponding results, including function values, constrain values and so on. 
    #        we only fill what we need. 
    def _evaluate(self, x, out, *args, **kwargs):
        # HOW to evaluate 

        out["F"] = (np.random.random()) 
        # # out["F"] -> Fitness
        # # out["G"] -> Constrains values
        # out["F"] = (np.random.random(),np.random.random()) 
        
        # print("\tI am doing an evaluation\n",flush=True)
        
        # objective1, objective2, objective3 = evaluation(x.astype(int), list_pair_od=self.list_pair_od, extremes_min_max=self.extremes_min_max)
        # out["F"] = tuple((objective1, objective2, objective3))


