
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from PIL import Image

from evaluation import *



#################################################
##  
##  PROBLEM CREATION PYMOO STYLE
##
#################################################
import os

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

        self.max_vertices_polygon = hp["maxvp"]
        self.number_polygons = hp["npoly"]
       
        
        number_floats_required = hp["number_color_representation"]
        self.size_block = (self.max_vertices_polygon*2 + number_floats_required )
        self.indvidual_size = self.size_block* self.number_polygons
        

        super().__init__(n_var=self.indvidual_size, n_obj=1, n_ieq_constr=0)
        
        # Adding BLOCK SIZE
        hp["size_block"] = self.size_block
        hp["individual_size"] = self.indvidual_size

        delete_test_images("./test_images/")

        target_image = Image.open(hp["img"]).convert("L").convert("RGBA")
        self.target = np.array(target_image)
        self.t_width = target_image.width
        self.t_height = target_image.height
        target_image.save("./test_images/target.png")

        Prediction.set_target(self.target)

        # self.image_real_path = image_real_path
        # self.generation_folder_path = generation_folder_path

        # self.NUMBER_POLYGONS = number_polygons
        # self.MAX_NUMBER_VERTICES = max_vertices_polygon
        # self.SIZE_BLOCK = (self.MAX_NUMBER_VERTICES*2 + (c_a*2+1) )
        # self.C_A = (c_a*2+1)
        # self.SIZE_VECTOR = self.SIZE_BLOCK * self.NUMBER_POLYGONS
        
        
        # print("Number of variables of the individual(==number of gene)",self.SIZE_VECTOR ,flush=True)
        # print("BLOCK SIZE",self.SIZE_BLOCK ,flush=True)
        
    def solution_to_prediction(self, x):
        tris = []
        float_interpolate = lambda val, d_size: int(val*(d_size-1))
        for i in range(0, self.indvidual_size, self.size_block):
            tone = float_interpolate(x[i + 2*self.max_vertices_polygon], 256)
            alpha = float_interpolate(x[i + 2*self.max_vertices_polygon + 1], 256)
            if alpha == 0: continue
            # alpha = 127
            points = []
            for j in range(i, i + 2*self.max_vertices_polygon, 2):
                points.append(Point(float_interpolate(x[j], self.t_width), float_interpolate(x[j+1], self.t_height)))
            tris.append(Triangle(points, (*((tone,) * 3), alpha)))
            
        return Prediction(tris, (self.t_width, self.t_height))

    #  x, out are needed, it's basic
    #  x -> solutions, for this kind of Problem, the x is a (population, n_var) matrix. 
    #  out -> the corresponding results, including function values, constrain values and so on. 
    #        we only fill what we need. 
    def _evaluate(self, x, out, *args, **kwargs):
        pred = self.solution_to_prediction(x)
        pred.render()
        fitness = pred.evaluate()
        # print(f"Fitness: {fitness}")
        # pred.show_error().save(f"./test_images/Error {fitness:.3f}|| {pred}.png")
        out["F"] = fitness
        #             self.MAX_NUMBER_VERTICES,
        #             sequence_number) transformed
        # print("\tI am doing an evaluation\n",flush=True)
        
        # objective1, objective2, objective3 = evaluation(x.astype(int), list_pair_od=self.list_pair_od, extremes_min_max=self.extremes_min_max)
        # out["F"] = tuple((objective1, objective2, objective3))

def delete_test_images(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # delete the file
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")
