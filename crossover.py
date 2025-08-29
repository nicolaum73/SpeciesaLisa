import numpy as np
from pymoo.core.crossover import Crossover
import copy



class OnePointCrossoverMonaLisa(Crossover):
    """Only One Block is Exchange"""
    def __init__(self, hp:dict, prob=0.5, ):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.CPBX = prob
        self.max_vertices_polygon = hp["maxvp"]
        self.number_polygons = hp["npoly"]
       
        
        self.number_floats_required = hp["number_color_representation"]
        self.size_block = (self.max_vertices_polygon*2 + self.number_floats_required )
        self.indvidual_size = self.size_block* self.number_polygons

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=float)
        
        if float(self.CPBX) < np.random.random():
            
            return X
        else:
            # print("X", X)
            # for each mating provided
            for k in range(n_matings):
                # np.random.choice(size=VEc)
                # get the first and the second parent
                a, b = X[1, k, :], X[0, k, :]

                
                # print(f"\n---\nX-sjape={X.shape} - (n_parents, n_matings, n_var)")
                # print("a=",a, flush=True)
                # print("b=",b, flush=True)
                
                # print("a[i]", a[0])
                # prepare the offsprings
                # print("BEFORE a.shape=",a.shape)
                # print("b.shape=",b.shape)
                off_a = copy.deepcopy(a)
                off_b = copy.deepcopy(b)

                until_which_block = np.random.randint(1, self.number_polygons)
                
                # print(f"until_which_block=",until_which_block)
                # print(f"problem.SIZE_BLOCK*until_which_block=",self.size_block*until_which_block)
                for i in range(0,self.size_block*until_which_block):
                    off_a[i] = b[i]
                    off_b[i] = a[i]

                # print("AFTER a.shape=",a.shape)
                # print("off_a=",off_a)
                # print("off_b=",off_b)
                # print("X.shape=",X.shape)
                # print("a.shape=",a.shape)
                # print("b.shape=",b.shape)
                # join the character list and set the output
                Y[1, k, :], Y[0, k, :] = off_a, off_b

            return Y



# class OnlyOneBlockCrossoverMonaLisa(Crossover):
#     """Only One Block is Exchange"""
#     def __init__(self, prob=0.5):

#         # define the crossover: number of parents and number of offsprings
#         super().__init__(2, 2)
#         self.CPBX = prob

#     def _do(self, problem, X, **kwargs):

#         # The input of has the following shape (n_parents, n_matings, n_var)
#         _, n_matings, n_var = X.shape

#         # The output owith the shape (n_offsprings, n_matings, n_var)
#         # Because there the number of parents and offsprings are equal it keeps the shape of X
#         Y = np.full_like(X, None, dtype=object)

#         # for each mating provided
#         for k in range(n_matings):

#             # get the first and the second parent
#             a, b = X[0, k, 0], X[1, k, 0]

#             # prepare the offsprings
#             off_a = copy.deepcopy(a)
#             off_b = copy.deepcopy(b)

#             for i in range(0,problem.SIZE_VECTOR, problem.SIZE_BLOCK):
#                 if np.random.random() > self.CPBX:
#                     for index_run_block in range(i, problem.SIZE_BLOCK):
#                         off_a[index_run_block] = b[index_run_block]
#                         off_b[index_run_block] = a[index_run_block]

#                     break # Only one block is exchanged

#             # join the character list and set the output
#             Y[0, k, 0], Y[1, k, 0] = off_a, off_b
