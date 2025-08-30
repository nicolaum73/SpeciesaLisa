import numpy as np
from pymoo.core.sampling import Sampling

## My BinaryOperator
class MonaLisaSampling(Sampling):
    def __init__(self, hp, **kwargs):
        super().__init__()
        self.size_individuals = hp["individual_size"]
        self.number_polygons = hp["npoly"]
        self.size_block = hp["size_block"]
        self.coords_per_polygon = hp["maxvp"]*2

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples,  self.size_individuals), 0, dtype=float)
        # pop = np.reshape(np.random.randint(0, 256, pop_size*polygons*self.size_block)/255, (pop_size, polygons*self.size_block))

        # ind = np.random.randint(0, 256,  self.size_individuals).astype(float)/255
        # ind /= 255
        # ind[self.coords_per_polygon::self.size_block] = 0  # Set alpha to 0 (deactivated)
        

        # X = np.full((n_samples,  self.size_individuals), 0, dtype=float)

        for i in range(n_samples):
            ind = np.random.randint(0, 256,  self.size_individuals).astype(float)/255
            
            ind[self.size_block-1::self.size_block] = 0
            ind[np.random.randint(self.number_polygons) * self.size_block + (self.size_block - 1)] = np.random.randint(51, 256) / 255
            # ind[np.random.randint(polygons) * values_per_polygon + (values_per_polygon - 1)] = np.random.randint(256) / 255
            # # X[ind] = np.random.randint(0, 256,  self.size_individuals).astype(float)/255
            # # X[ind][self.coords_per_polygon::self.size_block] = 0 # desactivated all
            # # X[ind][np.random.randint(self.number_polygons)] = np.random.randint(50, 256,  self.size_individuals).astype(float)/255
            # chosen_one = np.random.randint(0,  self.number_polygons)

            # start = self.size_block * chosen_one
            # end = start + self.size_block

            # # print("chosen_one",chosen_one)
            # # print("start",start, "end",end)
            # for i in range(start, end):
            #     X[ind,i] = np.random.random()
            X[i] = ind
            
        # print("\n-----\n","Sampling X", X, "\n----\n")

        return X