import numpy as np
from pymoo.core.sampling import Sampling

## My BinaryOperator
class MonaLisaSampling(Sampling):
    def __init__(self, hp, **kwargs):
        super().__init__()
        self.size_individuals = hp["individual_size"]
        self.number_polygons = hp["npoly"]
        self.size_block = hp["size_block"]
        

    def _do(self, problem, n_samples, **kwargs):
        
        X = np.full((n_samples,  self.size_individuals), 0, dtype=float)

        for ind in range(n_samples):
            chosen_one = np.random.randint(0,  self.number_polygons)

            start = self.size_block * chosen_one
            end = start + self.size_block

            # print("chosen_one",chosen_one)
            # print("start",start, "end",end)
            for i in range(start, end):
                X[ind,i] = np.round(np.random.random(),2)
            
        # print("\n-----\n","Sampling X", X, "\n----\n")

        return X