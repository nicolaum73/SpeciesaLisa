import numpy as np
from pymoo.core.sampling import Sampling

## My BinaryOperator
class MonaLisaSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        
        X = np.full((n_samples, problem.SIZE_VECTOR), None, dtype=object)

        for i in range(n_samples):
            X[i] = np.random.random(problem.SIZE_VECTOR)

        return X