import numpy as np
import copy


from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp
    

class MonaLisaMutation(Mutation):
    """
        Mutation for MonalisaProblem

        Permutation of the layer
        Mutations of the coordinates
        Mutation of the alpha
        Mutation of the color
    """
    def __init__(self, prob=1, prob_var=None, **kwargs):
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X
        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - oermutation between blocks (layers)
            if r < 0.4:

                perm = np.random.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [c if np.random.random() > prob
                       else np.random.choice(problem.ALPHABET) for c in X[i, 0]]
                X[i, 0] = "".join(mut)

        return X