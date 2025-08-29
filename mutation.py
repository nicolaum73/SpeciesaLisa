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
    def __init__(self, prob=1, hp={}, **kwargs):

        super().__init__(prob, **kwargs)

        self.MUPB = prob

        self.prob_swap_layers = hp["swapMUT_PB"]
        self.prob_color_mut = hp["colorMUT_PB"]
        self.prob_alpha_mut = hp["alphaMUT_PB"]

        self.prob_shape_mut =  hp["shapeMUT_PB"]
        self.prob_coord_mut = hp["coordMUT_PB"]
        self.prob_vertex_mut = hp["vertexMUT_PB"]

        self.add_remove_vertex = 0

        # Characteristics of the Polygons and Shapes
        self.max_vertices_polygon = hp["maxvp"]
        self.number_polygons = hp["npoly"]
        self.number_floats_required = hp["number_color_representation"]
        self.size_block = (self.max_vertices_polygon*2 + self.number_floats_required )
        self.indvidual_size = self.size_block* self.number_polygons


    def layers_alpha_enables(individual, number_polygons)->list:
        array_polygons = np.split(individual, number_polygons)
        matrix = np.stack(array_polygons)  # Forma (5,9)
        positions_list_list_layers = np.where(matrix[:, -1] > 0)
        active_layers = matrix[positions_list_list_layers]
        positions_list_layers = positions_list_list_layers[0]
        return positions_list_layers, active_layers, array_polygons


    def swap_two_layers_up(positions_list_layers, array_polygons):

        if (len(positions_list_layers)>1):
            a,b = np.random.choice(positions_list_layers, 2, replace=False) # layers should be differents
            print(f"a=",a,"b=", b, flush=True)
            print("-----------\n")
            array_polygons[a], array_polygons[b] =  array_polygons[b], array_polygons[a]        
        
        individual = np.vstack(array_polygons)
        return individual
    
    def _do(self, problem, X, **kwargs):
        if  self.MUPB < float(np.random.random()):
            return X

        elif self.prob_swap_layers < float(np.random.random()):
            print(X.shape)

            for ind in range(len(X)):
                individual = X[ind]
                positions_list_layers, _, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # print(f"Active_layers={active_layers}")
                # print(f"Array Poly{array_polygons}")
                new_ind = self.swap_two_layers_up(positions_list_layers, array_polygons)

                # print("BEFORE", individual)
                # print("AFTER", new_ind)

                # print("BEFORE",X[ind])
                # print("new_ind.shape", new_ind.shape)
                X[ind] = new_ind.flatten()
                # print("\nAFTER",X[ind])

            return X

        elif self.prob_color_mut < float(np.random.random()):
            # TODO Change color
            pass
        elif self.prob_alpha_mut < float(np.random.random()):
            # TODO Change alpha
            pass

        elif self.prob_shape_mut < float(np.random.random()):
            # TODO Change shape of a layer
            pass

        elif self.prob_coord_mut < float(np.random.random()):
            # TODO Change coord (x or y) of a layer
            pass

        elif self.prob_vertex_mut < float(np.random.random()):
            # TODO Change prob_vertex
            pass

        elif self.add_remove_vertex < float(np.random.random()):
            # TODO Add or Remove a Vertex
            pass




        return X


    def example(self, problem, X):
        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - oermutation between blocks (layers)
            if r < 0.4:
                layer_1, layer_2 = np.random.randint(2, problem.NUMBER_POLYGONS)
                print("layer_1=",layer_1, "layer_2=",layer_2, flush=True)
                perm = np.random.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [c if np.random.random() > prob
                       else np.random.choice(problem.ALPHABET) for c in X[i, 0]]
                X[i, 0] = "".join(mut)