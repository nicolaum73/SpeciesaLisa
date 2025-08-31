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

    @staticmethod
    def layers_alpha_enables(individual, number_polygons)->list:
        array_polygons = np.split(individual, number_polygons)
        matrix = np.stack(array_polygons)  # Forma (50,8

        positions_list_list_layers = np.where(matrix[:, -1] > 0)
        active_layers = matrix[positions_list_list_layers]
        positions_list_layers = positions_list_list_layers[0]
        return positions_list_layers, active_layers, array_polygons

    @staticmethod
    def all_layers(individual, number_polygons)->list:
        array_polygons = np.split(individual, number_polygons)
        return [i for i in range(number_polygons)], array_polygons


    @staticmethod
    def swap_two_layers_up(positions_list_layers, array_polygons):

        if (len(positions_list_layers)>1):
            a,b = np.random.choice(positions_list_layers, 2, replace=False) # layers should be differents
            # # print(f"a=",a,"b=", b, flush=True)
            # # print("-----------\n")
            array_polygons[a], array_polygons[b] =  array_polygons[b], array_polygons[a]        
        
        individual = np.vstack(array_polygons)
        return individual
    
    @staticmethod
    def choose_one_layer_from_active_layers(positions_list_layers, array_polygons)->int:
    
        position_layer_selected = np.random.choice(positions_list_layers, 1, replace=False) # layers should be differents
        # # print(f"position_layer_selected=",position_layer_selected, flush=True)
        # # print("-----------\n")
            
        
        return position_layer_selected[0]
    
    @staticmethod
    def choose_one_polygon_from_list(positions_list_layers)->int:

        position_layer_selected = np.random.choice(positions_list_layers, 1, replace=False) # layers should be differents
        # # print(f"position_layer_selected=",position_layer_selected, flush=True)
        # # print("-----------\n")
        return position_layer_selected[0]
    
    def _do(self, problem, X, **kwargs):
        if  self.MUPB < float(np.random.random()):
            # # print("ESTO ESTA OCURRIENDO PORQUE LA MUTACION NO DEBERIA HABER PASADO", flush=True)
            return X

        if self.prob_swap_layers > float(np.random.random()):
            # # print(X.shape)

            for ind in range(len(X)):
                individual = X[ind]
                positions_list_layers, _, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # # print(f"Active_layers={active_layers}")
                # # print(f"Array Poly{array_polygons}")
                if (len(positions_list_layers)==0):
                    continue
                new_ind = self.swap_two_layers_up(positions_list_layers, array_polygons)


                # # print("BEFORE", individual)
                # # print("AFTER", new_ind)

                # # print("BEFORE",X[ind])
                # # print("new_ind.shape", new_ind.shape)
                X[ind] = new_ind.flatten()
                # # print("\nAFTER",X[ind])

            # return X

        if self.prob_color_mut > float(np.random.random()):
            for ind in range(len(X)):
                individual = X[ind]
                positions_list_layers, active_layers, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # # print(f"Active_layers={active_layers}")
                # # print(f"Array Poly{array_polygons}")
                if (len(positions_list_layers)==0):
                    continue
                
                position_chosen = (self.choose_one_layer_from_active_layers(positions_list_layers=positions_list_layers, array_polygons=array_polygons))
                polygon = (array_polygons[position_chosen])
                
                coord_chosen = self.size_block- 2 # POSITION OF COLOR
                # # print("coord_chosen", coord_chosen)
                value_coord_chosen = polygon[coord_chosen]
                # # print("value_coord_chosen", value_coord_chosen)

                new_coord_value = max(0, min(1, np.random.normal(loc=value_coord_chosen, scale=0.1, size=1)[0]))

                # # print("new_coord_value", new_coord_value)

                # # print("......\n")


                polygon[coord_chosen] = new_coord_value
                array_polygons[position_chosen] = polygon

            # TODO Change color
            pass
        if self.prob_alpha_mut > float(np.random.random()):
            for ind in range(len(X)):
                individual = X[ind]
                
                positions_list_layers, array_polygons = self.all_layers(individual=individual, number_polygons=self.number_polygons )
                # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # print(f"Array Poly{array_polygons}")

                
                position_chosen = self.choose_one_polygon_from_list(positions_list_layers=positions_list_layers)
                polygon = (array_polygons[position_chosen])
                
                coord_chosen = self.size_block- 1 # POSITION OF ALPHA
                # print("coord_chosen", coord_chosen)
                value_coord_chosen = polygon[coord_chosen]
                # print("value_coord_chosen", value_coord_chosen)

                if (value_coord_chosen==0): # It is deasactivated
                    new_coord_value = np.random.randint(low=51, high=256, size=1)[0] /255 
                else:
                    new_coord_value = max(0, min(1, np.random.normal(loc=value_coord_chosen, scale=0.1,size=1)[0]))
                    if (new_coord_value<0.2):
                        new_coord_value = 0

                # print("new_coord_value", new_coord_value)

                # print("......\n")


                polygon[coord_chosen] = new_coord_value
                array_polygons[position_chosen] = polygon
                new_ind = np.vstack(array_polygons)
                # # print("BEFORE", individual)
                # print("AFTER", new_ind)

                # print("BEFORE",X[ind])
                # print("new_ind.shape", new_ind.shape)
                X[ind] = new_ind.flatten()
                # print("\nAFTER",X[ind])
            pass

        if self.prob_shape_mut > float(np.random.random()):
            # TODO Change shape of a layer
            for ind in range(len(X)):
                individual = X[ind]
                positions_list_layers, active_layers, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # # print(f"Active_layers={active_layers}")
                # # print(f"Array Poly{array_polygons}")
                if (len(positions_list_layers)==0):
                    continue
                
                position_chosen = (self.choose_one_layer_from_active_layers(positions_list_layers=positions_list_layers, array_polygons=array_polygons))
                polygon = (array_polygons[position_chosen])
                # # print("Polygon before mutation", polygon, flush=True)
                for i in range(0, self.size_block):
                    polygon[i] = max(0, min(1, np.random.normal(loc=polygon[i], scale=0.1,size=1)[0]))
                # # print("Polygon after mutation", polygon, flush=True)

                array_polygons[position_chosen] = polygon

                new_ind = np.vstack(array_polygons)
                # # print("BEFORE", individual)
                # # print("AFTER", new_ind)

                # # print("ARE THE SAME X[i] and new_ind=", X[ind]==new_ind.flatten())
                # # print("BEFORE",X[ind])
                # # print("new_ind.shape", new_ind.shape)
                X[ind] = new_ind.flatten()

            pass

        if self.prob_coord_mut > float(np.random.random()):
        # elif True:
            # TODO Change coord (x or y) of a layer
            # # print("I AM USING MUT_COORD",flush=True)
            for ind in range(len(X)):
                individual = X[ind]
                # print("INDIVIDUAL", individual)
                positions_list_layers, active_layers, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # print(f"Active_layers={active_layers}")
                # print(f"Array Poly{array_polygons}")
                if (len(positions_list_layers)==0):
                    continue
                
                position_chosen = (self.choose_one_layer_from_active_layers(positions_list_layers=positions_list_layers, array_polygons=array_polygons))
                polygon = (array_polygons[position_chosen])
                
                coord_chosen = np.random.randint(0, self.max_vertices_polygon)
                # # print("coord_chosen", coord_chosen)
                value_coord_chosen = polygon[coord_chosen]
                # # print("value_coord_chosen", value_coord_chosen)

                new_coord_value = max(0, min(1, np.random.normal(loc=value_coord_chosen, scale=0.1,size=1)[0]))

                # # print("new_coord_value", new_coord_value)

                # # print("......\n")


                polygon[coord_chosen] = new_coord_value
                array_polygons[position_chosen] = polygon

        if self.prob_vertex_mut < float(np.random.random()):
            for ind in range(len(X)):
                individual = X[ind]
                positions_list_layers, active_layers, array_polygons = self.layers_alpha_enables(individual=individual, number_polygons=self.number_polygons)
                # # print(f"pos={positions_list_layers} and len={len(positions_list_layers)}")
                # # print(f"Active_layers={active_layers}")
                # # print(f"Array Poly{array_polygons}")
                if (len(positions_list_layers)==0):
                    continue
                
                position_chosen = (self.choose_one_layer_from_active_layers(positions_list_layers=positions_list_layers, array_polygons=array_polygons))
                polygon = (array_polygons[position_chosen])
                
                vertex_chosen = np.random.randint(0, self.max_vertices_polygon)
                # # print("coord_chosen", vertex_chosen)
                index_start_coord = vertex_chosen*2
                index_end_coord = vertex_chosen*2 + 1
                # # print(f"\tstart_coord {index_start_coord} \n\tend_coord {index_end_coord}")
                start_coord_chosen = polygon[index_start_coord] 
                end_coord_chosen = polygon[index_end_coord] 

                # # print("x_choosen", start_coord_chosen)
                # # print("y_choosen", end_coord_chosen)
                
                start_new_coord_value = max(0, min(1, np.random.normal(loc=start_coord_chosen, scale=0.1,size=1)[0]))
                start_end_coord_value = max(0, min(1, np.random.normal(loc=end_coord_chosen, scale=0.1,size=1)[0]))

                # # print("new_coord_value for x", start_new_coord_value)
                # # print("new_coord_value for y", start_end_coord_value)

                # # print("......\n")


                polygon[index_start_coord] = start_new_coord_value
                polygon[index_end_coord] = start_end_coord_value
                array_polygons[position_chosen] = polygon
            # TODO Change prob_vertex
            pass

        if self.add_remove_vertex < float(np.random.random()):
            # TODO Add or Remove a Vertex
            pass




        return X


    def example(self, problem, X):
        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - oermutation between blocks (layers)
            if r < 0.4:
                layer_1, layer_2 = np.random.randint(2, self.number_polygons )
                # print("layer_1=",layer_1, "layer_2=",layer_2, flush=True)
                perm = np.random.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [c if np.random.random() > prob
                       else np.random.choice(problem.ALPHABET) for c in X[i, 0]]
                X[i, 0] = "".join(mut)