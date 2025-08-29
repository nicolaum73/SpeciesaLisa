from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
import pandas as pd
import numpy as np
import copy


class FileOutput(Output):
    
    def __init__(self, filepath=None, show_ind=False):
        """_summary_

        Args:
            header (List String): List of columns name. If show_ind is True, 
                last_column is the name of the indivuals
            filepath (Path, optional): Filepath to save the fitness values. Defaults to None.
            show_ind (bool, optional): Enable the display of indivudual representation. Defaults to False.
        """
        super().__init__()
        self.filepath = filepath
        # print(self.headers)
        if(filepath is not None):
            previous_df = pd.DataFrame({})
            previous_df.to_csv(self.filepath, index=False)
            # print(f"Filepath_Output = {self.filepath}")
    
        self.show_ind = show_ind
        self.x_mean = Column("x_mean", width=21)
        self.x_std = Column("x_std",  width=21)
        self.x_min = Column("x_min", width=21)
        self.x_max = Column("x_max", width=21)
        # self.columns = [self.n_gen] #[self.n_gen, self.n_eval] # Dont care about n_eval
        
        self.columns += [self.x_mean, self.x_std, self.x_min, self.x_max]

    def update(self, algorithm):
        super().update(algorithm)
        
        # if(self.show_ind):
        #     columns_fitness = self.headers[1:-1]
        # else:
        #     columns_fitness = self.headers[1:]
        

        fitness_df = pd.DataFrame(algorithm.pop.get("F")) #, columns=columns_fitness)
        
        fitness1 = fitness_df.values
        self.x_mean.set(round(np.mean(fitness1),3))
        self.x_min.set(round(np.min(fitness1),3))
        self.x_max.set(round(np.max(fitness1),3))
        self.x_std.set(round(np.std(fitness1),3))
        # self.x_std.set(np.std(fitness_df))
        # self.x_min.set(np.min(fitness_df))
        # self.x_max.set(np.max(fitness_df))
        # self.x_ind.set()
        
        n_gen = pd.DataFrame([algorithm.n_gen] * len(algorithm.pop))
        
        if(self.show_ind):
            # ind_df = [np.array2string(algorithm.pop.get("X").astype("int"),separator=',')]
            # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int),columns=[self.headers[-1]])

            # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int))
            with np.printoptions(linewidth=np.inf,threshold=np.inf):
                # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int))
                # STRING_BAD = '\n '
                individuals_list = list()
                
                for i in algorithm.pop.get("X").astype("float"):
                    string_array= np.array2string(i, separator=",") #.replace(STRING_BAD,"")
                    individuals_list.append(string_array)
                ind_df = pd.DataFrame(individuals_list)
            # X_df
            output_df = pd.concat([n_gen,fitness_df,ind_df],axis=1)
        else:
            output_df = pd.concat([n_gen,fitness_df],axis=1)
        if(self.filepath is not None):        
            output_df.to_csv(self.filepath, mode="a", index=False, header=False)


# def saving_res_output(result,filepath_file=None, show_ind=False, display_verbose=False):
#     """Given the pymoo result, display the fitness and (optional) the indiviual
#         If filepath_file is different to None, it save it in the given filepath as csv. 

#     Args:
#         result (pymoo.core.result.Result): The result of the minimize problem
#         filepath_file (str, optional):A filepath to save the output. Defaults to None.
#         show_ind (bool, optional): Save/Display the indvidual representation. Defaults to False.
#     """
#     # Without this np.printoptions, the output is (therehold) summarize [1,0, ... 0,11]
#     # and (linewidth) added \n 
#     with np.printoptions(linewidth=np.inf,threshold=np.inf):
#         n_gen = pd.DataFrame([result.algorithm.n_gen]*(result.F.shape[0]),columns=["gen"])
#         fitness_df = pd.DataFrame(result.F, columns=["fitness1","fitness2"])
#         if(show_ind):
#             # ind_df = pd.DataFrame([[str(x)] for x in result.X.astype(int)],columns=["Individuo"])
#             # # STRING_BAD = '\n '
#             individuals_list = list()
#             for i in result.X.astype(int):
#                 # print(i)
#                 # string_array= np.array2string(i, separator=",", max_line_width=np.inf, threshold=sys.maxsize).replace(STRING_BAD,"")
#                 string_array= np.array2string(i, separator=",") #, max_line_width=np.inf, threshold=sys.maxsize)
#                 individuals_list.append(string_array)
#             ind_df = pd.DataFrame(individuals_list, columns=["Ind"])
#             output_df = pd.concat([n_gen,fitness_df,ind_df],axis=1)
#         else:
#             output_df = pd.concat([n_gen,fitness_df],axis=1)
#         if(filepath_file is not None):
#             output_df.to_csv(filepath_file, index=False)
#         if(display_verbose):
#             print(output_df)
        
#         # Return output
#         # return output_df 


