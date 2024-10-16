from ..libs.pde import *

class Elasticity(PDEData):
    def __init__(self,
                 grid_size : int,
                 k : list,
                 epsilonList : list,
                 thetaList : list,
                 dataset_option : str,
                 debug = False,
                 same_vecs = False,
                 ):
        """
        grid_size : size of training mesh
        k : list of integers
            k[0] contains the number of smooth vectors in level 2
            k[1] contains the number of smooth vectors in level 3
        dataset_option available : {'fixed E'}
        same_vecs : if true, smooth vectors on level 3 will be the same as those on level 2
        
        Data generation has only been implemented for <= 3-level training

        """
        
        super().__init__()
        self.n = grid_size
        self.epsList = epsilonList
        self.thetaList = thetaList
        self.k = k
        self.dataset_option = dataset_option
    









