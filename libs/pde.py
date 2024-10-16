from .utils import *

class PDEData(object):
    """
    A template for defining the test problems.
    Data generation pipeline are implemented within the object.
    Models are trained outside the object. 
    Model testing functions are implemented within the object.

    Pipeline: 
            1) initialize a PDEData object
            2) get training data from PDEData
            3) train models with the data 
            4) test the models with PDEData.test_model
    """
    def __init__(self):
        super().__init__()
    
    def get_stencil(self):
        pass

    def get_prolongation(self):
        pass

    def generate_data(self):
        pass

    def geometric_solver(self):
        pass

    def extend_hierarchy(self):
        pass
    