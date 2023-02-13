
import os
import mjrlenvs

 
class PkgPath():

    '''
    This class is used to store the paths to the folders in the package.
    '''

    _PACKAGE_PATH = mjrlenvs.__path__[0]  

    ENV_DESC_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data", "envdesc")  
    OUT_TRAIN_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data", "train")  
    OUT_TEST_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data", "test")  
    PLOT_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data", "plots")  
  
 