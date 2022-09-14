
import os
import mjrlenvs

 
class PkgPath():

    _PACKAGE_PATH = mjrlenvs.__path__[0]  

    ENV_DESC_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/envdesc")  
    OUT_TRAIN_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/train")  
    OUT_TEST_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/test")  
 