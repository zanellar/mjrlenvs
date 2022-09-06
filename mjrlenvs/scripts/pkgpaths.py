
import os
import mjrlenvs

 
class PkgPath(object):

    PACKAGE_PATH = mjrlenvs.__path__[0] 

    def envdata(file=""):
        return os.path.join(PkgPath.PACKAGE_PATH, os.pardir, "data/envdesc", file) 

    def trainingdata(file=""):
        return os.path.join(PkgPath.PACKAGE_PATH, os.pardir, "data/train", file) 

    def testingdata(file=""):
        return os.path.join(PkgPath.PACKAGE_PATH, os.pardir, "data/test", file) 
 